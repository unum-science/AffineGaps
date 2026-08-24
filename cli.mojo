"""
Native command line for the AffineGaps kernels, reaching every recurrence the library ships.

One verb per recurrence, because folding takes one sequence where alignment and cofolding take two,
and because alignment's gap is affine while cofolding's is linear. Under a single command those two
gap models would sit side by side in one help text inviting a reader to mix them.

`main` lives here rather than beside the bindings because Mojo refuses to emit a shared library
from a module that defines it.

The flag surface mirrors `affinegaps.py` so the two can be differentially tested, except that this
binary is the compiled backend and so has no `--backend` to choose one.
"""

from std.ffi import c_int, external_call
from std.io import FileDescriptor
from std.sys import argv, exit
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from alignment import (
    AffineGapCosts,
    AlignmentMode,
    AlignmentResult,
    DEFAULT_GAP_EXTENSION,
    DEFAULT_GAP_OPENING,
    DEFAULT_LEAF_CELLS,
    colorize,
    default_proteins_matrix,
    device_align,
    serial_align,
)
from cofolding import SankoffScoring, device_cofold, serial_cofold
from common import (
    DEFAULT_PROTEINS_ALPHABET,
    DEFAULT_RNA_ALPHABET,
    Executor,
    Placement,
    SubstitutionDType,
    SymbolDType,
    hardware_threads,
    translate,
    uniform_matrix,
)
from errors import AffineGapsError, ErrorKind
from folding import device_fold, serial_fold

# region Usage

comptime USAGE = """Usage: affinegaps VERB [ARGUMENTS]

Verbs:
  align   Gotoh alignment of two sequences, global or local
  fold    Zuker minimum free energy folding of one RNA sequence
  cofold  Sankoff simultaneous alignment and folding of two RNA sequences

Run `affinegaps VERB --help` for the arguments a verb takes."""
"""What a bare invocation or an unknown verb prints."""

comptime USAGE_ALIGN = """Usage: affinegaps align FIRST SECOND [OPTIONS]

Gotoh affine-gap alignment. Reconstructs the alignment, not just the score.

  --local            Smith-Waterman instead of Needleman-Wunsch
  --match N          Uniform match score, instead of scaled BLOSUM62
  --mismatch N       Uniform mismatch score, instead of scaled BLOSUM62
  --open N           Gap opening penalty
  --extend N         Gap extension penalty
  --device cpu|gpu   Where to run it
  --gpu-id N         Which accelerator; implies --device gpu
  --threads N        Host threads the traceback may fork across
  --format human|json
  --color auto|always|never
  --verbose          Report placement and throughput on stderr
  --help"""
"""What `align --help` prints. Every flag named here is one the parser accepts."""

comptime USAGE_FOLD = """Usage: affinegaps fold SEQUENCE [OPTIONS]

Zuker minimum free energy folding over the Turner nearest-neighbour model.

  --device cpu|gpu   Where to run it
  --gpu-id N         Which accelerator; implies --device gpu
  --format human|json
  --color auto|always|never
  --verbose          Report placement and throughput on stderr
  --help"""
"""What `fold --help` prints."""

comptime USAGE_COFOLD = """Usage: affinegaps cofold FIRST SECOND [OPTIONS]

Sankoff simultaneous alignment and folding. Credits a base pair only where both can form it.

  --match N          Score for aligning two equal bases
  --mismatch N       Score for aligning two different bases
  --gap N            Linear gap cost; Sankoff has no affine model
  --device cpu|gpu   Where to run it
  --gpu-id N         Which accelerator; implies --device gpu
  --format human|json
  --color auto|always|never
  --verbose          Report placement and throughput on stderr
  --help"""
"""What `cofold --help` prints."""

# endregion Usage

# region Options


@fieldwise_init
struct Format(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """How the answer is printed."""

    var identifier: UInt8
    """Which format this names."""
    comptime HUMAN = Self(0)
    """Labelled rows, one per value."""
    comptime JSON = Self(1)
    """One object on one line, for a caller that will parse it."""


@fieldwise_init
struct Coloring(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Whether an alignment's rows are coloured."""

    var identifier: UInt8
    """Which choice this names."""
    comptime AUTO = Self(0)
    """Colour only when standard output is a terminal."""
    comptime ALWAYS = Self(1)
    """Colour regardless, for a caller that will render the escapes."""
    comptime NEVER = Self(2)
    """Never colour."""


@fieldwise_init
struct Requested(ImplicitlyCopyable, Movable):
    """What the placement flags said, before they become a `Placement`.

    A flag loop fills fields one at a time and cannot go through the factories, so what it names is
    collected here and turned into a placement once, after the loop has seen everything.
    """

    var executor: Optional[Executor]
    """The sweep the caller named, absent when they named none."""
    var gpu_id: Optional[Int]
    """The accelerator the caller named, absent when they named none."""
    var threads: Int
    """The width the caller asked for, defaulting to every thread this process may use."""

    @staticmethod
    def default() -> Self:
        """Nothing named, across every thread this process may use."""
        return Self(None, None, hardware_threads())


def placement_of(request: Requested) raises AffineGapsError -> Placement:
    """Turns what the flags named into a placement.

    Naming an accelerator is asking for one, so it settles a device the caller left unnamed and
    contradicts one they named as the host.
    """
    var on_host = request.executor and request.executor.value() == Executor.HOST
    if request.gpu_id:
        if on_host:
            raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, "--gpu-id with --device cpu")
        return Placement.device(request.gpu_id.value(), request.threads)
    if request.executor and request.executor.value() == Executor.DEVICE:
        return Placement.device(0, request.threads)
    return Placement.host(request.threads)


@fieldwise_init
struct Options(ImplicitlyCopyable, Movable):
    """The settings every verb shares, so they cannot drift between verbs."""

    var request: Requested
    """Where the caller asked the sweep to run, and on which of the machine's resources."""
    var format: Format
    """How the answer is printed."""
    var coloring: Coloring
    """Whether to colour an alignment."""
    var verbose: Bool
    """Whether to report placement and throughput on stderr."""

    @staticmethod
    def default() -> Self:
        """Host, human-readable, colour when the terminal wants it, and quiet."""
        return Self(Requested.default(), Format.HUMAN, Coloring.AUTO, False)


def parse_int(text: String) -> Optional[Int]:
    """Reads a command-line integer, returning nothing when the text is not one."""
    try:
        var parsed: Optional[Int] = Int(text)
        return parsed
    except:
        return None


def wants_color(coloring: Coloring) -> Bool:
    """Whether to colour, which `auto` answers by asking whether standard output is a terminal."""
    if coloring == Coloring.NEVER:
        return False
    if coloring == Coloring.ALWAYS:
        return True
    return Int(external_call["isatty", c_int](c_int(1))) == 1


def shared_flag(flag: String, value: String, mut options: Options) raises AffineGapsError -> Int:
    """Consumes one shared flag, answering how many arguments it took, or zero if it is not ours."""
    if flag == "--verbose":
        options.verbose = True
        return 1
    if flag == "--device":
        if value == "gpu":
            options.request.executor = Executor.DEVICE
        elif value == "cpu":
            options.request.executor = Executor.HOST
        else:
            raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, String("device ", value))
        return 2
    if flag == "--gpu-id":
        var index = parse_int(value)
        if not index or index.value() < 0:
            raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, String("gpu-id ", value))
        options.request.gpu_id = index.value()
        return 2
    if flag == "--format":
        if value == "json":
            options.format = Format.JSON
        elif value == "human":
            options.format = Format.HUMAN
        else:
            raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, String("format ", value))
        return 2
    if flag == "--color":
        if value == "always":
            options.coloring = Coloring.ALWAYS
        elif value == "never":
            options.coloring = Coloring.NEVER
        elif value == "auto":
            options.coloring = Coloring.AUTO
        else:
            raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, String("color ", value))
        return 2
    return 0


# endregion Options

# region Reporting


def quote(text: String) -> String:
    """A JSON string. Every value reaching here is alphabet-checked or dot-bracket, so no escaping."""
    return String('"', text, '"')


def rounded(value: Float64, places: Int) -> String:
    """A float at a fixed number of decimals, because the default spelling prints every digit it has."""
    var scale = Float64(10) ** places
    var whole = Int(value)
    var fraction = Int(round((value - Float64(whole)) * scale))
    if fraction >= Int(scale):
        whole += 1
        fraction -= Int(scale)
    var digits = String(fraction)
    return String(whole, ".", "0" * (places - digits.byte_length()), digits)


def report_placement(placement: Placement, options: Options, cells: Int, nanoseconds: Int):
    """Which backend ran and how fast, on stderr so a piped payload stays parseable.

    Elapsed is printed beside the rate because one short problem measures mostly dispatch overhead,
    and a bare cell-update rate would invite reading more into it than it says.
    """
    var errors = FileDescriptor(2)
    # The index only names something when there is an accelerator for it to name.
    var device = String("gpu:", placement.gpu_id) if placement.executor == Executor.DEVICE else String("cpu")

    var seconds = Float64(nanoseconds) / 1e9
    var rate = Float64(cells) / seconds / 1e6 if seconds > 0 else 0.0
    print("  backend:     mojo on ", device, sep="", file=errors)
    print("  cells:       ", cells, sep="", file=errors)
    print("  elapsed:     ", rounded(Float64(nanoseconds) / 1e6, 3), " ms", sep="", file=errors)
    print("  throughput:  ", rounded(rate, 2), " MCUPS", sep="", file=errors)


# endregion Reporting

# region Verbs


def aligned_pair[
    mode: AlignmentMode
](
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    alphabet: String,
    placement: Placement,
) raises -> AlignmentResult:
    """Runs one pair where the caller asked, which is the only thing the verb decides."""
    if placement.executor == Executor.DEVICE:
        return device_align[mode](
            DeviceContext(device_id=placement.gpu_id),
            first,
            second,
            substitutions,
            alphabet_size,
            scoring,
            alphabet,
            DEFAULT_LEAF_CELLS,
            placement,
        )
    return serial_align[mode](first, second, substitutions, alphabet_size, scoring, alphabet)


def run_align(arguments: List[String], mut options: Options) raises -> Int:
    """Aligns two sequences and prints the report."""
    if len(arguments) < 2 or arguments[0] == "--help":
        print(USAGE_ALIGN)
        return 0 if len(arguments) > 0 and arguments[0] == "--help" else 2

    var first_text = arguments[0]
    var second_text = arguments[1]
    var mode = AlignmentMode.GLOBAL
    var opening = Int(DEFAULT_GAP_OPENING)
    var extension = Int(DEFAULT_GAP_EXTENSION)
    var match_score = Optional[Int]()
    var mismatch_score = Optional[Int]()

    var index = 2
    while index < len(arguments):
        var flag = arguments[index]
        var value = arguments[index + 1] if index + 1 < len(arguments) else String("")
        var taken = shared_flag(flag, value, options)
        if taken > 0:
            index += taken
            continue
        if flag == "--help":
            print(USAGE_ALIGN)
            return 0
        if flag == "--local":
            mode = AlignmentMode.LOCAL
            index += 1
            continue
        if index + 1 >= len(arguments):
            raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, String(flag, " needs a value"))
        var number = parse_int(value)
        var numeric = flag == "--open" or flag == "--extend" or flag == "--match" or flag == "--mismatch"
        if (numeric or flag == "--threads") and not number:
            raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, String(flag, " needs an integer [", value, "]"))
        if flag == "--open":
            opening = number.value()
        elif flag == "--extend":
            extension = number.value()
        elif flag == "--match":
            match_score = number
        elif flag == "--mismatch":
            mismatch_score = number
        elif flag == "--threads":
            if not number or number.value() < 0:
                raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, String("threads ", value))
            options.request.threads = number.value()
        else:
            raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, flag)
        index += 2

    if Bool(match_score) != Bool(mismatch_score):
        raise AffineGapsError(ErrorKind.INVALID_SCORING, "match without mismatch")

    var placement = placement_of(options.request)

    var alphabet = String(DEFAULT_PROTEINS_ALPHABET)
    var alphabet_size = alphabet.byte_length()
    var scoring = AffineGapCosts.checked(Int32(opening), Int32(extension))
    var substitutions = default_proteins_matrix() if not match_score else uniform_matrix(
        alphabet_size, match_score.value(), mismatch_score.value()
    )
    var left = translate(first_text, alphabet)
    var right = translate(second_text, alphabet)

    var started = perf_counter_ns()
    var result = aligned_pair[AlignmentMode.LOCAL](
        left, right, substitutions, alphabet_size, scoring, alphabet, placement
    ) if mode == AlignmentMode.LOCAL else aligned_pair[AlignmentMode.GLOBAL](
        left, right, substitutions, alphabet_size, scoring, alphabet, placement
    )
    var elapsed = perf_counter_ns() - started

    if options.format == Format.JSON:
        print(
            String(
                '{{"operation": "align", "mode": {}, "first": {}, "second": {},'
                ' "first_gapped": {}, "second_gapped": {}, "score": {},'
                ' "backend": "mojo", "device": {}, "gpu_id": {}, "threads": {}}}'
            ).format(
                quote("local" if mode == AlignmentMode.LOCAL else "global"),
                quote(first_text),
                quote(second_text),
                quote(result.first_gapped),
                quote(result.second_gapped),
                Int(result.score),
                quote("gpu" if placement.executor == Executor.DEVICE else "cpu"),
                placement.gpu_id,
                placement.threads,
            )
        )
    else:
        var first_shown = result.first_gapped
        var second_shown = result.second_gapped
        if wants_color(options.coloring):
            var painted = colorize(result.first_gapped, result.second_gapped)
            first_shown = painted[0]
            second_shown = painted[1]
        print(String("Sequence 1:  {}").format(first_text))
        print(String("Sequence 2:  {}").format(second_text))
        print(String("Alignment 1: {}").format(first_shown))
        print(String("Alignment 2: {}").format(second_shown))
        print(String("Score:       {}").format(Int(result.score)))

    if options.verbose:
        report_placement(placement, options, len(left) * len(right), elapsed)
    return 0


def run_fold(arguments: List[String], mut options: Options) raises -> Int:
    """Folds one RNA sequence and prints the report."""
    if len(arguments) < 1 or arguments[0] == "--help":
        print(USAGE_FOLD)
        return 0 if len(arguments) > 0 else 2

    var sequence_text = arguments[0]
    var index = 1
    while index < len(arguments):
        var flag = arguments[index]
        var value = arguments[index + 1] if index + 1 < len(arguments) else String("")
        var taken = shared_flag(flag, value, options)
        if taken > 0:
            index += taken
            continue
        if flag == "--help":
            print(USAGE_FOLD)
            return 0
        raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, flag)

    var placement = placement_of(options.request)
    var started = perf_counter_ns()
    var result = device_fold(
        DeviceContext(device_id=placement.gpu_id), sequence_text
    ) if placement.executor == Executor.DEVICE else serial_fold(sequence_text)
    var elapsed = perf_counter_ns() - started
    var energy = Float64(Int(result.decikcal)) / 10.0

    if options.format == Format.JSON:
        print(
            String(
                '{{"operation": "fold", "sequence": {}, "structure": {}, "energy_kcal_per_mol": {},'
                ' "backend": "mojo", "device": {}, "gpu_id": {}}}'
            ).format(
                quote(sequence_text),
                quote(result.structure),
                energy,
                quote("gpu" if placement.executor == Executor.DEVICE else "cpu"),
                placement.gpu_id,
            )
        )
    else:
        print(String("Sequence:  {}").format(sequence_text))
        print(String("Structure: {}").format(result.structure))
        print(String("Energy:    {} kcal/mol").format(energy))

    if options.verbose:
        var length = sequence_text.byte_length()
        report_placement(placement, options, length * length, elapsed)
    return 0


def run_cofold(arguments: List[String], mut options: Options) raises -> Int:
    """Aligns and folds two RNA sequences together, and prints the report."""
    if len(arguments) < 2 or arguments[0] == "--help":
        print(USAGE_COFOLD)
        return 0 if len(arguments) > 0 and arguments[0] == "--help" else 2

    var first_text = arguments[0]
    var second_text = arguments[1]
    var match_score = 2
    var mismatch_score = -1
    var gap = -2

    var index = 2
    while index < len(arguments):
        var flag = arguments[index]
        var value = arguments[index + 1] if index + 1 < len(arguments) else String("")
        var taken = shared_flag(flag, value, options)
        if taken > 0:
            index += taken
            continue
        if flag == "--help":
            print(USAGE_COFOLD)
            return 0
        if index + 1 >= len(arguments):
            raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, String(flag, " needs a value"))
        var number = parse_int(value)
        var numeric = flag == "--match" or flag == "--mismatch" or flag == "--gap"
        if numeric and not number:
            raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, String(flag, " needs an integer [", value, "]"))
        if flag == "--match":
            match_score = number.value()
        elif flag == "--mismatch":
            mismatch_score = number.value()
        elif flag == "--gap":
            gap = number.value()
        else:
            raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, flag)
        index += 2

    var placement = placement_of(options.request)
    var alphabet = String(DEFAULT_RNA_ALPHABET)
    var scoring = SankoffScoring(Int32(gap))
    var started = perf_counter_ns()
    var result = device_cofold(
        DeviceContext(device_id=placement.gpu_id),
        first_text,
        second_text,
        alphabet,
        scoring,
        match_score,
        mismatch_score,
    ) if placement.executor == Executor.DEVICE else serial_cofold(
        first_text, second_text, alphabet, scoring, match_score, mismatch_score
    )
    var elapsed = perf_counter_ns() - started

    if options.format == Format.JSON:
        print(
            String(
                '{{"operation": "cofold", "first": {}, "second": {},'
                ' "first_gapped": {}, "second_gapped": {}, "structure": {}, "score": {},'
                ' "backend": "mojo", "device": {}, "gpu_id": {}}}'
            ).format(
                quote(first_text),
                quote(second_text),
                quote(result.gapped_first),
                quote(result.gapped_second),
                quote(result.structure),
                Int(result.score),
                quote("gpu" if placement.executor == Executor.DEVICE else "cpu"),
                placement.gpu_id,
            )
        )
    else:
        print(String("Sequence 1: {}").format(first_text))
        print(String("Sequence 2: {}").format(second_text))
        print(String("Structure:  {}").format(result.structure))
        print(String("Score:      {}").format(Int(result.score)))

    if options.verbose:
        var cells = first_text.byte_length() * second_text.byte_length()
        report_placement(placement, options, cells * cells, elapsed)
    return 0


# endregion Verbs


def main():
    """Selects a verb and runs it, or prints the verb list.

    The status a verb computes becomes the process status, and a raised error prints to stderr, so
    a shell can branch on either and a piped payload stays parseable.
    """
    var given = argv()
    if len(given) < 2:
        print(USAGE, file=FileDescriptor(2))
        exit(2)

    var verb = String(given[1])
    if verb == "--help":
        print(USAGE)
        return

    var rest = List[String]()
    for index in range(2, len(given)):
        rest.append(String(given[index]))

    var options = Options.default()
    var status = 0
    try:
        if verb == "align":
            status = run_align(rest, options)
        elif verb == "fold":
            status = run_fold(rest, options)
        elif verb == "cofold":
            status = run_cofold(rest, options)
        else:
            print(USAGE, file=FileDescriptor(2))
            status = 2
    except error:
        print(String("Error: ", error), file=FileDescriptor(2))
        status = 1
    if status != 0:
        exit(status)
