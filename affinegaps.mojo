"""
Python bindings for the AffineGaps kernels, and the root of the shared-library build.

This module holds nothing but argument marshalling and the module table. Each recurrence lives
beside it and is imported here to be bound:

    common.mojo      symbol codes, device staging, the shared-memory budget
    alignment.mojo   Gotoh, with Hirschberg traceback on the GPU
    cofolding.mojo   Sankoff simultaneous alignment and folding
    folding.mojo     Zuker minimum free energy folding
    turner.mojo      the nearest-neighbour energy parameters

Sibling modules resolve from the same directory, so the build names only this file.

## Usage

```bash
pixi run build      # build/affinegaps_mojo.so, importable from Python
pixi run build-cli  # build/affinegaps, the native command line
pixi run test       # the differential suite against affinegaps.py
```
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceContext

from common import (
    DEFAULT_PROTEINS_ALPHABET, DEFAULT_RNA_ALPHABET, Executor, FALLBACK_LETTER, OFFSET_DTYPE,
    Placement, SCORE_DTYPE, SUBSTITUTION_DTYPE, SYMBOL_DTYPE, hardware_threads, translate,
    uniform_matrix,
)
from errors import AffineGapsError, ErrorKind
from folding import FoldResult, device_fold, serial_fold
from cofolding import (
    CofoldResult,
    DEFAULT_GAP,
    DEFAULT_MATCH,
    DEFAULT_MISMATCH,
    SankoffScoring,
    device_cofold,
    serial_cofold,
)
from alignment import (
    ALL_MODES, AffineGapCosts, AlignmentMode, DEFAULT_GAP_EXTENSION, DEFAULT_GAP_OPENING,
    DEFAULT_TILE_CELLS, DEVICE_STORED_CELLS, GapRun, Layer, MAX_BAND_LENGTH,
    Sweep, SweepBuffers, SweepHalf, colorize, default_proteins_matrix, device_local_extremum,
    direct_alignments, expand_path, hirschberg_path_gpu, hirschberg_window, local_extremum, score_path,
    serial_align, serial_score, sweep_buffers, sweep_level, wavefront_scores,
)

# region Python Bindings


def optional_int(value: PythonObject) -> Optional[Int]:
    """Reads a Python argument that may be `None`, without asking Python what `None` is."""
    try:
        var present: Optional[Int] = Int(String(value))
        return present
    except:
        return None


def scoring_from(gaps: PythonObject) raises -> AffineGapCosts:
    """Reads the two gap penalties off a caller's gap-cost record by name."""
    var opening = optional_int(gaps.open).or_else(Int(DEFAULT_GAP_OPENING))
    var extension = optional_int(gaps.extend).or_else(Int(DEFAULT_GAP_EXTENSION))
    return AffineGapCosts(Int32(opening), Int32(extension))


def matrix_from(substitution: PythonObject, alphabet_size: Int) raises -> List[Scalar[SUBSTITUTION_DTYPE]]:
    """BLOSUM62 unless the caller passed a uniform-cost record.

    No record at all means the default table. A tabulated record never reaches here, because the
    caller refuses it before the boundary; these kernels carry only the default.
    """
    var given_match: Optional[Int] = None
    var given_mismatch: Optional[Int] = None
    try:
        given_match = optional_int(substitution.match)
        given_mismatch = optional_int(substitution.mismatch)
    except:
        return default_proteins_matrix()
    if not given_match:
        return default_proteins_matrix()
    return uniform_matrix(alphabet_size, given_match.value(), given_mismatch.value())


def protein_defaults(gaps: PythonObject) raises -> Tuple[String, Int, AffineGapCosts]:
    """The alphabet, its size and the gap costs a binding call resolves to.

    The substitution table stays out of this because a `List` cannot be copied out of a tuple,
    and it is the one part that needs the alphabet size first anyway.
    """
    var alphabet = String(DEFAULT_PROTEINS_ALPHABET)
    return (alphabet, alphabet.byte_length(), scoring_from(gaps))


def alignment_triple(first_gapped: String, second_gapped: String, score: Int32) raises -> PythonObject:
    """The shape every alignment entry point returns: both gapped strings and the score."""
    var triple = Python().list()
    triple.append(PythonObject(first_gapped))
    triple.append(PythonObject(second_gapped))
    triple.append(PythonObject(Int(score)))
    return triple


def gotoh_score[
    mode: AlignmentMode
](first: PythonObject, second: PythonObject, substitution: PythonObject, gaps: PythonObject,) raises -> PythonObject:
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)
    return PythonObject(Int(serial_score[mode](left, right, substitutions, alphabet_size, scoring)))


def gotoh_alignment[
    mode: AlignmentMode
](first: PythonObject, second: PythonObject, substitution: PythonObject, gaps: PythonObject,) raises -> PythonObject:
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)
    var result = serial_align[mode](left, right, substitutions, alphabet_size, scoring, alphabet)
    var triple = alignment_triple(result.first_gapped, result.second_gapped, result.score)
    return triple


def python_length(value: PythonObject) raises -> Int:
    return Int(String(value.__len__()))


@fieldwise_init
struct BatchTape(Movable):
    """Both sides of a batch on one tape, which is the shape a kernel can take a pointer to."""

    var sequences: List[Scalar[SYMBOL_DTYPE]]
    """Every sequence concatenated, first and second of each pair alternating."""
    var offsets: List[Scalar[OFFSET_DTYPE]]
    """Where each sequence begins, so a block can find its own pair."""


def pack_batch(firsts: PythonObject, seconds: PythonObject, alphabet: String) raises -> BatchTape:
    """Both sides of a batch concatenated onto one tape, with offsets marking where each begins.

    A kernel takes a pointer and a length, so the batch crosses as one allocation rather than as a
    list of them; the offsets are what let a block find its own pair.
    """
    var pairs = paired_length(firsts, seconds)
    var sequences = List[Scalar[SYMBOL_DTYPE]]()
    var offsets = List[Scalar[OFFSET_DTYPE]]()
    offsets.append(0)
    for index in range(pairs):
        var left = translate(String(firsts[index]), alphabet)
        if len(left) > MAX_BAND_LENGTH:
            raise AffineGapsError(ErrorKind.SEQUENCE_TOO_LONG, "shared-memory band")
        sequences.extend(left^)
        offsets.append(Scalar[OFFSET_DTYPE](len(sequences)))
        var right = translate(String(seconds[index]), alphabet)
        sequences.extend(right^)
        offsets.append(Scalar[OFFSET_DTYPE](len(sequences)))
    return BatchTape(sequences^, offsets^)


def gotoh_scores_batch[
    mode: AlignmentMode
](
    firsts: PythonObject,
    seconds: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
    placement: Placement,
) raises -> PythonObject:
    """Scores a whole batch on the GPU, one thread block per pair.

    Both sides are packed into one Arrow-like tape — every sequence concatenated into a flat
    symbol buffer, with an offset array carrying two entries per pair — so a batch reaches the
    device as two buffers rather than as one transfer per sequence.
    """
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)

    var pairs = python_length(firsts)
    if pairs != python_length(seconds):
        raise AffineGapsError(ErrorKind.LENGTH_MISMATCH, "batch sides")
    if pairs == 0:
        return Python().list()

    var tape = pack_batch(firsts, seconds, alphabet)

    var ctx = DeviceContext(device_id=placement.gpu_id)
    var scores = wavefront_scores[mode](ctx, tape.sequences, tape.offsets, substitutions, alphabet_size, scoring)
    var output = Python().list()
    for index in range(len(scores)):
        output.append(PythonObject(Int(scores[index])))
    return output


def gotoh_alignments_batch[
    mode: AlignmentMode
](
    firsts: PythonObject,
    seconds: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
    placement: Placement,
) raises -> PythonObject:
    """Aligns a whole batch on the GPU, returning `[first_gapped, second_gapped, score]` triples.

    Packed into the same Arrow-like tape the scoring batch uses.
    """
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)

    var pairs = python_length(firsts)
    if pairs != python_length(seconds):
        raise AffineGapsError(ErrorKind.LENGTH_MISMATCH, "batch sides")
    if pairs == 0:
        return Python().list()

    var tape = pack_batch(firsts, seconds, alphabet)

    var ctx = DeviceContext(device_id=placement.gpu_id)
    var aligned = direct_alignments[mode](ctx, tape.sequences, tape.offsets, substitutions, alphabet, scoring)
    var output = Python().list()
    for index in range(len(aligned)):
        var triple = alignment_triple(aligned[index].first_gapped, aligned[index].second_gapped, aligned[index].score)
        output.append(triple)
    return output


def combined_alphabet(first: String, second: String) -> String:
    """The distinct characters of both strings, so unit-cost alignment needs no fixed alphabet."""
    var seen = List[Bool](length=256, fill=False)
    var letters = List[UInt8]()
    var first_bytes = first.as_bytes()
    var second_bytes = second.as_bytes()
    for index in range(len(first_bytes)):
        if not seen[Int(first_bytes[index])]:
            seen[Int(first_bytes[index])] = True
            letters.append(first_bytes[index])
    for index in range(len(second_bytes)):
        if not seen[Int(second_bytes[index])]:
            seen[Int(second_bytes[index])] = True
            letters.append(second_bytes[index])
    if len(letters) == 0:
        letters.append(FALLBACK_LETTER)
    return String(unsafe_from_utf8=letters)


def levenshtein_alignment(first: PythonObject, second: PythonObject) raises -> PythonObject:
    """Unit-cost edit distance, expressed as the global recurrence with linear gaps.

    Setting both penalties to minus one and the substitution scores to zero and minus one turns
    the maximizing Gotoh recurrence into the negated Levenshtein minimization, tie-break chain
    included, so this needs no kernel of its own.
    """
    var left = String(first)
    var right = String(second)
    for byte in left.as_bytes():
        if byte >= 0x80:
            raise AffineGapsError(ErrorKind.NOT_ASCII, "unit-cost alignment")
    for byte in right.as_bytes():
        if byte >= 0x80:
            raise AffineGapsError(ErrorKind.NOT_ASCII, "unit-cost alignment")
    var alphabet = combined_alphabet(left, right)
    var alphabet_size = alphabet.byte_length()
    var substitutions = uniform_matrix(alphabet_size, 0, -1)
    var scoring = AffineGapCosts(Int32(-1), Int32(-1))
    var encoded_left = translate(left, alphabet)
    var encoded_right = translate(right, alphabet)
    var result = serial_align[AlignmentMode.GLOBAL](
        encoded_left, encoded_right, substitutions, alphabet_size, scoring, alphabet
    )
    var triple = Python().list()
    triple.append(PythonObject(result.first_gapped))
    triple.append(PythonObject(result.second_gapped))
    triple.append(PythonObject(-Int(result.score)))
    return triple


def zuker_fold(sequence: PythonObject, requested: PythonObject) raises -> PythonObject:
    """Minimum free energy folding of one RNA sequence over the Turner nearest-neighbour model.

    Returns the dot-bracket structure and the free energy in kilocalories per mole. The recurrence
    works in integer decikilocalories, so the same sequence always folds to the same answer.
    """
    var text = String(sequence)
    var placement = placement_from(requested)
    var outcome: FoldResult
    if String(requested.device) == "gpu":
        var ctx = DeviceContext(device_id=placement.gpu_id)
        outcome = device_fold(ctx, text)
    else:
        outcome = serial_fold(text)

    var couple = Python().list()
    couple.append(PythonObject(outcome.structure))
    couple.append(PythonObject(Float64(Int(outcome.decikcal)) / 10.0))
    return couple


def sankoff_cofold(
    first: PythonObject,
    second: PythonObject,
    gap: PythonObject,
    match_reward: PythonObject,
    mismatch_penalty: PythonObject,
    requested: PythonObject,
) raises -> PythonObject:
    """Exact simultaneous alignment and folding of two RNA sequences.

    Returns both gapped sequences, the dot-bracket structure they agree on, and the score. Memory
    grows as the fourth power of the sequence length, so this is bounded to a few hundred bases.
    """
    var scoring = SankoffScoring(Int32(optional_int(gap).or_else(Int(DEFAULT_GAP))))
    var match_score = optional_int(match_reward).or_else(Int(DEFAULT_MATCH))
    var mismatch_score = optional_int(mismatch_penalty).or_else(Int(DEFAULT_MISMATCH))
    var alphabet = String(DEFAULT_RNA_ALPHABET)
    var left = String(first)
    var right = String(second)

    var placement = placement_from(requested)
    var outcome: CofoldResult
    if String(requested.device) == "gpu":
        var ctx = DeviceContext(device_id=placement.gpu_id)
        outcome = device_cofold(ctx, left, right, alphabet, scoring, match_score, mismatch_score)
    else:
        outcome = serial_cofold(left, right, alphabet, scoring, match_score, mismatch_score)

    var quadruple = Python().list()
    quadruple.append(PythonObject(outcome.gapped_first))
    quadruple.append(PythonObject(outcome.gapped_second))
    quadruple.append(PythonObject(outcome.structure))
    quadruple.append(PythonObject(Int(outcome.score)))
    return quadruple


def colorize_alignment(first_gapped: PythonObject, second_gapped: PythonObject) raises -> PythonObject:
    """Wraps each column in an ANSI colour: green for a match, red for a mismatch, dim for a gap."""
    var painted = colorize(String(first_gapped), String(second_gapped))
    var pair = Python().list()
    pair.append(PythonObject(painted[0]))
    pair.append(PythonObject(painted[1]))
    return pair


def needleman_wunsch_gotoh_alignment_linear(
    first: PythonObject,
    second: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
) raises -> PythonObject:
    """Global alignment in linear space, splitting rows and joining halves Myers-Miller style.

    `tile_cells` is the only knob: subproblems at or below it are solved outright, above it they
    are split. Raising it past the whole matrix collapses to a single direct traceback, which is
    what makes the two paths comparable.
    """
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var cells = DEFAULT_TILE_CELLS
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)

    var path_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
    var path_entries = List[Layer](length=len(left) + 1, fill=Layer.ALIGNING)
    hirschberg_window(
        left,
        right,
        0,
        len(left),
        0,
        len(right),
        substitutions,
        alphabet_size,
        scoring,
        cells,
        path_columns,
        path_entries,
    )
    var score = score_path(left, right, path_columns, path_entries, substitutions, alphabet_size, scoring, len(left))
    var expanded = expand_path(left, right, path_columns, path_entries, alphabet, AlignmentMode.GLOBAL, 0, len(left))
    var triple = alignment_triple(expanded[0], expanded[1], score)
    return triple


def needleman_wunsch_gotoh_alignment_linear_gpu(
    first: PythonObject,
    second: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
    placement: Placement,
) raises -> PythonObject:
    """Global alignment in linear space with every sweep running on the device."""
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var cells = DEFAULT_TILE_CELLS
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)

    var path_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
    var path_entries = List[Layer](length=len(left) + 1, fill=Layer.ALIGNING)
    var ctx = DeviceContext(device_id=placement.gpu_id)
    hirschberg_path_gpu(
        ctx,
        left,
        right,
        0,
        len(left),
        0,
        len(right),
        substitutions,
        alphabet_size,
        scoring,
        cells,
        placement.threads,
        path_columns,
        path_entries,
    )
    var score = score_path(left, right, path_columns, path_entries, substitutions, alphabet_size, scoring, len(left))
    var expanded = expand_path(left, right, path_columns, path_entries, alphabet, AlignmentMode.GLOBAL, 0, len(left))
    var triple = alignment_triple(expanded[0], expanded[1], score)
    return triple


def smith_waterman_gotoh_alignment_linear(
    first: PythonObject,
    second: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
) raises -> PythonObject:
    """Local alignment in linear space, by reduction to the global problem.

    A forward local sweep finds where the best alignment ends, a backward sweep over those
    prefixes finds where it starts, and the global recursion then runs on that rectangle alone.
    The untrimmed flanks the Python leaves in front of a local result are filled in afterwards.

    Hirschberg can in fact be aimed at a local matrix directly, at twice the cell count and the
    same linear space — the local optimum is a maximum over sub-rectangles, so the join gains
    cases for an optimum lying wholly above or wholly below the cut. The reduction is used anyway
    because it keeps one well-understood global recursion instead of four subproblem modes, which
    is what Myers-Miller's own authors prescribe.
    """
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var cells = DEFAULT_TILE_CELLS
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)

    var last_row, last_column, score = local_extremum[SweepHalf.FORWARD](
        left, right, len(left), len(right), substitutions, alphabet_size, scoring
    )

    var path_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
    var path_entries = List[Layer](length=len(left) + 1, fill=Layer.ALIGNING)

    var first_row = last_row
    if score > 0:
        var back_rows, back_columns, _ = local_extremum[SweepHalf.REVERSE](
            left, right, last_row, last_column, substitutions, alphabet_size, scoring
        )
        first_row = last_row - back_rows
        var first_column = last_column - back_columns

        path_columns[last_row] = Int32(last_column)
        var core_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
        var core_entries = List[Layer](length=len(left) + 1, fill=Layer.ALIGNING)
        for index in range(len(path_columns)):
            core_columns[index] = path_columns[index]
            core_entries[index] = path_entries[index]
        hirschberg_window(
            left,
            right,
            first_row,
            last_row,
            first_column,
            last_column,
            substitutions,
            alphabet_size,
            scoring,
            cells,
            core_columns,
            core_entries,
        )
        for index in range(first_row, last_row + 1):
            path_columns[index] = core_columns[index]
            path_entries[index] = core_entries[index]

    var expanded = expand_path(
        left, right, path_columns, path_entries, alphabet, AlignmentMode.LOCAL, first_row, last_row
    )
    var triple = alignment_triple(expanded[0], expanded[1], score)
    return triple

def smith_waterman_gotoh_alignment_linear_gpu(
    first: PythonObject,
    second: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
    placement: Placement,
) raises -> PythonObject:
    """Local alignment in linear space with every sweep running on the device."""
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var cells = DEFAULT_TILE_CELLS
    var left = translate(String(first), alphabet)
    var right = translate(String(second), alphabet)

    var sequences = List[Scalar[SYMBOL_DTYPE]](capacity=len(left) + len(right))
    sequences.extend(Span(left))
    sequences.extend(Span(right))

    var ctx = DeviceContext(device_id=placement.gpu_id)
    var buffers = sweep_buffers(ctx, len(left), len(right), Span(sequences), substitutions)
    var last_row, last_column, score = device_local_extremum[SweepHalf.FORWARD](
        ctx, buffers, len(left), len(left), len(right), alphabet_size, scoring
    )

    var path_columns = List[Int32](length=len(left) + 1, fill=Int32(0))
    var path_entries = List[Layer](length=len(left) + 1, fill=Layer.ALIGNING)

    var first_row = last_row
    if score > 0:
        var back_rows, back_columns, _ = device_local_extremum[SweepHalf.REVERSE](
            ctx, buffers, len(left), last_row, last_column, alphabet_size, scoring
        )
        first_row = last_row - back_rows
        var first_column = last_column - back_columns

        path_columns[last_row] = Int32(last_column)
        hirschberg_path_gpu(
            ctx,
            left,
            right,
            first_row,
            last_row,
            first_column,
            last_column,
            substitutions,
            alphabet_size,
            scoring,
            cells,
            placement.threads,
            path_columns,
            path_entries,
        )

    var expanded = expand_path(
        left, right, path_columns, path_entries, alphabet, AlignmentMode.LOCAL, first_row, last_row
    )
    var triple = alignment_triple(expanded[0], expanded[1], score)
    return triple


def mode_from(value: PythonObject) raises -> AlignmentMode:
    """Reads the alignment mode a caller named."""
    var name = String(value)
    if name == "global":
        return AlignmentMode.GLOBAL
    if name == "local":
        return AlignmentMode.LOCAL
    raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, String("alignment mode ", name))


def placement_from(requested: PythonObject) raises -> Placement:
    """Reads the caller's placement record, which names the executor, the accelerator and the width."""
    var gpu_id = optional_int(requested.gpu_id).or_else(0)
    var threads = optional_int(requested.threads).or_else(hardware_threads())
    if executor_from(requested.device) == Executor.DEVICE:
        return Placement.device(gpu_id, threads)
    return Placement.host(threads)


def executor_from(value: PythonObject) raises -> Executor:
    """Reads the device a caller named."""
    var name = String(value)
    if name == "cpu":
        return Executor.HOST
    if name == "gpu":
        return Executor.DEVICE
    raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, String("device ", name))


def paired_length(firsts: PythonObject, seconds: PythonObject) raises -> Int:
    """The number of pairs, refusing two sides that do not line up."""
    var pairs = python_length(firsts)
    if pairs != python_length(seconds):
        raise AffineGapsError(ErrorKind.LENGTH_MISMATCH, "batch sides")
    return pairs


def gotoh_scores(
    firsts: PythonObject,
    seconds: PythonObject,
    mode: PythonObject,
    requested: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
) raises -> PythonObject:
    """Scores every pair. The score kernels are two-row, so linear space is the only space."""
    var requested_mode = mode_from(mode)
    if executor_from(requested.device) == Executor.DEVICE:
        comptime for index in range(len(ALL_MODES)):
            comptime candidate = ALL_MODES[index]
            if requested_mode == candidate:
                return gotoh_scores_batch[candidate](firsts, seconds, substitution, gaps, placement_from(requested))

    var results = Python().list()
    for index in range(paired_length(firsts, seconds)):
        comptime for choice in range(len(ALL_MODES)):
            comptime candidate = ALL_MODES[choice]
            if requested_mode == candidate:
                results.append(gotoh_score[candidate](firsts[index], seconds[index], substitution, gaps))
    return results


def align_pair(
    first: PythonObject,
    second: PythonObject,
    mode: AlignmentMode,
    executor: Executor,
    stored_budget: Int,
    substitution: PythonObject,
    gaps: PythonObject,
    placement: Placement,
) raises -> PythonObject:
    """One pair, on the path its matrix can afford."""
    var cells = python_length(first) * python_length(second)
    var limit = min(stored_budget, DEVICE_STORED_CELLS) if executor == Executor.DEVICE else stored_budget
    var too_tall = executor == Executor.DEVICE and python_length(first) > MAX_BAND_LENGTH
    if cells > limit or too_tall:
        if executor == Executor.DEVICE:
            if mode == AlignmentMode.LOCAL:
                return smith_waterman_gotoh_alignment_linear_gpu(first, second, substitution, gaps, placement)
            return needleman_wunsch_gotoh_alignment_linear_gpu(first, second, substitution, gaps, placement)
        if mode == AlignmentMode.LOCAL:
            return smith_waterman_gotoh_alignment_linear(first, second, substitution, gaps)
        return needleman_wunsch_gotoh_alignment_linear(first, second, substitution, gaps)

    if executor == Executor.DEVICE:
        var lefts = Python().list()
        """No single-pair device entry exists for the stored traceback, so this is a batch of one."""
        var rights = Python().list()
        lefts.append(first)
        rights.append(second)
        comptime for index in range(len(ALL_MODES)):
            comptime candidate = ALL_MODES[index]
            if mode == candidate:
                return gotoh_alignments_batch[candidate](lefts, rights, substitution, gaps, placement)[0]

    comptime for index in range(len(ALL_MODES)):
        comptime candidate = ALL_MODES[index]
        if mode == candidate:
            return gotoh_alignment[candidate](first, second, substitution, gaps)
    raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, "alignment mode")


def gotoh_alignments(
    firsts: PythonObject,
    seconds: PythonObject,
    mode: PythonObject,
    requested: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
    stored_budget: PythonObject,
) raises -> PythonObject:
    """Aligns every pair, taking the linear-space traceback once a matrix outgrows the budget.

    A device batch whose every matrix fits goes out as one launch; anything else walks pair by
    pair, because the two tracebacks cannot share a launch.
    """
    var requested_mode = mode_from(mode)
    var executor = executor_from(requested.device)
    var placement = placement_from(requested)
    var budget = Int(String(stored_budget))
    var pairs = paired_length(firsts, seconds)

    var limit = budget if pairs > 1 else min(budget, DEVICE_STORED_CELLS)
    """A batch of one is a single pair however it arrived, and gets the single pair's crossover."""

    var batchable = List[Int]()
    """
    Both bounds are real and independent: the stored kernel indexes its carry by the first sequence, so a tall pair
    fails it even when the whole matrix would fit. A pair that fails either one takes the linear path by itself rather
    than dragging the batch down with it.
    """
    if executor == Executor.DEVICE:
        for index in range(pairs):
            var rows = python_length(firsts[index])
            if rows * python_length(seconds[index]) <= limit and rows <= MAX_BAND_LENGTH:
                batchable.append(index)

    var results = Python().list()
    for _ in range(pairs):
        results.append(Python().none())
    var placed = List[Bool](length=pairs, fill=False)

    if len(batchable) > 0:
        var batch_firsts = Python().list()
        var batch_seconds = Python().list()
        for slot in range(len(batchable)):
            batch_firsts.append(firsts[batchable[slot]])
            batch_seconds.append(seconds[batchable[slot]])
        comptime for index in range(len(ALL_MODES)):
            comptime candidate = ALL_MODES[index]
            if requested_mode == candidate:
                var aligned = gotoh_alignments_batch[candidate](batch_firsts, batch_seconds, substitution, gaps, placement)
                for slot in range(len(batchable)):
                    results[batchable[slot]] = aligned[slot]
                    placed[batchable[slot]] = True

    for index in range(pairs):
        if not placed[index]:
            results[index] = align_pair(
                firsts[index], seconds[index], requested_mode, executor, budget, substitution, gaps, placement
            )
    return results


@export
def PyInit_affinegaps_mojo() abi("C") -> PythonObject:
    try:
        var builder = PythonModuleBuilder("affinegaps_mojo")
        builder.def_function[gotoh_scores]("gotoh_scores")
        builder.def_function[gotoh_alignments]("gotoh_alignments")
        builder.def_function[levenshtein_alignment]("levenshtein_alignment")
        builder.def_function[colorize_alignment]("colorize_alignment")
        builder.def_function[sankoff_cofold]("sankoff_cofold")
        builder.def_function[zuker_fold]("zuker_fold")
        return builder.finalize()
    except error:
        abort(String("Failed to initialize affinegaps_mojo: ", error))


# endregion Python Bindings
