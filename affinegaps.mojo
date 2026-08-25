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


from common import (
    DEFAULT_PROTEINS_ALPHABET,
    DEFAULT_RNA_ALPHABET,
    Device,
    DeviceScope,
    FALLBACK_LETTER,
    OffsetDType,
    Placement,
    ScoreDType,
    SubstitutionDType,
    SymbolDType,
    hardware_threads,
    translate,
    uniform_matrix,
)
from errors import AffineGapsError, ErrorKind
from folding import FoldResult, device_fold, serial_fold
from cofolding import (
    CofoldResult,
    DEFAULT_SANKOFF_GAP,
    DEFAULT_SANKOFF_MATCH,
    DEFAULT_SANKOFF_MISMATCH,
    SankoffScoring,
    device_cofold,
    serial_cofold,
)
from alignment import (
    ALL_MODES,
    AffineGapCosts,
    AlignmentMode,
    DEFAULT_GAP_EXTENSION,
    DEFAULT_GAP_OPENING,
    DEFAULT_LEAF_CELLS,
    DEVICE_STORED_CELLS,
    GapRun,
    Layer,
    Space,
    Sweep,
    SweepBuffers,
    SweepHalf,
    colorize,
    default_proteins_matrix,
    device_alignments,
    device_align,
    device_score,
    expand_path,
    serial_hirschberg,
    serial_local_extremum,
    score_path,
    serial_align,
    band_length,
    serial_score,
    serving_space,
    device_sweep_level,
    device_scores,
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
    return AffineGapCosts.checked(Int32(opening), Int32(extension))


def matrix_from(substitution: PythonObject, alphabet_size: Int) raises -> List[Scalar[SubstitutionDType]]:
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
    # Both halves or neither: a record carrying only one would otherwise read an empty Optional.
    if not given_match or not given_mismatch:
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
    var gapped_pair = Python().list()
    gapped_pair.append(PythonObject(first_gapped))
    gapped_pair.append(PythonObject(second_gapped))
    gapped_pair.append(PythonObject(Int(score)))
    return gapped_pair


def gotoh_score[
    mode: AlignmentMode
](first: PythonObject, second: PythonObject, substitution: PythonObject, gaps: PythonObject) raises -> PythonObject:
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var encoded_first = translate(String(first), alphabet)
    var encoded_second = translate(String(second), alphabet)
    return PythonObject(Int(serial_score[mode](encoded_first, encoded_second, substitutions, alphabet_size, scoring)))


def gotoh_score_linear_gpu[
    mode: AlignmentMode
](
    first: PythonObject, second: PythonObject, substitution: PythonObject, gaps: PythonObject, scope: DeviceScope
) raises -> PythonObject:
    """One pair scored with every sweep on the device, for a pair too tall for one block's carry."""
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var encoded_first = translate(String(first), alphabet)
    var encoded_second = translate(String(second), alphabet)
    var score = device_score[mode](scope, encoded_first, encoded_second, substitutions, alphabet_size, scoring)
    return PythonObject(Int(score))


def gotoh_alignment[
    mode: AlignmentMode
](first: PythonObject, second: PythonObject, substitution: PythonObject, gaps: PythonObject) raises -> PythonObject:
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var encoded_first = translate(String(first), alphabet)
    var encoded_second = translate(String(second), alphabet)
    var aligned = serial_align[mode](encoded_first, encoded_second, substitutions, alphabet_size, scoring, alphabet)
    var gapped_pair = alignment_triple(aligned.first_gapped, aligned.second_gapped, aligned.score)
    return gapped_pair


def python_length(value: PythonObject) raises -> Int:
    return Int(String(value.__len__()))


@fieldwise_init
struct BatchTape(Movable):
    """Both sides of a batch on one tape, which is the shape a kernel can take a pointer to."""

    var sequences: List[Scalar[SymbolDType]]
    """Every sequence concatenated, first and second of each pair alternating."""
    var offsets: List[Scalar[OffsetDType]]
    """Where each sequence begins, so a block can find its own pair."""


def pack_batch(firsts: PythonObject, seconds: PythonObject, alphabet: String) raises -> BatchTape:
    """Both sides of a batch concatenated onto one tape, with offsets marking where each begins.

    A kernel takes a pointer and a length, so the batch crosses as one allocation rather than as a
    list of them; the offsets are what let a block find its own pair.
    """
    var pairs = paired_length(firsts, seconds)
    var sequences = List[Scalar[SymbolDType]]()
    var offsets = List[Scalar[OffsetDType]]()
    offsets.append(0)
    for index in range(pairs):
        var encoded_first = translate(String(firsts[index]), alphabet)
        sequences.extend(encoded_first^)
        offsets.append(Scalar[OffsetDType](len(sequences)))
        var encoded_second = translate(String(seconds[index]), alphabet)
        sequences.extend(encoded_second^)
        offsets.append(Scalar[OffsetDType](len(sequences)))
    return BatchTape(sequences^, offsets^)


def gotoh_scores_batch[
    mode: AlignmentMode
](
    firsts: PythonObject, seconds: PythonObject, substitution: PythonObject, gaps: PythonObject, scope: DeviceScope
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

    var scores = device_scores[mode](scope, tape.sequences, tape.offsets, substitutions, alphabet_size, scoring)
    var scored = Python().list()
    for index in range(len(scores)):
        scored.append(PythonObject(Int(scores[index])))
    return scored


def gotoh_alignments_batch[
    mode: AlignmentMode
](
    firsts: PythonObject, seconds: PythonObject, substitution: PythonObject, gaps: PythonObject, scope: DeviceScope
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

    var aligned = device_alignments[mode](scope, tape.sequences, tape.offsets, substitutions, alphabet, scoring)
    var alignments = Python().list()
    for index in range(len(aligned)):
        var gapped_pair = alignment_triple(
            aligned[index].first_gapped, aligned[index].second_gapped, aligned[index].score
        )
        alignments.append(gapped_pair)
    return alignments


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
    var first_text = String(first)
    var second_text = String(second)
    for byte in first_text.as_bytes():
        if byte >= 0x80:
            raise AffineGapsError(ErrorKind.NOT_ASCII, "unit-cost alignment")
    for byte in second_text.as_bytes():
        if byte >= 0x80:
            raise AffineGapsError(ErrorKind.NOT_ASCII, "unit-cost alignment")
    var alphabet = combined_alphabet(first_text, second_text)
    var alphabet_size = alphabet.byte_length()
    var substitutions = uniform_matrix(alphabet_size, 0, -1)
    var scoring = AffineGapCosts.checked(Int32(-1), Int32(-1))
    var encoded_first = translate(first_text, alphabet)
    var encoded_second = translate(second_text, alphabet)
    var aligned = serial_align[AlignmentMode.GLOBAL](
        encoded_first, encoded_second, substitutions, alphabet_size, scoring, alphabet
    )
    var gapped_pair = Python().list()
    gapped_pair.append(PythonObject(aligned.first_gapped))
    gapped_pair.append(PythonObject(aligned.second_gapped))
    gapped_pair.append(PythonObject(-Int(aligned.score)))
    return gapped_pair


def gpu_specs(gpu_id: PythonObject) raises -> PythonObject:
    """What the named accelerator reports about itself, in the order `GpuSpecs` declares."""
    var scope = DeviceScope(optional_int(gpu_id).or_else(0))
    var reported = Python().list()
    reported.append(PythonObject(scope.specs.shared_memory_per_multiprocessor))
    reported.append(PythonObject(scope.specs.reserved_memory_per_block))
    reported.append(PythonObject(scope.specs.largest_allocation))
    reported.append(PythonObject(scope.specs.streaming_multiprocessors))
    reported.append(PythonObject(scope.specs.max_blocks_per_multiprocessor))
    return reported


def proteins_matrix() raises -> PythonObject:
    """The compiled default table, flat and row-major, so a test can hold it against the reference."""
    var table = default_proteins_matrix()
    var reported = Python().list()
    for index in range(len(table)):
        reported.append(PythonObject(Int(table[index])))
    return reported


def zuker_fold(sequence: PythonObject, requested: PythonObject) raises -> PythonObject:
    """Minimum free energy folding of one RNA sequence over the Turner nearest-neighbour model.

    Returns the dot-bracket structure and the free energy in kilocalories per mole. The recurrence
    works in integer decikilocalories, so the same sequence always folds to the same answer.
    """
    var text = String(sequence)
    var placement = placement_from(requested)
    var aligned: FoldResult
    if String(requested.device) == "gpu":
        var scope = DeviceScope(placement.gpu_id)
        aligned = device_fold(scope, text)
    else:
        aligned = serial_fold(text)

    var couple = Python().list()
    couple.append(PythonObject(aligned.structure))
    couple.append(PythonObject(Float64(Int(aligned.decikcal)) / 10.0))
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
    var scoring = SankoffScoring(Int32(optional_int(gap).or_else(Int(DEFAULT_SANKOFF_GAP))))
    var match_score = optional_int(match_reward).or_else(Int(DEFAULT_SANKOFF_MATCH))
    var mismatch_score = optional_int(mismatch_penalty).or_else(Int(DEFAULT_SANKOFF_MISMATCH))
    var alphabet = String(DEFAULT_RNA_ALPHABET)
    var encoded_first = String(first)
    var encoded_second = String(second)

    var placement = placement_from(requested)
    var aligned: CofoldResult
    if String(requested.device) == "gpu":
        var scope = DeviceScope(placement.gpu_id)
        aligned = device_cofold(scope, encoded_first, encoded_second, alphabet, scoring, match_score, mismatch_score)
    else:
        aligned = serial_cofold(encoded_first, encoded_second, alphabet, scoring, match_score, mismatch_score)

    var quadruple = Python().list()
    quadruple.append(PythonObject(aligned.gapped_first))
    quadruple.append(PythonObject(aligned.gapped_second))
    quadruple.append(PythonObject(aligned.structure))
    quadruple.append(PythonObject(Int(aligned.score)))
    return quadruple


def colorize_alignment(first_gapped: PythonObject, second_gapped: PythonObject) raises -> PythonObject:
    """Wraps each column in an ANSI colour: green for a match, red for a mismatch, dim for a gap."""
    var painted = colorize(String(first_gapped), String(second_gapped))
    var pair = Python().list()
    pair.append(PythonObject(painted[0]))
    pair.append(PythonObject(painted[1]))
    return pair


def needleman_wunsch_gotoh_alignment_linear(
    first: PythonObject, second: PythonObject, substitution: PythonObject, gaps: PythonObject
) raises -> PythonObject:
    """Global alignment in linear space, splitting rows and joining halves Myers-Miller style.

    `leaf_cells` is the only knob: subproblems at or below it are solved outright, above it they
    are split. Raising it past the whole matrix collapses to a single direct traceback, which is
    what makes the two paths comparable.
    """
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var cells = DEFAULT_LEAF_CELLS
    var encoded_first = translate(String(first), alphabet)
    var encoded_second = translate(String(second), alphabet)

    var path_columns = List[Int32](length=len(encoded_first) + 1, fill=Int32(0))
    var path_layers = List[Layer](length=len(encoded_first) + 1, fill=Layer.ALIGNING)
    serial_hirschberg(
        encoded_first,
        encoded_second,
        0,
        len(encoded_first),
        0,
        len(encoded_second),
        substitutions,
        alphabet_size,
        scoring,
        cells,
        path_columns,
        path_layers,
    )
    var score = score_path(
        encoded_first,
        encoded_second,
        path_columns,
        path_layers,
        substitutions,
        alphabet_size,
        scoring,
        len(encoded_first),
    )
    var expanded = expand_path(
        encoded_first, encoded_second, path_columns, path_layers, alphabet, AlignmentMode.GLOBAL, 0, len(encoded_first)
    )
    var gapped_pair = alignment_triple(expanded[0], expanded[1], score)
    return gapped_pair


def needleman_wunsch_gotoh_alignment_linear_gpu(
    first: PythonObject,
    second: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
    scope: DeviceScope,
    placement: Placement,
) raises -> PythonObject:
    """Global alignment in linear space with every sweep running on the device."""
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var encoded_first = translate(String(first), alphabet)
    var encoded_second = translate(String(second), alphabet)
    var aligned = device_align[AlignmentMode.GLOBAL](
        scope,
        encoded_first,
        encoded_second,
        substitutions,
        alphabet_size,
        scoring,
        alphabet,
        DEFAULT_LEAF_CELLS,
        placement,
    )
    return alignment_triple(aligned.first_gapped, aligned.second_gapped, aligned.score)


def smith_waterman_gotoh_alignment_linear(
    first: PythonObject, second: PythonObject, substitution: PythonObject, gaps: PythonObject
) raises -> PythonObject:
    """Local alignment in linear space, by reduction to the global problem.

    A forward local sweep finds where the best alignment ends, a backward sweep over those
    prefixes finds where it starts, and the global recursion then runs on that rectangle alone.
    The untrimmed flanks the Python leaves in front of a local aligned are filled in afterwards.

    Hirschberg can in fact be aimed at a local matrix directly, at twice the cell count and the
    same linear space — the local optimum is a maximum over sub-rectangles, so the join gains
    cases for an optimum lying wholly above or wholly below the cut. The reduction is used anyway
    because it keeps one well-understood global recursion instead of four subproblem modes, which
    is what Myers-Miller's own authors prescribe.
    """
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var cells = DEFAULT_LEAF_CELLS
    var encoded_first = translate(String(first), alphabet)
    var encoded_second = translate(String(second), alphabet)

    var last_row, last_column, score = serial_local_extremum[SweepHalf.FORWARD](
        encoded_first, encoded_second, len(encoded_first), len(encoded_second), substitutions, alphabet_size, scoring
    )

    var path_columns = List[Int32](length=len(encoded_first) + 1, fill=Int32(0))
    var path_layers = List[Layer](length=len(encoded_first) + 1, fill=Layer.ALIGNING)

    var first_row = last_row
    if score > 0:
        var back_rows, back_columns, _ = serial_local_extremum[SweepHalf.REVERSE](
            encoded_first, encoded_second, last_row, last_column, substitutions, alphabet_size, scoring
        )
        first_row = last_row - back_rows
        var first_column = last_column - back_columns

        path_columns[last_row] = Int32(last_column)
        var core_columns = List[Int32](length=len(encoded_first) + 1, fill=Int32(0))
        var core_layers = List[Layer](length=len(encoded_first) + 1, fill=Layer.ALIGNING)
        for index in range(len(path_columns)):
            core_columns[index] = path_columns[index]
            core_layers[index] = path_layers[index]
        serial_hirschberg(
            encoded_first,
            encoded_second,
            first_row,
            last_row,
            first_column,
            last_column,
            substitutions,
            alphabet_size,
            scoring,
            cells,
            core_columns,
            core_layers,
        )
        for index in range(first_row, last_row + 1):
            path_columns[index] = core_columns[index]
            path_layers[index] = core_layers[index]

    var expanded = expand_path(
        encoded_first, encoded_second, path_columns, path_layers, alphabet, AlignmentMode.LOCAL, first_row, last_row
    )
    var gapped_pair = alignment_triple(expanded[0], expanded[1], score)
    return gapped_pair


def smith_waterman_gotoh_alignment_linear_gpu(
    first: PythonObject,
    second: PythonObject,
    substitution: PythonObject,
    gaps: PythonObject,
    scope: DeviceScope,
    placement: Placement,
) raises -> PythonObject:
    """Local alignment in linear space with every sweep running on the device."""
    var alphabet, alphabet_size, scoring = protein_defaults(gaps)
    var substitutions = matrix_from(substitution, alphabet_size)
    var encoded_first = translate(String(first), alphabet)
    var encoded_second = translate(String(second), alphabet)
    var aligned = device_align[AlignmentMode.LOCAL](
        scope,
        encoded_first,
        encoded_second,
        substitutions,
        alphabet_size,
        scoring,
        alphabet,
        DEFAULT_LEAF_CELLS,
        placement,
    )
    return alignment_triple(aligned.first_gapped, aligned.second_gapped, aligned.score)


def mode_from(value: PythonObject) raises -> AlignmentMode:
    """Reads the alignment mode a caller named."""
    var name = String(value)
    if name == "global":
        return AlignmentMode.GLOBAL
    if name == "local":
        return AlignmentMode.LOCAL
    raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, String("alignment mode ", name))


def placement_from(requested: PythonObject) raises -> Placement:
    """Reads the caller's placement record, which names the device, the accelerator and the width."""
    var gpu_id = optional_int(requested.gpu_id).or_else(0)
    var threads = optional_int(requested.threads).or_else(hardware_threads())
    if device_from(requested.device) == Device.GPU:
        return Placement.on_gpu(gpu_id, threads)
    return Placement.on_cpu(threads)


def device_from(value: PythonObject) raises -> Device:
    """Reads the device a caller named, refusing anything this build does not serve."""
    var name = String(value)
    if name == "cpu":
        return Device.CPU
    if name == "gpu":
        return Device.GPU
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
    """Scores every pair, on the sweep each one's height can afford.

    The score is two-row everywhere, so only the sweep changes: a pair the strip kernel's carry
    can index goes out with the batch, and a taller one takes the tiled sweep by itself.
    """
    var requested_mode = mode_from(mode)
    var device = device_from(requested.device)
    var pairs = paired_length(firsts, seconds)

    var results = Python().list()
    if pairs == 0:
        return results
    for _ in range(pairs):
        results.append(Python().none())

    if device == Device.CPU:
        for index in range(pairs):
            comptime for choice in range(len(ALL_MODES)):
                comptime candidate = ALL_MODES[choice]
                if requested_mode == candidate:
                    results[index] = gotoh_score[candidate](firsts[index], seconds[index], substitution, gaps)
        return results

    var placement = placement_from(requested)
    var scope = DeviceScope(placement.gpu_id)
    """One context for the whole call, so the band query is not a second accelerator handshake."""
    var band = band_length(scope.specs)
    var banded = List[Int]()
    """The strip kernel indexes its carry by the first sequence, so height alone decides."""
    var tiled = List[Int]()
    for index in range(pairs):
        if serving_space(python_length(firsts[index]), band) == Space.BANDED:
            banded.append(index)
        else:
            tiled.append(index)

    if len(banded) > 0:
        var batch_firsts = Python().list()
        var batch_seconds = Python().list()
        for slot in range(len(banded)):
            batch_firsts.append(firsts[banded[slot]])
            batch_seconds.append(seconds[banded[slot]])
        comptime for index in range(len(ALL_MODES)):
            comptime candidate = ALL_MODES[index]
            if requested_mode == candidate:
                var scored = gotoh_scores_batch[candidate](batch_firsts, batch_seconds, substitution, gaps, scope)
                for slot in range(len(banded)):
                    results[banded[slot]] = scored[slot]

    for slot in range(len(tiled)):
        var index = tiled[slot]
        comptime for choice in range(len(ALL_MODES)):
            comptime candidate = ALL_MODES[choice]
            if requested_mode == candidate:
                results[index] = gotoh_score_linear_gpu[candidate](
                    firsts[index], seconds[index], substitution, gaps, scope
                )
    return results


def align_pair_host(
    first: PythonObject,
    second: PythonObject,
    mode: AlignmentMode,
    stored_budget: Int,
    substitution: PythonObject,
    gaps: PythonObject,
) raises -> PythonObject:
    """One pair on the host, stored while its matrix fits the budget and linear once it does not."""
    if python_length(first) * python_length(second) > stored_budget:
        if mode == AlignmentMode.LOCAL:
            return smith_waterman_gotoh_alignment_linear(first, second, substitution, gaps)
        return needleman_wunsch_gotoh_alignment_linear(first, second, substitution, gaps)

    comptime for index in range(len(ALL_MODES)):
        comptime candidate = ALL_MODES[index]
        if mode == candidate:
            return gotoh_alignment[candidate](first, second, substitution, gaps)
    raise AffineGapsError(ErrorKind.INVALID_ARGUMENT, "alignment mode")


def align_pair_device(
    first: PythonObject,
    second: PythonObject,
    mode: AlignmentMode,
    stored_budget: Int,
    band: Int,
    substitution: PythonObject,
    gaps: PythonObject,
    scope: DeviceScope,
    placement: Placement,
) raises -> PythonObject:
    """One pair on the device, on the sweep its height and its matrix can afford.

    Both bounds are real and independent: the stored kernel indexes its carry by the first
    sequence, so a tall pair fails it even when the whole matrix would fit.
    """
    var cells = python_length(first) * python_length(second)
    var stored = cells <= min(stored_budget, DEVICE_STORED_CELLS)
    if not stored or serving_space(python_length(first), band) == Space.TILED:
        if mode == AlignmentMode.LOCAL:
            return smith_waterman_gotoh_alignment_linear_gpu(first, second, substitution, gaps, scope, placement)
        return needleman_wunsch_gotoh_alignment_linear_gpu(first, second, substitution, gaps, scope, placement)

    var lefts = Python().list()
    """No single-pair device entry exists for the stored traceback, so this is a batch of one."""
    var rights = Python().list()
    lefts.append(first)
    rights.append(second)
    comptime for index in range(len(ALL_MODES)):
        comptime candidate = ALL_MODES[index]
        if mode == candidate:
            return gotoh_alignments_batch[candidate](lefts, rights, substitution, gaps, scope)[0]
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
    var device = device_from(requested.device)
    var placement = placement_from(requested)
    var budget = Int(String(stored_budget))
    var pairs = paired_length(firsts, seconds)

    var limit = budget if pairs > 1 else min(budget, DEVICE_STORED_CELLS)
    """A batch of one is a single pair however it arrived, and gets the single pair's crossover."""

    var results = Python().list()
    for _ in range(pairs):
        results.append(Python().none())
    var placed = List[Bool](length=pairs, fill=False)
    var band = 0

    # One context serves the whole device call, and a host call names no accelerator at all.
    if device == Device.GPU:
        var scope = DeviceScope(placement.gpu_id)
        band = band_length(scope.specs)

        var batchable = List[Int]()
        """
        Both bounds are real and independent: the stored kernel indexes its carry by the first sequence, so a tall
        pair fails it even when the whole matrix would fit. A pair that fails either one takes the linear path by
        itself rather than dragging the batch down with it.
        """
        for index in range(pairs):
            var rows = python_length(firsts[index])
            if rows * python_length(seconds[index]) <= limit and serving_space(rows, band) == Space.BANDED:
                batchable.append(index)

        if len(batchable) > 0:
            var batch_firsts = Python().list()
            var batch_seconds = Python().list()
            for slot in range(len(batchable)):
                batch_firsts.append(firsts[batchable[slot]])
                batch_seconds.append(seconds[batchable[slot]])
            comptime for index in range(len(ALL_MODES)):
                comptime candidate = ALL_MODES[index]
                if requested_mode == candidate:
                    var aligned = gotoh_alignments_batch[candidate](
                        batch_firsts, batch_seconds, substitution, gaps, scope
                    )
                    for slot in range(len(batchable)):
                        results[batchable[slot]] = aligned[slot]
                        placed[batchable[slot]] = True

        for index in range(pairs):
            if not placed[index]:
                results[index] = align_pair_device(
                    firsts[index], seconds[index], requested_mode, budget, band, substitution, gaps, scope, placement
                )
        return results

    for index in range(pairs):
        results[index] = align_pair_host(firsts[index], seconds[index], requested_mode, budget, substitution, gaps)
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
        builder.def_function[gpu_specs]("gpu_specs")
        builder.def_function[proteins_matrix]("proteins_matrix")
        return builder.finalize()
    except error:
        abort(String("Failed to initialize affinegaps_mojo: ", error))


# endregion Python Bindings
