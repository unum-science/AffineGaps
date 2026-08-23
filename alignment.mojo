"""
Gotoh affine-gap alignment for CPU and GPU, with the reconstruction itself on the device.

The alignment is recovered in linear space, not just the score. The scoring recurrences, the
tie-breaking and the border initialization are transcribed from `affinegaps.py`, which stays
the parity oracle.

A row is the wrong sweep axis for a GPU, because the insertion term of a cell reads the
insertion term of its left neighbour. This module sweeps anti-diagonals instead, where every
cell of `d = i + j` reads only `d - 1` and `d - 2` and the whole above_left is independent.

Traceback is Hirschberg with a Myers-Miller affine join, splitting on rows rather than
anti-diagonals: a substitution anti_diagonal advances `i + j` by two and can skip a above_left entirely,
while the row index advances by zero or one per anti_diagonal. The recursion bottoms out in a direct
traceback over a stored decision tile.

The traceback walks all three layers — match, deletion and insertion — so every path realizes
the score reported alongside it, which is not automatic for affine gaps.

For score-only work at much higher throughput, see `ashvardanian/StringZilla`, whose
`stringzillas/similarities` kernels compute the same distances and scores. It carries no
traceback, which is what this module exists to provide.
"""

from std.gpu import block_dim, block_idx, grid_dim, lane_id, thread_idx
from std.math import ceildiv
from std.gpu.primitives.warp import WARP_SIZE, shuffle_down, shuffle_up, shuffle_xor
from std.memory import stack_allocation
from std.memory.pointer import AddressSpace
from std.sys.info import size_of

from max.algorithm import parallelize
from max.gpu import barrier
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.memory import external_memory

from errors import AffineGapsError, ErrorKind
from common import (
    GAP_BYTE,
    MAX_ALPHABET_SIZE,
    NEGATIVE_INFINITY,
    OffsetDType,
    Placement,
    ScoreDType,
    SubstitutionDType,
    SymbolDType,
    THREADS_PER_BLOCK,
    max_dynamic_shared,
    translate,
    upload,
    zeroed,
)

# region Scoring

comptime ChangeDType = DType.uint32

comptime ALL_MODES = (AlignmentMode.GLOBAL, AlignmentMode.LOCAL)

comptime DEFAULT_GAP_OPENING = Int32(-20)
comptime DEFAULT_GAP_EXTENSION = Int32(-1)

comptime BLOCKS_PER_MULTIPROCESSOR = 4
"""
How many blocks should stay resident per multiprocessor. This is the knob; the band capacity below follows from it and
from the device, rather than being guessed. Four is chosen because a batch has to carry more than nine hundred pairs
before occupancy stops being bound by batch size on a hundred-and-thirty-two-multiprocessor device, so a longer reach
is usually free.
"""

comptime CORNER_BYTES = 16
"""One aligned slot for the corner score the walk reads, which is all the strip stages beyond its table."""
comptime STATIC_SHARED_USED = MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE + CORNER_BYTES

comptime MAX_DYNAMIC_SHARED = max_dynamic_shared[BLOCKS_PER_MULTIPROCESSOR]()
comptime WARPS_PER_BLOCK = THREADS_PER_BLOCK // WARP_SIZE
comptime CARRY_BANDS = 2
"""
A strip carries the score and insertion layers of the column to its left, one entering_run per row, which is what bounds how
long a first sequence one block can take.
"""
comptime MAX_BAND_LENGTH = (MAX_DYNAMIC_SHARED - STATIC_SHARED_USED) // (CARRY_BANDS * 4) - 1


@fieldwise_init
struct Layer(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which of Gotoh's three layers a score came from, and which one a traceback is inside.

    The walk never needs to know whether an aligning anti_diagonal matched or substituted, so the two
    collapse into one layer here; only the emitted letters differ, and those come from the
    sequences.
    """

    var identifier: UInt8
    """Which case this names."""
    comptime ALIGNING = Self(0)
    """The score came from aligning two symbols."""
    comptime DELETING = Self(1)
    """The score came from a run of gaps in the second sequence."""
    comptime INSERTING = Self(2)
    """The score came from a run of gaps in the first sequence."""


@fieldwise_init
struct GapRun(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Whether a gap run arriving at a cell was already open one anti_diagonal earlier.

    Ties resolve towards opening, because both extension tests use a strict `>`.
    """

    var identifier: UInt8
    """Which case this names."""
    comptime OPENS = Self(0)
    """The run starts here and pays the opening cost."""
    comptime EXTENDS = Self(1)
    """The run was already open, so it pays only the extension."""


@fieldwise_init
struct PathReach(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Whether a local path passes through a cell, or its score falls to zero there."""

    var identifier: UInt8
    """Which case this names."""
    comptime CONTINUES_PAST = Self(0)
    """The local path runs through this cell."""
    comptime ENDS_HERE = Self(1)
    """The local score fell to zero, so the path stops."""


@fieldwise_init
struct Move(ImplicitlyCopyable, TrivialRegisterPassable):
    """One backward move: how far to walk on each axis, and the layer it lands in.

    A zero move on an axis is a gap on that sequence, so the emitted pair follows from the move.
    """

    var row_advance: Int
    """How far the walk moves along the first sequence, zero or one."""
    var column_advance: Int
    """How far it moves along the second, zero or one."""
    var lands_in: Layer
    """Which layer the move arrives in, which decides what the next anti_diagonal may do."""


# BLOSUM62 scaled by five, trimmed to the 23 letters the alphabet emits. The canonical table
# is 24 by 24; its last row and column are the `*` stop codon, which `translate` never produces.
# fmt: off
comptime BLOSUM62_SCALED: Array[Scalar[SubstitutionDType], 529] = [
        20, -5, -10, -10, 0, -5, -5, 0, -10, -5, -5, -5, -5, -10, -5, 5, 0, -15, -10, 0, -10, -5, 0,
        -5, 25, 0, -10, -15, 5, 0, -10, 0, -15, -10, 10, -5, -15, -10, -5, -5, -15, -10, -15, -5, 0, -5,
        -10, 0, 30, 5, -15, 0, 0, 0, 5, -15, -15, 0, -10, -15, -10, 5, 0, -20, -10, -15, 15, 0, -5,
        -10, -10, 5, 30, -15, 0, 10, -5, -5, -15, -20, -5, -15, -15, -5, 0, -5, -20, -15, -15, 20, 5, -5,
        0, -15, -15, -15, 45, -15, -20, -15, -15, -5, -5, -15, -5, -10, -15, -5, -5, -10, -10, -5, -15, -15, -10,
        -5, 5, 0, 0, -15, 25, 10, -10, 0, -15, -10, 5, 0, -15, -5, 0, -5, -10, -5, -10, 0, 15, -5,
        -5, 0, 0, 10, -20, 10, 25, -10, 0, -15, -15, 5, -10, -15, -5, 0, -5, -15, -10, -10, 5, 20, -5,
        0, -10, 0, -5, -15, -10, -10, 30, -10, -20, -20, -10, -15, -15, -10, 0, -10, -10, -15, -15, -5, -10, -5,
        -10, 0, 5, -5, -15, 0, 0, -10, 40, -15, -15, -5, -10, -5, -10, -5, -10, -10, 10, -15, 0, 0, -5,
        -5, -15, -15, -15, -5, -15, -15, -20, -15, 20, 10, -15, 5, 0, -15, -10, -5, -15, -5, 15, -15, -15, -5,
        -5, -10, -15, -20, -5, -10, -15, -20, -15, 10, 20, -10, 10, 0, -15, -10, -5, -10, -5, 5, -20, -15, -5,
        -5, 10, 0, -5, -15, 5, 5, -10, -5, -15, -10, 25, -5, -15, -5, 0, -5, -15, -10, -10, 0, 5, -5,
        -5, -5, -10, -15, -5, 0, -10, -15, -10, 5, 10, -5, 25, 0, -10, -5, -5, -5, -5, 5, -15, -5, -5,
        -10, -15, -15, -15, -10, -15, -15, -15, -5, 0, 0, -15, 0, 30, -20, -10, -10, 5, 15, -5, -15, -15, -5,
        -5, -10, -10, -5, -15, -5, -5, -10, -10, -15, -15, -5, -10, -20, 35, -5, -5, -20, -15, -10, -10, -5, -10,
        5, -5, 5, 0, -5, 0, 0, 0, -5, -10, -10, 0, -5, -10, -5, 20, 5, -15, -10, -10, 0, 0, 0,
        0, -5, 0, -5, -5, -5, -5, -10, -10, -5, -5, -5, -5, -10, -5, 5, 25, -10, -10, 0, -5, -5, 0,
        -15, -15, -20, -20, -10, -10, -15, -10, -10, -15, -10, -15, -5, 5, -20, -15, -10, 55, 10, -15, -20, -15, -10,
        -10, -10, -10, -15, -10, -5, -10, -15, 10, -5, -5, -10, -5, 15, -15, -10, -10, 10, 35, -5, -15, -10, -5,
        0, -15, -15, -15, -5, -10, -10, -15, -15, 15, 5, -10, 5, -5, -10, -10, 0, -15, -5, 20, -15, -10, -5,
        -10, -5, 15, 20, -15, 0, 5, -5, 0, -15, -20, 0, -15, -15, -10, 0, -5, -20, -15, -15, 20, 5, -5,
        -5, 0, 0, 5, -15, 15, 20, -10, 0, -15, -15, 5, -5, -15, -5, 0, -5, -15, -10, -10, 5, 20, -5,
        0, -5, -5, -5, -10, -5, -5, -5, -5, -5, -5, -5, -5, -5, -10, 0, 0, -10, -5, -5, -5, -5, -5,
]
# fmt: on


def default_proteins_matrix() -> List[Scalar[SubstitutionDType]]:
    """BLOSUM62 scaled by five, trimmed to the 23 letters the alphabet actually emits.

    The canonical table is 24 by 24; its last row and column are the `*` stop codon, which
    `translate` never produces.
    """
    var table = materialize[BLOSUM62_SCALED]()
    var matrix = List[Scalar[SubstitutionDType]](capacity=529)
    for index in range(529):
        matrix.append(table[index])
    return matrix^


@fieldwise_init
struct AffineGapCosts(ImplicitlyCopyable, TrivialRegisterPassable):
    """Gotoh's two-parameter gap model. Both penalties are negative."""

    var open: Int32
    """Charged once when a gap run begins."""
    var extend: Int32
    """Charged for every position the run continues."""


@fieldwise_init
struct Cell(ImplicitlyCopyable, TrivialRegisterPassable):
    """The three layers of one dynamic-programming cell."""

    var score: Int32
    """Best of the three layers, which is what a neighbour reads."""
    var deletion: Int32
    """Best score ending in a gap in the second sequence."""
    var insertion: Int32
    """Best score ending in a gap in the first sequence."""


@fieldwise_init
struct AlignmentMode(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which of the two alignment problems the recurrence solves."""

    var identifier: UInt8
    """Which case this names."""
    comptime GLOBAL = Self(0)
    """Needleman-Wunsch: the path spans both sequences end to end."""
    comptime LOCAL = Self(1)
    """Smith-Waterman: the score is clamped at zero and the best cell wins."""


@always_inline
def gotoh_cell[
    mode: AlignmentMode
](
    above_left: Int32,
    above: Int32,
    above_delete: Int32,
    left: Int32,
    left_insert: Int32,
    substitution: Int32,
    scoring: AffineGapCosts,
) -> Cell:
    """One interior cell of the Gotoh recurrence, with the local clamp folded in at comptime.

    This is the single transcription of the recurrence that `affinegaps.py` holds as the oracle;
    every sweep on the host and on the device goes through it.
    """
    var deletion = max(above + scoring.open, above_delete + scoring.extend)
    var insertion = max(left + scoring.open, left_insert + scoring.extend)
    var score = max(max(above_left + substitution, deletion), insertion)
    comptime if mode == AlignmentMode.LOCAL:
        score = max(score, Int32(0))
    return Cell(score, deletion, insertion)


@always_inline
def source_layer(cell: Cell, replacement: Int32) -> Layer:
    """Which layer the score came from; ties resolve to aligning, then deleting."""
    if cell.score == replacement:
        return Layer.ALIGNING
    if cell.score == cell.deletion:
        return Layer.DELETING
    return Layer.INSERTING


@fieldwise_init
struct CellDecision(ImplicitlyCopyable, TrivialRegisterPassable):
    """One stored byte per cell: the layer the score came from, and what both gap runs did.

    Each field answers a question asked from a different walk state, so they are three
    independent facts about one cell rather than one composite state. Five of the eight bits
    are used.
    """

    var bits: UInt8
    """Two bits for the source layer and one for each gap run's state."""

    @staticmethod
    @always_inline
    def recording(source: Layer, deletion: GapRun, insertion: GapRun, reach: PathReach) -> Self:
        return Self(
            source.identifier | (deletion.identifier << 2) | (insertion.identifier << 3) | (reach.identifier << 7)
        )

    @always_inline
    def source(self) -> Layer:
        return Layer(self.bits & 0x03)

    @always_inline
    def deletion(self) -> GapRun:
        return GapRun((self.bits >> 2) & 0x01)

    @always_inline
    def insertion(self) -> GapRun:
        return GapRun((self.bits >> 3) & 0x01)

    @always_inline
    def reach(self) -> PathReach:
        return PathReach(self.bits >> 7)

    @always_inline
    def nibble(self) -> UInt32:
        """The four bits `advance` reads, with a clamped local cell folded into a spare code.

        `Layer` has three values, so the two-bit source field has a fourth code free. Spending it
        on `reach` is sound because `advance` consults `source` only from the aligning layer, and
        a walk in that layer breaks on `reach` before it ever gets there.
        """
        var code = UInt32(self.bits & 0x0F)
        return (code | 0x03) if self.reach() == PathReach.ENDS_HERE else code

    @staticmethod
    @always_inline
    def unpacking(code: UInt32) -> Self:
        """Inverse of `nibble`, restoring the flag from the spare source code."""
        var bits = UInt8(code & 0x0F)
        if (bits & 0x03) == 0x03:
            return Self((bits & 0x0C) | (PathReach.ENDS_HERE.identifier << 7))
        return Self(bits)


@always_inline
def decide[
    mode: AlignmentMode
](cell: Cell, replacement: Int32, above: Cell, left: Cell, scoring: AffineGapCosts) -> CellDecision:
    """The four answers a kernel packs, re-derived from the three score layers."""
    var deletion_run = GapRun.EXTENDS if above.deletion + scoring.extend > above.score + scoring.open else GapRun.OPENS
    var insertion_run = GapRun.EXTENDS if left.insertion + scoring.extend > left.score + scoring.open else GapRun.OPENS
    var reach = PathReach.CONTINUES_PAST
    comptime if mode == AlignmentMode.LOCAL:
        reach = PathReach.ENDS_HERE if cell.score <= 0 else PathReach.CONTINUES_PAST
    return CellDecision.recording(source_layer(cell, replacement), deletion_run, insertion_run, reach)


@always_inline
def advance(state: Layer, decision: CellDecision) -> Move:
    """The traceback's transition function, shared by every walk in this file.

    Entering a gap run from the aligning layer moves in the same anti_diagonal rather than re-reading the
    cell, which the layered walk is free to do because that entering_run never moves on its own.
    """
    var layer = decision.source() if state == Layer.ALIGNING else state
    if layer == Layer.DELETING:
        var next_layer = Layer.DELETING if decision.deletion() == GapRun.EXTENDS else Layer.ALIGNING
        return Move(-1, 0, next_layer)
    if layer == Layer.INSERTING:
        var next_layer = Layer.INSERTING if decision.insertion() == GapRun.EXTENDS else Layer.ALIGNING
        return Move(0, -1, next_layer)
    return Move(-1, -1, Layer.ALIGNING)


# endregion Scoring

# region Serial Reference


@fieldwise_init
struct AlignmentResult(Copyable, Movable):
    """A score and the two gapped strings that realize it."""

    var score: Int32
    """Best of the three layers, which is what a neighbour reads."""
    var first_gapped: String
    """The first sequence with gaps inserted, one character per column."""
    var second_gapped: String
    """The second sequence, gapped to the same columns."""


def serial_score[
    mode: AlignmentMode
](
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
) -> Int32:
    """Two-row reference, transcribed from the `*_score_kernel` functions of `affinegaps.py`."""
    var rows = len(first)
    var columns = len(second)
    var scores_above = List[Int32](length=columns + 1, fill=Int32(0))
    var scores_row = List[Int32](length=columns + 1, fill=Int32(0))
    var deletes_above = List[Int32](length=columns + 1, fill=Int32(0))
    var deletes_row = List[Int32](length=columns + 1, fill=Int32(0))
    var inserts_above = List[Int32](length=columns + 1, fill=Int32(0))
    var inserts_row = List[Int32](length=columns + 1, fill=Int32(0))

    scores_above[0] = 0
    for column in range(1, columns + 1):
        if mode == AlignmentMode.GLOBAL:
            scores_above[column] = scoring.open + Int32(column - 1) * scoring.extend
            deletes_above[column] = scores_above[column] + scoring.open + scoring.extend
        else:
            scores_above[column] = 0
            deletes_above[column] = scoring.open + scoring.extend

    var best = Int32(0)
    for row in range(1, rows + 1):
        if mode == AlignmentMode.GLOBAL:
            scores_row[0] = scoring.open + Int32(row - 1) * scoring.extend
        else:
            scores_row[0] = 0
        inserts_row[0] = scores_row[0] + scoring.open + scoring.extend

        for column in range(1, columns + 1):
            var substitution = Int32(substitutions[Int(first[row - 1]) * alphabet_size + Int(second[column - 1])])
            var cell = gotoh_cell[mode](
                scores_above[column - 1],
                scores_above[column],
                deletes_above[column],
                scores_row[column - 1],
                inserts_row[column - 1],
                substitution,
                scoring,
            )
            comptime if mode == AlignmentMode.LOCAL:
                best = max(best, cell.score)
            scores_row[column] = cell.score
            deletes_row[column] = cell.deletion
            inserts_row[column] = cell.insertion

        swap(scores_above, scores_row)
        swap(deletes_above, deletes_row)
        swap(inserts_above, inserts_row)

    if mode == AlignmentMode.LOCAL:
        return best
    return scores_above[columns]


def serial_align[
    mode: AlignmentMode
](
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    alphabet: String,
) -> AlignmentResult:
    """Full-matrix reference, transcribed from the `_*_kernel` plus `_reconstruct_alignment`."""
    var rows = len(first)
    var columns = len(second)
    var stride = columns + 1
    var cells = (rows + 1) * stride
    var scores = List[Int32](length=cells, fill=Int32(0))
    var deletes = List[Int32](length=cells, fill=Int32(0))
    var inserts = List[Int32](length=cells, fill=Int32(0))

    scores[0] = 0
    for column in range(1, stride):
        if mode == AlignmentMode.GLOBAL:
            scores[column] = scoring.open + Int32(column - 1) * scoring.extend
            deletes[column] = scores[column] + scoring.open + scoring.extend
        else:
            scores[column] = 0
            deletes[column] = scoring.open + scoring.extend

    var best = Int32(0)
    var best_row = 0
    var best_column = 0

    for row in range(1, rows + 1):
        var base = row * stride
        var above = base - stride
        if mode == AlignmentMode.GLOBAL:
            scores[base] = scoring.open + Int32(row - 1) * scoring.extend
            inserts[base] = scores[base] + scoring.open + scoring.extend
        else:
            scores[base] = 0
            inserts[base] = scoring.open + scoring.extend

        for column in range(1, stride):
            var substitution = Int32(substitutions[Int(first[row - 1]) * alphabet_size + Int(second[column - 1])])
            var cell = gotoh_cell[mode](
                scores[above + column - 1],
                scores[above + column],
                deletes[above + column],
                scores[base + column - 1],
                inserts[base + column - 1],
                substitution,
                scoring,
            )
            scores[base + column] = cell.score
            deletes[base + column] = cell.deletion
            inserts[base + column] = cell.insertion

            comptime if mode == AlignmentMode.LOCAL:
                if cell.score > best:
                    best = cell.score
                    best_row = row
                    best_column = column

    var start_row = rows
    var start_column = columns
    if mode == AlignmentMode.LOCAL:
        start_row = best_row
        start_column = best_column

    var reconstruction = reconstruct(
        scores,
        deletes,
        inserts,
        stride,
        first,
        second,
        substitutions,
        alphabet_size,
        start_row,
        start_column,
        alphabet,
        scoring,
        mode,
    )
    var final_score = scores[start_row * stride + start_column]
    return AlignmentResult(final_score, reconstruction[0], reconstruction[1])


def reconstruct(
    scores: ImmSpan[Int32, _],
    deletes: ImmSpan[Int32, _],
    inserts: ImmSpan[Int32, _],
    stride: Int,
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    start_row: Int,
    start_column: Int,
    alphabet: String,
    scoring: AffineGapCosts,
    mode: AlignmentMode,
) -> Tuple[String, String]:
    """Three-state backward walk over the match, deletion and insertion layers.

    The Python walks a single layer: after a deletion anti_diagonal it reads the winning operation of the
    next cell instead of asking whether that deletion was opened or extended, so it can split one
    run into two and return a path that does not achieve its own reported score. Carrying the state
    fixes that. Ties resolve towards opening, which leaves the linear-gap case walking exactly as
    the Python does.
    """
    var letters = alphabet.as_bytes()
    var first_reversed = List[UInt8]()
    var second_reversed = List[UInt8]()
    var row = start_row
    var column = start_column
    var state = Layer.ALIGNING

    while row > 0 and column > 0:
        var here = row * stride + column
        var substitution = Int32(substitutions[Int(first[row - 1]) * alphabet_size + Int(second[column - 1])])
        var decision = decide[AlignmentMode.LOCAL](
            Cell(scores[here], deletes[here], inserts[here]),
            scores[here - stride - 1] + substitution,
            Cell(scores[here - stride], deletes[here - stride], inserts[here - stride]),
            Cell(scores[here - 1], deletes[here - 1], inserts[here - 1]),
            scoring,
        )
        """
        `mode` is a runtime argument here, and only the local walk below reads `reach`, so the flag is always computed
        and always gated at the point of use.
        """
        if mode == AlignmentMode.LOCAL and state == Layer.ALIGNING:
            if decision.reach() == PathReach.ENDS_HERE:
                break
        var anti_diagonal = advance(state, decision)
        first_reversed.append(letters[Int(first[row - 1])] if anti_diagonal.row_advance != 0 else GAP_BYTE)
        second_reversed.append(letters[Int(second[column - 1])] if anti_diagonal.column_advance != 0 else GAP_BYTE)
        row += anti_diagonal.row_advance
        column += anti_diagonal.column_advance
        state = anti_diagonal.lands_in

    # A global path must reach the origin, so what remains really is aligned against gaps. A local
    # path stops wherever the score falls to zero, and everything before that is outside it.
    if mode == AlignmentMode.GLOBAL:
        while row > 0:
            first_reversed.append(letters[Int(first[row - 1])])
            second_reversed.append(GAP_BYTE)
            row -= 1
        while column > 0:
            first_reversed.append(GAP_BYTE)
            second_reversed.append(letters[Int(second[column - 1])])
            column -= 1

    first_reversed.reverse()
    second_reversed.reverse()
    return (
        String(unsafe_from_utf8=first_reversed),
        String(unsafe_from_utf8=second_reversed),
    )


@fieldwise_init
struct SweepHalf(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which half of a Hirschberg split a sweep is computing.

    The reverse half walks both sequences from their far ends, which is how it is computed
    without a second recurrence, and it leaves its frontier in the reverse band pair.
    """

    var identifier: UInt8
    """Which case this names."""
    comptime FORWARD = Self(0)
    """Sweeping from the top-left corner towards the split."""
    comptime REVERSE = Self(1)
    """Sweeping from the bottom-right corner back towards it."""


@fieldwise_init
struct Frame(Copyable, Movable, TrivialRegisterPassable):
    """One pending subproblem of the Hirschberg recursion.

    Rows `[first_from, first_to)` against columns `[second_from, second_to)`, plus whether a
    deletion run already touches each horizontal edge.
    """

    var first_from: Int
    """First row of the sub-rectangle."""
    var first_to: Int
    """One past its last row."""
    var second_from: Int
    """First column of the sub-rectangle."""
    var second_to: Int
    """One past its last column."""
    var top: GapRun
    """Whether a gap run is already open entering the frame from above."""
    var bottom: GapRun
    """Whether one is already open entering it from below."""


def sweep_bands[
    half: SweepHalf
](
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    first_from: Int,
    first_to: Int,
    second_from: Int,
    second_to: Int,
    entering_run: GapRun,
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    final_scores: MutSpan[Int32, _],
    final_deletes: MutSpan[Int32, _],
):
    """Linear-space sweep of one sub-rectangle, leaving the last row's two layers behind.

    `entering_run` carries the open deletion run across the boundary: an already-open run makes the
    first deletion cost only an extension.
    """
    var rows = first_to - first_from
    var columns = second_to - second_from
    var scores_above = List[Int32](length=columns + 1, fill=Int32(0))
    var deletes_above = List[Int32](length=columns + 1, fill=Int32(0))
    var scores_row = List[Int32](length=columns + 1, fill=Int32(0))
    var deletes_row = List[Int32](length=columns + 1, fill=Int32(0))
    var inserts_row = List[Int32](length=columns + 1, fill=Int32(0))

    scores_above[0] = 0
    deletes_above[0] = scoring.open + scoring.extend
    # An open run arrives at the top-left corner only. Reaching any other cell of the top row
    # means the run already ended, so a deletion from there pays a fresh opening.
    for column in range(1, columns + 1):
        scores_above[column] = scoring.open + Int32(column - 1) * scoring.extend
        deletes_above[column] = scores_above[column] + scoring.open + scoring.extend

    for row in range(1, rows + 1):
        if entering_run == GapRun.EXTENDS:
            scores_row[0] = Int32(row) * scoring.extend
        else:
            scores_row[0] = scoring.open + Int32(row - 1) * scoring.extend
        deletes_row[0] = scores_row[0]
        inserts_row[0] = scores_row[0] + scoring.open + scoring.extend

        comptime reversed_order = half == SweepHalf.REVERSE
        var first_index = first_to - row if reversed_order else first_from + row - 1
        for column in range(1, columns + 1):
            var second_index = second_to - column if reversed_order else second_from + column - 1
            var substitution = Int32(substitutions[Int(first[first_index]) * alphabet_size + Int(second[second_index])])
            var cell = gotoh_cell[AlignmentMode.GLOBAL](
                scores_above[column - 1],
                scores_above[column],
                deletes_above[column],
                scores_row[column - 1],
                inserts_row[column - 1],
                substitution,
                scoring,
            )
            scores_row[column] = cell.score
            deletes_row[column] = cell.deletion
            inserts_row[column] = cell.insertion

        swap(scores_above, scores_row)
        swap(deletes_above, deletes_row)

    for column in range(columns + 1):
        final_scores[column] = scores_above[column]
        final_deletes[column] = deletes_above[column]


def solve_rectangle(
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    first_from: Int,
    first_to: Int,
    second_from: Int,
    second_to: Int,
    top: GapRun,
    bottom: GapRun,
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    path_columns: MutSpan[Int32, _],
    path_layers: MutSpan[Layer, _],
) -> Int32:
    """Solves one sub-rectangle outright and walks it back through the three layers.

    Writes `path_columns[i]` for `i` in `[first_from, first_to)` and `path_layers[i]` for `i` in
    `(first_from, first_to]`, so sibling subproblems tile the row axis without overlapping.
    Returns the corner score, which for a root-level call is the alignment's own score.
    """
    var rows = first_to - first_from
    var columns = second_to - second_from
    var stride = columns + 1
    var cells = (rows + 1) * stride
    var scores = List[Int32](length=cells, fill=Int32(0))
    var deletes = List[Int32](length=cells, fill=Int32(0))
    var inserts = List[Int32](length=cells, fill=Int32(0))

    scores[0] = 0
    # An open run arrives at the top-left corner only. Reaching any other cell of the top row
    # means the run already ended, so a deletion from there pays a fresh opening.
    for column in range(1, stride):
        scores[column] = scoring.open + Int32(column - 1) * scoring.extend
        deletes[column] = scores[column] + scoring.open + scoring.extend

    for row in range(1, rows + 1):
        var base = row * stride
        var above = base - stride
        if top == GapRun.EXTENDS:
            scores[base] = Int32(row) * scoring.extend
        else:
            scores[base] = scoring.open + Int32(row - 1) * scoring.extend
        deletes[base] = scores[base]
        inserts[base] = scores[base] + scoring.open + scoring.extend
        for column in range(1, stride):
            var substitution = Int32(
                substitutions[Int(first[first_from + row - 1]) * alphabet_size + Int(second[second_from + column - 1])]
            )
            var cell = gotoh_cell[AlignmentMode.GLOBAL](
                scores[above + column - 1],
                scores[above + column],
                deletes[above + column],
                scores[base + column - 1],
                inserts[base + column - 1],
                substitution,
                scoring,
            )
            scores[base + column] = cell.score
            deletes[base + column] = cell.deletion
            inserts[base + column] = cell.insertion

    var row = rows
    var column = columns
    var state = Layer.ALIGNING
    # An open run at the bottom edge is a fact the join established, not a candidate to weigh:
    # the two halves were scored on the assumption that this walk leaves in the deleting layer.
    if bottom == GapRun.EXTENDS:
        state = Layer.DELETING

    # Only a row-consuming anti_diagonal records anything, so sibling subproblems tile the row axis.
    while row > 0 and column > 0:
        var here = row * stride + column
        var substitution = Int32(
            substitutions[Int(first[first_from + row - 1]) * alphabet_size + Int(second[second_from + column - 1])]
        )
        var decision = decide[AlignmentMode.GLOBAL](
            Cell(scores[here], deletes[here], inserts[here]),
            scores[here - stride - 1] + substitution,
            Cell(scores[here - stride], deletes[here - stride], inserts[here - stride]),
            Cell(scores[here - 1], deletes[here - 1], inserts[here - 1]),
            scoring,
        )
        var anti_diagonal = advance(state, decision)
        if anti_diagonal.row_advance != 0:
            path_layers[first_from + row] = Layer.ALIGNING if anti_diagonal.column_advance != 0 else Layer.DELETING
            path_columns[first_from + row - 1] = Int32(second_from + column + anti_diagonal.column_advance)
        row += anti_diagonal.row_advance
        column += anti_diagonal.column_advance
        state = anti_diagonal.lands_in

    while row > 0:
        path_layers[first_from + row] = Layer.DELETING
        path_columns[first_from + row - 1] = Int32(second_from + column)
        row -= 1

    return scores[rows * stride + columns]


@fieldwise_init
struct Crossing(ImplicitlyCopyable, TrivialRegisterPassable):
    """Where a Myers-Miller join puts the cut, and what each of its two candidates scored."""

    var plain: Int32
    """Best score for a join that does not carry a gap across the cut."""
    var plain_column: Int
    """Column where that join puts the cut."""
    var gapped: Int32
    """Best score for a join that carries an open gap across it."""
    var gapped_column: Int
    """Column where that join puts the cut."""


def solve_frame(
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    frame: Frame,
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    path_columns: MutSpan[Int32, _],
    path_layers: MutSpan[Layer, _],
):
    """Solves one pending subproblem outright, which is how every branch of the recursion ends."""
    _ = solve_rectangle(
        first,
        second,
        frame.first_from,
        frame.first_to,
        frame.second_from,
        frame.second_to,
        frame.top,
        frame.bottom,
        substitutions,
        alphabet_size,
        scoring,
        path_columns,
        path_layers,
    )


def best_crossing(
    forward_scores: ImmSpan[Int32, _],
    forward_deletes: ImmSpan[Int32, _],
    reverse_scores: ImmSpan[Int32, _],
    reverse_deletes: ImmSpan[Int32, _],
    width: Int,
    scoring: AffineGapCosts,
) -> Crossing:
    """Scans the two frontiers for the best cut, in the match layer and across a straddling run."""
    var best = Crossing(NEGATIVE_INFINITY, 0, NEGATIVE_INFINITY, 0)
    var refund = scoring.extend - scoring.open
    for offset in range(width + 1):
        var plain = forward_scores[offset] + reverse_scores[width - offset]
        if plain > best.plain:
            best.plain = plain
            best.plain_column = offset
        var gapped = forward_deletes[offset] + reverse_deletes[width - offset] + refund
        if gapped > best.gapped:
            best.gapped = gapped
            best.gapped_column = offset
    return best


def serial_hirschberg(
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    window_first_from: Int,
    window_first_to: Int,
    window_second_from: Int,
    window_second_to: Int,
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    leaf_cells: Int,
    path_columns: MutSpan[Int32, _],
    path_layers: MutSpan[Layer, _],
) raises:
    """Linear-space traceback: split on rows, join the two halves, recurse without recursion.

    A substitution anti_diagonal advances `i + j` by two and can skip an anti-diagonal entirely, while the
    row index advances by exactly zero or one per anti_diagonal, so the cut has to be a row. The two halves
    are joined by Myers-Miller: either the path crosses in the match layer, or a deletion run
    straddles the cut, in which case both halves charged an opening and one is refunded.
    """
    if scoring.open > scoring.extend:
        raise AffineGapsError(ErrorKind.INVALID_SCORING, "gap opening cheaper than extension")

    var columns = window_second_to - window_second_from
    path_columns[window_first_to] = Int32(window_second_to)

    var frames = List[Frame]()
    frames.append(
        Frame(
            window_first_from,
            window_first_to,
            window_second_from,
            window_second_to,
            GapRun.OPENS,
            GapRun.OPENS,
        )
    )

    var forward_scores = List[Int32](length=columns + 1, fill=Int32(0))
    var forward_deletes = List[Int32](length=columns + 1, fill=Int32(0))
    var reverse_scores = List[Int32](length=columns + 1, fill=Int32(0))
    var reverse_deletes = List[Int32](length=columns + 1, fill=Int32(0))

    while len(frames) > 0:
        var frame = frames.pop()
        var first_from = frame.first_from
        var first_to = frame.first_to
        var second_from = frame.second_from
        var second_to = frame.second_to
        var top = frame.top
        var bottom = frame.bottom

        var height = first_to - first_from
        var width = second_to - second_from
        if height == 0:
            continue
        if width == 0 or height <= 2 or (height + 1) * (width + 1) <= leaf_cells:
            solve_frame(
                first,
                second,
                frame,
                substitutions,
                alphabet_size,
                scoring,
                path_columns,
                path_layers,
            )
            continue

        var split = (first_from + first_to) // 2
        sweep_bands[SweepHalf.FORWARD](
            first,
            second,
            first_from,
            split,
            second_from,
            second_to,
            top,
            substitutions,
            alphabet_size,
            scoring,
            forward_scores,
            forward_deletes,
        )
        sweep_bands[SweepHalf.REVERSE](
            first,
            second,
            split,
            first_to,
            second_from,
            second_to,
            bottom,
            substitutions,
            alphabet_size,
            scoring,
            reverse_scores,
            reverse_deletes,
        )

        var join = best_crossing(forward_scores, forward_deletes, reverse_scores, reverse_deletes, width, scoring)
        var best_plain = join.plain
        var best_plain_column = join.plain_column
        var best_gapped = join.gapped
        var best_gapped_column = join.gapped_column

        var crosses_in_a_gap = best_gapped > best_plain
        """
        Either the path crosses the cut in the aligning layer, or it crosses inside a deletion run. Both are ordinary,
        and the second is what the boundary flags exist to carry: the two halves each charged an opening for their
        part of the run, the refund removes one, and both children are told the run is already open at the edge they
        share.
        """
        var crossing = second_from + (best_gapped_column if crosses_in_a_gap else best_plain_column)
        var shared_edge = GapRun.EXTENDS if crosses_in_a_gap else GapRun.OPENS
        # Lower half first, so the upper half pops first and the two row ranges tile in order.
        frames.append(Frame(split, first_to, crossing, second_to, shared_edge, bottom))
        frames.append(Frame(first_from, split, second_from, crossing, top, shared_edge))


def score_path(
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    path_columns: ImmSpan[Int32, _],
    path_layers: ImmSpan[Layer, _],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    rows: Int,
) -> Int32:
    """Scores a reconstructed path under the affine rule, in time linear in the alignment.

    Deriving the score from the path rather than from a second dynamic-programming pass costs
    $O(rows + columns)$ instead of $O(rows * columns)$, and makes the reported score consistent
    with the returned strings by construction rather than by coincidence.
    """
    var total = Int32(0)
    var in_first = False
    var in_second = False

    for _ in range(Int(path_columns[0])):
        total += scoring.extend if in_first else scoring.open
        in_first = True
        in_second = False

    for row in range(1, rows + 1):
        var before = Int(path_columns[row - 1])
        var after = Int(path_columns[row])
        if path_layers[row] == Layer.DELETING:
            total += scoring.extend if in_second else scoring.open
            in_first = False
            in_second = True
        else:
            total += Int32(substitutions[Int(first[row - 1]) * alphabet_size + Int(second[before])])
            before += 1
            in_first = False
            in_second = False
        for _ in range(before, after):
            total += scoring.extend if in_first else scoring.open
            in_first = True
            in_second = False

    return total


def expand_path(
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    path_columns: ImmSpan[Int32, _],
    path_layers: ImmSpan[Layer, _],
    alphabet: String,
    mode: AlignmentMode,
    from_row: Int,
    rows: Int,
) -> Tuple[String, String]:
    """Turns the per-row crossing columns back into the two gapped strings.

    The walk covers rows `(from_row, rows]`. A global alignment spans the whole matrix and opens
    with however many insertions precede its first row; a local one spans only its own core, and
    the sequence outside that core is not part of the alignment.
    """
    var letters = alphabet.as_bytes()
    var left = List[UInt8]()
    var right = List[UInt8]()

    if mode == AlignmentMode.GLOBAL:
        for column in range(Int(path_columns[0])):
            left.append(GAP_BYTE)
            right.append(letters[Int(second[column])])
    for row in range(from_row + 1, rows + 1):
        var before = Int(path_columns[row - 1])
        var after = Int(path_columns[row])
        if path_layers[row] == Layer.DELETING:
            left.append(letters[Int(first[row - 1])])
            right.append(GAP_BYTE)
        else:
            left.append(letters[Int(first[row - 1])])
            right.append(letters[Int(second[before])])
            before += 1
        for column in range(before, after):
            left.append(GAP_BYTE)
            right.append(letters[Int(second[column])])
    return (String(unsafe_from_utf8=left), String(unsafe_from_utf8=right))


def serial_local_extremum[
    half: SweepHalf
](
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    first_to: Int,
    second_to: Int,
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
) -> Tuple[Int, Int, Int32]:
    """Linear-space local sweep returning the first row-major maximum and its value.

    Strict `>` recording the earliest maximum in row-major order, which is the cell the Python's scan
    settles on. Run backwards over the same prefixes, it instead reports how far the best local
    alignment reaches back, which is where the alignment starts.
    """
    var scores_above = List[Int32](length=second_to + 1, fill=Int32(0))
    var deletes_above = List[Int32](length=second_to + 1, fill=Int32(0))
    var scores_row = List[Int32](length=second_to + 1, fill=Int32(0))
    var deletes_row = List[Int32](length=second_to + 1, fill=Int32(0))
    var inserts_row = List[Int32](length=second_to + 1, fill=Int32(0))

    for column in range(1, second_to + 1):
        deletes_above[column] = scoring.open + scoring.extend

    var best = Int32(0)
    var best_row = 0
    var best_column = 0

    for row in range(1, first_to + 1):
        scores_row[0] = 0
        inserts_row[0] = scoring.open + scoring.extend
        comptime reversed_order = half == SweepHalf.REVERSE
        var first_index = first_to - row if reversed_order else row - 1
        for column in range(1, second_to + 1):
            var second_index = second_to - column if reversed_order else column - 1
            var substitution = Int32(substitutions[Int(first[first_index]) * alphabet_size + Int(second[second_index])])
            var cell = gotoh_cell[AlignmentMode.LOCAL](
                scores_above[column - 1],
                scores_above[column],
                deletes_above[column],
                scores_row[column - 1],
                inserts_row[column - 1],
                substitution,
                scoring,
            )
            scores_row[column] = cell.score
            deletes_row[column] = cell.deletion
            inserts_row[column] = cell.insertion
            if cell.score > best:
                best = cell.score
                best_row = row
                best_column = column
        swap(scores_above, scores_row)
        swap(deletes_above, deletes_row)

    return (best_row, best_column, best)


# endregion Serial Reference


# region GPU Wavefront


@always_inline
def block_argmax(
    scores: Pointer[Scalar[ScoreDType], MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    places: Pointer[Scalar[DType.int64], MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    best: Int32,
    best_place: Int64,
):
    """Block-wide maximum keeping the earliest cell in row-major order on a tie.

    The stdlib block reductions carry a scalar, and this one has to carry the place alongside the
    score to break ties the way the Python scan's strict `>` does, so the tree stays here.
    Thread zero holds the winner afterwards.

    Two stages: each warp collapses through registers, then one barrier and a walk over the per-warp
    winners. The shuffle travels downward rather than by butterfly because the tie rule recording the
    lower index, which only a directional reduction reproduces.
    """
    var lane = Int(thread_idx.x) % WARP_SIZE
    var winner = best
    var winner_place = best_place
    var reach = UInt32(WARP_SIZE // 2)
    while reach > 0:
        var theirs = shuffle_down(winner, reach)
        var their_place = shuffle_down(winner_place, reach)
        if theirs > winner or (theirs == winner and theirs != 0 and their_place < winner_place):
            winner = theirs
            winner_place = their_place
        reach //= 2
    if lane == 0:
        scores[unsafe_offset=Int(thread_idx.x) // WARP_SIZE] = winner
        places[unsafe_offset=Int(thread_idx.x) // WARP_SIZE] = winner_place
    barrier()
    if Int(thread_idx.x) == 0:
        for index in range(1, WARPS_PER_BLOCK):
            var theirs = scores[unsafe_offset=index]
            var their_place = places[unsafe_offset=index]
            if theirs > winner or (theirs == winner and theirs != 0 and their_place < winner_place):
                winner = theirs
                winner_place = their_place
        scores[unsafe_offset=0] = winner
        places[unsafe_offset=0] = winner_place


def device_scores[
    mode: AlignmentMode
](
    ctx: DeviceContext,
    sequences: ImmSpan[Scalar[SymbolDType], _],
    offsets: List[Scalar[OffsetDType]],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
) raises -> List[Int32]:
    """Scores every pair in the batch, one thread block each."""
    var pairs = (len(offsets) - 1) // 2
    if alphabet_size > MAX_ALPHABET_SIZE:
        raise AffineGapsError(ErrorKind.ALPHABET_TOO_LARGE, "substitution table")

    var longest_first = 0
    """The bands are indexed by row, so one launch only needs the longest first sequence it carries."""
    for pair in range(pairs):
        longest_first = max(longest_first, Int(offsets[2 * pair + 1]) - Int(offsets[2 * pair]))
    var band_stride = longest_first + 1
    var dynamic_bytes = 2 * band_stride * 4

    var sequences_buffer = upload(ctx, sequences)
    var offsets_buffer = upload(ctx, offsets)
    var substitutions_buffer = upload(ctx, substitutions)
    var results_buffer = zeroed[ScoreDType](ctx, pairs)
    var unused_symbols = ctx.enqueue_create_buffer[SymbolDType](1)
    """A discarding sweep never reads or writes these, but the one kernel still names them."""
    var unused_changes = ctx.enqueue_create_buffer[ChangeDType](1)
    var unused_offsets = zeroed[DType.int64](ctx, 1)
    var unused_lengths = zeroed[ScoreDType](ctx, 1)
    var nowhere = unused_symbols.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    """
    One placeholder fills every symbol slot. The origin cast is what lets it appear more than once in a launch, and it
    is sound because a discarding sweep reads none of them.
    """

    ctx.enqueue_function[strip_pair_kernel[mode, Recording.DISCARDED]](
        sequences_buffer.unsafe_ptr(),
        offsets_buffer.unsafe_ptr(),
        substitutions_buffer.unsafe_ptr(),
        nowhere,
        unused_changes.unsafe_ptr(),
        unused_offsets.unsafe_ptr(),
        results_buffer.unsafe_ptr(),
        nowhere,
        nowhere,
        unused_lengths.unsafe_ptr(),
        Int32(0),
        Int32(band_stride),
        Int32(alphabet_size),
        scoring.open,
        scoring.extend,
        grid_dim=pairs,
        block_dim=STRIP_LANES,
        shared_mem_bytes=dynamic_bytes,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(dynamic_bytes)),
    )
    ctx.synchronize()

    var results = List[Int32](capacity=pairs)
    with results_buffer.map_to_host() as host:
        for index in range(pairs):
            results.append(host[index])
    return results^


def strip_pair_kernel[
    mode: AlignmentMode, recording: Recording
](
    sequences: Pointer[Scalar[SymbolDType], MutAnyOrigin],
    offsets: Pointer[Scalar[OffsetDType], MutAnyOrigin],
    substitutions: Pointer[Scalar[SubstitutionDType], MutAnyOrigin],
    letters: Pointer[Scalar[SymbolDType], MutAnyOrigin],
    changes: Pointer[Scalar[ChangeDType], MutAnyOrigin],
    change_offsets: Pointer[Scalar[DType.int64], MutAnyOrigin],
    results: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    first_gapped: Pointer[Scalar[SymbolDType], MutAnyOrigin],
    second_gapped: Pointer[Scalar[SymbolDType], MutAnyOrigin],
    gapped_lengths: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    gapped_stride: Int32,
    carry_stride: Int32,
    alphabet_size: Int32,
    open: Int32,
    extend: Int32,
):
    """A recording strip: every decision is packed as the cell is computed, then walked back.

    A lane packs its `STRIP_COLUMNS` decisions into one word and stores it under `anti_diagonal` rather
    than `row`. The 32 lanes of one anti_diagonal sit on 32 different rows but share the anti_diagonal, so the
    anti_diagonal-major address makes the warp lay down 128 contiguous bytes where a row-major address
    would scatter the same warp over 32 sectors. The walk inverts it in closed form.

    Four bits hold a cell because `advance` reads exactly the source layer and the two run flags,
    and a clamped local cell borrows `Layer`'s spare source code instead of a fifth bit.
    """
    var pair = Int(block_idx.x)
    var first_start = Int(offsets[unsafe_offset=2 * pair])
    var second_start = Int(offsets[unsafe_offset=2 * pair + 1])
    var rows = second_start - first_start
    var columns = Int(offsets[unsafe_offset=2 * pair + 2]) - second_start
    var width = Int(alphabet_size)
    var scoring = AffineGapCosts(open, extend)
    var lane = Int(lane_id())
    var stride = columns + 1
    var tile = 0
    comptime if recording == Recording.TO_GLOBAL_MEMORY:
        tile = Int(change_offsets[unsafe_offset=pair])
    var strip_span = (rows + STRIP_LANES) * STRIP_LANES

    var table = stack_allocation[
        MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE,
        Scalar[SubstitutionDType],
        address_space=AddressSpace.SHARED,
    ]()
    for index in range(Int(thread_idx.x), width * width, Int(block_dim.x)):
        table[unsafe_offset=index] = substitutions[unsafe_offset=index]

    var reported = stack_allocation[1, Scalar[ScoreDType], address_space=AddressSpace.SHARED]()
    """
    The bottom-right cell belongs to whichever lane owns the last column, and the walk runs on lane zero, so the
    global answer crosses the warp through one shared word.
    """
    if thread_idx.x == 0:
        var span = rows + columns
        comptime if mode == AlignmentMode.GLOBAL:
            reported[unsafe_offset=0] = 0 if span == 0 else scoring.open + Int32(span - 1) * scoring.extend
        else:
            reported[unsafe_offset=0] = 0

    var carry = external_memory[Scalar[ScoreDType], address_space=AddressSpace.SHARED, alignment=16, name="carry"]()
    var carry_h = carry
    var carry_i = carry.unsafe_offset(Int(carry_stride))
    for index in range(Int(thread_idx.x), rows + 1, Int(block_dim.x)):
        var border = Int32(0)
        comptime if mode == AlignmentMode.GLOBAL:
            border = 0 if index == 0 else scoring.open + Int32(index - 1) * scoring.extend
        carry_h[unsafe_offset=index] = border
        carry_i[unsafe_offset=index] = border + scoring.open + scoring.extend
    barrier()

    var best = Int32(0)
    var best_place = Int64(0)
    var strips = ceildiv(columns, STRIP_WIDTH)

    for strip in range(strips):
        var first_column = strip * STRIP_WIDTH + lane * STRIP_COLUMNS
        var owned = min(max(columns - first_column, 0), STRIP_COLUMNS)

        var symbols = InlineArray[Int32, STRIP_COLUMNS](fill=0)
        comptime for k in range(STRIP_COLUMNS):
            var column = min(first_column + k, columns - 1)
            symbols[k] = Int32(sequences[unsafe_offset=second_start + max(column, 0)])

        var h = InlineArray[Int32, STRIP_COLUMNS](fill=0)
        var e = InlineArray[Int32, STRIP_COLUMNS](fill=0)
        comptime for k in range(STRIP_COLUMNS):
            comptime if mode == AlignmentMode.GLOBAL:
                h[k] = scoring.open + Int32(first_column + k) * scoring.extend
            e[k] = h[k] + scoring.open + scoring.extend

        var past = first_column + STRIP_COLUMNS
        var edge_h = Int32(0)
        comptime if mode == AlignmentMode.GLOBAL:
            edge_h = scoring.open + Int32(past - 1) * scoring.extend
        var edge_i = edge_h + scoring.open + scoring.extend

        var above_left_carry = Int32(0)
        comptime if mode == AlignmentMode.GLOBAL:
            above_left_carry = 0 if first_column == 0 else scoring.open + Int32(first_column - 1) * scoring.extend

        for anti_diagonal in range(1, rows + STRIP_LANES + 1):
            var row = anti_diagonal - lane
            var left_h = shuffle_up(edge_h, 1)
            var left_i = shuffle_up(edge_i, 1)
            if lane == 0:
                var here = min(max(row, 0), rows)
                left_h = carry_h[unsafe_offset=here]
                left_i = carry_i[unsafe_offset=here]
            var above_left_score = above_left_carry
            above_left_carry = left_h

            if row >= 1 and row <= rows:
                var symbol = Int(sequences[unsafe_offset=first_start + row - 1])
                var above_left = above_left_score
                var running_h = left_h
                var running_i = left_i
                var packed = UInt32(0)
                comptime for k in range(STRIP_COLUMNS):
                    var substitution = Int32(table[unsafe_offset=symbol * width + Int(symbols[k])])
                    var computed = gotoh_cell[mode](above_left, h[k], e[k], running_h, running_i, substitution, scoring)
                    comptime if recording == Recording.TO_GLOBAL_MEMORY:
                        var decision = decide[mode](
                            computed,
                            above_left + substitution,
                            Cell(h[k], e[k], 0),
                            Cell(running_h, 0, running_i),
                            scoring,
                        )
                        """Both neighbours the decision needs are still in registers, unread."""
                        packed |= decision.nibble() << UInt32(4 * k)
                    above_left = h[k]
                    h[k] = computed.score
                    e[k] = computed.deletion
                    running_h = computed.score
                    running_i = computed.insertion
                    comptime if mode == AlignmentMode.LOCAL:
                        if k < owned:
                            var place = Int64(row) * Int64(stride) + Int64(first_column + k + 1)
                            if computed.score > best:
                                best = computed.score
                                best_place = place
                            elif computed.score == best and computed.score != 0 and place < best_place:
                                best_place = place
                comptime if recording == Recording.TO_GLOBAL_MEMORY:
                    changes[unsafe_offset=tile + strip * strip_span + anti_diagonal * STRIP_LANES + lane] = packed
                edge_h = running_h
                edge_i = running_i
                if lane == STRIP_LANES - 1 and owned > 0:
                    carry_h[unsafe_offset=row] = h[STRIP_COLUMNS - 1]
                    carry_i[unsafe_offset=row] = running_i
                comptime if mode == AlignmentMode.GLOBAL:
                    if row == rows and owned > 0 and first_column + owned == columns:
                        reported[unsafe_offset=0] = h[owned - 1]
        barrier()

    # Warp-wide, keeping the earliest cell in row-major order on a tie, which is what the
    # Python scan's strict `>` picks. Every lane holds the winner afterwards.
    comptime if mode == AlignmentMode.LOCAL:
        var span = UInt32(1)
        while span < UInt32(STRIP_LANES):
            var theirs = shuffle_xor(best, span)
            var their_place = shuffle_xor(best_place, span)
            if theirs > best or (theirs == best and theirs != 0 and their_place < best_place):
                best = theirs
                best_place = their_place
            span *= 2

    if thread_idx.x != 0:
        return

    var start_row = rows
    var start_column = columns
    var final_score = reported[unsafe_offset=0]
    comptime if mode == AlignmentMode.LOCAL:
        final_score = best
        start_row = Int(best_place // Int64(stride))
        start_column = Int(best_place % Int64(stride))
        if final_score == 0:
            start_row = 0
            start_column = 0

    results[unsafe_offset=pair] = final_score
    comptime if recording == Recording.DISCARDED:
        return

    var row = start_row
    var column = start_column
    var produced = 0
    var base = pair * Int(gapped_stride)
    var state = Layer.ALIGNING

    while row > 0 and column > 0:
        var offset = column - 1
        var walk_lane = (offset % STRIP_WIDTH) // STRIP_COLUMNS
        var word = changes[
            unsafe_offset=tile + (offset // STRIP_WIDTH) * strip_span + (row + walk_lane) * STRIP_LANES + walk_lane
        ]
        var decision = CellDecision.unpacking(word >> UInt32(4 * (offset % STRIP_COLUMNS)))
        if mode == AlignmentMode.LOCAL and state == Layer.ALIGNING:
            if decision.reach() == PathReach.ENDS_HERE:
                break
        var anti_diagonal = advance(state, decision)
        var gap = Scalar[SymbolDType](GAP_BYTE)
        first_gapped[unsafe_offset=base + produced] = (
            letters[unsafe_offset=Int(sequences[unsafe_offset=first_start + row - 1])] if anti_diagonal.row_advance
            != 0 else gap
        )
        second_gapped[unsafe_offset=base + produced] = (
            letters[
                unsafe_offset=Int(sequences[unsafe_offset=second_start + column - 1])
            ] if anti_diagonal.column_advance
            != 0 else gap
        )
        row += anti_diagonal.row_advance
        column += anti_diagonal.column_advance
        produced += 1
        state = anti_diagonal.lands_in

    # Only a global path is required to reach the origin; see the host reconstruction.
    if mode == AlignmentMode.GLOBAL:
        while row > 0:
            first_gapped[unsafe_offset=base + produced] = letters[
                unsafe_offset=Int(sequences[unsafe_offset=first_start + row - 1])
            ]
            second_gapped[unsafe_offset=base + produced] = Scalar[SymbolDType](GAP_BYTE)
            row -= 1
            produced += 1
        while column > 0:
            first_gapped[unsafe_offset=base + produced] = Scalar[SymbolDType](GAP_BYTE)
            second_gapped[unsafe_offset=base + produced] = letters[
                unsafe_offset=Int(sequences[unsafe_offset=second_start + column - 1])
            ]
            column -= 1
            produced += 1

    gapped_lengths[unsafe_offset=pair] = Int32(produced)


def device_alignments[
    mode: AlignmentMode
](
    ctx: DeviceContext,
    sequences: ImmSpan[Scalar[SymbolDType], _],
    offsets: List[Scalar[OffsetDType]],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet: String,
    scoring: AffineGapCosts,
) raises -> List[AlignmentResult]:
    """Aligns a batch on the GPU by recording every decision, one thread block per pair."""
    var alphabet_bytes = alphabet.as_bytes()
    var alphabet_size = len(alphabet_bytes)
    if alphabet_size > MAX_ALPHABET_SIZE:
        raise AffineGapsError(ErrorKind.ALPHABET_TOO_LARGE, "substitution table")
    var pairs = (len(offsets) - 1) // 2

    var change_offsets = List[Int64](capacity=pairs + 1)
    var running = Int64(0)
    var widest = 1
    var longest_first = 0
    for pair in range(pairs):
        var rows = Int(offsets[2 * pair + 1]) - Int(offsets[2 * pair])
        longest_first = max(longest_first, rows)
        var columns = Int(offsets[2 * pair + 2]) - Int(offsets[2 * pair + 1])
        change_offsets.append(running)
        var strips = ceildiv(columns, STRIP_WIDTH)
        running += Int64(strips) * Int64(rows + STRIP_LANES) * Int64(STRIP_LANES)
        widest = max(widest, rows + columns)
    change_offsets.append(running)

    var sequences_buffer = upload(ctx, sequences)
    var offsets_buffer = upload(ctx, offsets)
    var substitutions_buffer = upload(ctx, substitutions)
    var letters = List[Scalar[SymbolDType]](capacity=alphabet_size)
    for index in range(alphabet_size):
        letters.append(Scalar[SymbolDType](alphabet_bytes[index]))
    var letters_buffer = upload(ctx, letters)
    var changes_buffer = ctx.enqueue_create_buffer[ChangeDType](Int(max(running, Int64(1))))
    var change_offsets_buffer = upload(ctx, change_offsets)
    var results_buffer = zeroed[ScoreDType](ctx, pairs)
    var first_buffer = ctx.enqueue_create_buffer[SymbolDType](max(pairs * widest, 1))
    var second_buffer = ctx.enqueue_create_buffer[SymbolDType](max(pairs * widest, 1))
    var lengths_buffer = zeroed[ScoreDType](ctx, pairs)

    ctx.enqueue_function[strip_pair_kernel[mode, Recording.TO_GLOBAL_MEMORY]](
        sequences_buffer.unsafe_ptr(),
        offsets_buffer.unsafe_ptr(),
        substitutions_buffer.unsafe_ptr(),
        letters_buffer.unsafe_ptr(),
        changes_buffer.unsafe_ptr(),
        change_offsets_buffer.unsafe_ptr(),
        results_buffer.unsafe_ptr(),
        first_buffer.unsafe_ptr(),
        second_buffer.unsafe_ptr(),
        lengths_buffer.unsafe_ptr(),
        Int32(widest),
        Int32(longest_first + 1),
        Int32(alphabet_size),
        scoring.open,
        scoring.extend,
        grid_dim=pairs,
        block_dim=STRIP_LANES,
        shared_mem_bytes=2 * (longest_first + 1) * 4,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(2 * (longest_first + 1) * 4)),
    )
    ctx.synchronize()

    var aligned = List[AlignmentResult](capacity=pairs)
    with results_buffer.map_to_host() as scores_host, lengths_buffer.map_to_host() as lengths_host, first_buffer.map_to_host() as first_host, second_buffer.map_to_host() as second_host:
        for pair in range(pairs):
            var produced = Int(lengths_host[pair])
            var base = pair * widest
            var left = List[UInt8](capacity=produced + 1)
            var right = List[UInt8](capacity=produced + 1)
            for index in range(produced - 1, -1, -1):
                left.append(UInt8(first_host[base + index]))
                right.append(UInt8(second_host[base + index]))
            aligned.append(
                AlignmentResult(
                    scores_host[pair],
                    String(unsafe_from_utf8=left),
                    String(unsafe_from_utf8=right),
                )
            )
    return aligned^


def device_hirschberg(
    ctx: DeviceContext,
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    window_first_from: Int,
    window_first_to: Int,
    window_second_from: Int,
    window_second_to: Int,
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    leaf_cells: Int,
    placement: Placement,
    path_columns: MutSpan[Int32, _],
    path_layers: MutSpan[Layer, _],
) raises:
    """Hirschberg with every sweep on the device.

    The recursion itself is a few hundred bookkeeping steps and stays on the host; the sweeps carry
    the whole quadratic cost and run as kernels over global-memory bands, so nothing here is bounded
    by shared memory. Each launch is its own barrier, which is what stands in for the grid-wide sync
    Mojo does not expose.
    """
    if scoring.open > scoring.extend:
        raise AffineGapsError(ErrorKind.INVALID_SCORING, "gap opening cheaper than extension")

    var rows = len(first)
    var columns = len(second)
    path_columns[window_first_to] = Int32(window_second_to)

    var sequences = List[Scalar[SymbolDType]](capacity=rows + columns)
    for index in range(rows):
        sequences.append(first[index])
    for index in range(columns):
        sequences.append(second[index])

    var buffers = device_sweep_buffers(ctx, rows, columns, Span(sequences), substitutions)

    var frames = List[Frame]()
    frames.append(
        Frame(
            window_first_from,
            window_first_to,
            window_second_from,
            window_second_to,
            GapRun.OPENS,
            GapRun.OPENS,
        )
    )

    # Level-synchronous rather than depth-first: every frame of a level is independent, so they
    # sweep together instead of taking turns. Frames at a level partition both axes, so they share
    # the frontier arrays without touching.
    while len(frames) > 0:
        var splitting = List[Frame]()
        var leaves = List[Frame]()
        for index in range(len(frames)):
            var frame = frames[index]
            var height = frame.first_to - frame.first_from
            var width = frame.second_to - frame.second_from
            if height == 0:
                continue
            if width == 0 or height <= 2 or (height + 1) * (width + 1) <= leaf_cells:
                leaves.append(frame)
            else:
                splitting.append(frame)

        # Leaves tile the row axis without overlapping, so each writes its own slice of the path
        # and they need no ordering between them. A fork-join costs milliseconds, so the shallow
        # levels, which hold only a handful of leaves, still run them in place.
        if len(leaves) >= PARALLEL_LEAF_FLOOR:

            @parameter
            def solve_leaf(slot: Int):
                solve_frame(
                    first,
                    second,
                    leaves[slot],
                    substitutions,
                    alphabet_size,
                    scoring,
                    path_columns,
                    path_layers,
                )

            parallelize[solve_leaf](len(leaves), placement.threads)
        else:
            for slot in range(len(leaves)):
                solve_frame(
                    first,
                    second,
                    leaves[slot],
                    substitutions,
                    alphabet_size,
                    scoring,
                    path_columns,
                    path_layers,
                )

        if len(splitting) == 0:
            break

        var sweeps = List[Sweep]()
        var joins = List[Scalar[ScoreDType]](length=len(splitting) * 3, fill=0)
        for index in range(len(splitting)):
            var frame = splitting[index]
            var split = (frame.first_from + frame.first_to) // 2
            var width = frame.second_to - frame.second_from
            # The second sequence is stored after the first, so its indices carry that offset.
            sweeps.append(
                Sweep(
                    split - frame.first_from,
                    width,
                    frame.first_from,
                    rows + frame.second_from,
                    frame.first_from,
                    frame.second_from,
                    frame.top,
                    SweepHalf.FORWARD,
                )
            )
            sweeps.append(
                Sweep(
                    frame.first_to - split,
                    width,
                    split,
                    rows + frame.second_from,
                    split,
                    frame.second_from,
                    frame.bottom,
                    SweepHalf.REVERSE,
                )
            )
            joins[index * 3 + 2] = Scalar[ScoreDType](width)

        device_sweep_level[AlignmentMode.GLOBAL](ctx, buffers, sweeps, alphabet_size, scoring)

        for index in range(len(splitting)):
            joins[index * 3] = Scalar[ScoreDType](sweeps[index * 2].column_base)
            joins[index * 3 + 1] = Scalar[ScoreDType](sweeps[index * 2 + 1].column_base)
        var joins_buffer = upload(ctx, Span(joins))
        ctx.enqueue_function[crossing_kernel](
            buffers.top_scores.unsafe_ptr(),
            buffers.top_deletes.unsafe_ptr(),
            buffers.reverse_scores.unsafe_ptr(),
            buffers.reverse_deletes.unsafe_ptr(),
            joins_buffer.unsafe_ptr(),
            buffers.crossing.unsafe_ptr(),
            scoring.extend - scoring.open,
            grid_dim=len(splitting),
            block_dim=THREADS_PER_BLOCK,
        )
        ctx.synchronize()

        var children = List[Frame]()
        with buffers.crossing.map_to_host() as host:
            for index in range(len(splitting)):
                var frame = splitting[index]
                var split = (frame.first_from + frame.first_to) // 2
                var best_plain = host[index * 4]
                var best_plain_column = Int(host[index * 4 + 1])
                var best_gapped = host[index * 4 + 2]
                var best_gapped_column = Int(host[index * 4 + 3])
                var crosses_in_a_gap = best_gapped > best_plain
                """
                Either the path crosses in the aligning layer, or inside a deletion run. Both are ordinary; the second
                is what the boundary flags carry, since the two halves each charged an opening and the refund removes
                one.
                """
                var crossing = frame.second_from + (best_gapped_column if crosses_in_a_gap else best_plain_column)
                var shared_edge = GapRun.EXTENDS if crosses_in_a_gap else GapRun.OPENS
                children.append(Frame(frame.first_from, split, frame.second_from, crossing, frame.top, shared_edge))
                children.append(Frame(split, frame.first_to, crossing, frame.second_to, shared_edge, frame.bottom))
        frames = children^


comptime DEFAULT_LEAF_CELLS = 4096

comptime PARALLEL_LEAF_FLOOR = 256
"""
A fork-join costs a few milliseconds, and the shallow levels of the recursion hold only a handful of leaves, so below
this many the dispatch costs more than the work it spreads.
"""

comptime DEVICE_STORED_CELLS = 1_000_000
"""
One pair on the stored device path gets a single warp, while the linear recursion spreads the same matrix over the
whole machine, so the device crossover sits far below what device memory would allow. Measured here: level near a
million cells, and the recursion is three times faster by sixteen million. A batch inverts the argument, since the
recursion takes its pairs in turn, and recording the caller's budget.
"""

comptime STRIP_COLUMNS = 8
comptime STRIP_LANES = 32
comptime STRIP_WIDTH = STRIP_COLUMNS * STRIP_LANES


comptime TILE_SIDE = STRIP_WIDTH

comptime MIN_TILE_HEIGHT = 32
"""Below this a tile spends more steps ramping the skew in and out than sweeping rows."""

comptime TARGET_TILES = 4224
"""
Warps a level aims to put in flight at once. Past roughly this many the strip stops gaining, so further splitting only
pays the skew ramp again.
"""


comptime PlanDType = DType.int64
"""
rows, columns, first_from, second_from, row_base, column_base, corner_base, entering_run, half, tile_rows_count,
tile_columns_count, tile_height A device buffer is typed by `DType`, so the plan travels as words and is read back
through this struct on both sides. Naming the fields once is what recording the host writer and the two device readers
from disagreeing about a position seven hundred lines apart.
"""


@fieldwise_init
struct Recording(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Where a sweep puts the decision it makes for each cell, if it recording one at all.

    A sweep that only needs a score throws every decision away; one that has to reconstruct recording
    them, and where they go is what separates a batch pair from a recursion leaf.
    """

    var identifier: UInt8
    """Which case this names."""
    comptime DISCARDED = Self(0)
    """The sweep recording only scores, because the caller will recurse instead."""
    comptime TO_GLOBAL_MEMORY = Self(1)
    """The sweep stores one decision byte per cell for a direct traceback."""


@fieldwise_init
struct SweepPlan(ImplicitlyCopyable, TrivialRegisterPassable):
    """One sub-rectangle's assignment: what to sweep, and where its edges live."""

    var rows: Int64
    """How many rows this sub-rectangle covers."""
    var columns: Int64
    """How many columns it covers."""
    var first_from: Int64
    """First row of the sub-rectangle, in the full sequence."""
    var second_from: Int64
    """First column of the sub-rectangle, in the full sequence."""
    var row_base: Int64
    """Where its row-indexed scratch begins."""
    var column_base: Int64
    """Where its column-indexed scratch begins."""
    var corner_base: Int64
    """Where its tile-corner scratch begins."""
    var tile_rows_count: Int64
    """Tiles down."""
    var tile_columns_count: Int64
    """Tiles across."""
    var tile_height: Int64
    """Rows per tile."""
    var entering_run: GapRun
    """Whether a gap run is already open where the sweep starts."""
    var half: SweepHalf
    """Which direction the sweep runs."""


comptime PLAN_WORDS = size_of[SweepPlan]() // size_of[Int64]()


def tiled_sweep_kernel[
    mode: AlignmentMode
](
    sequences: Pointer[Scalar[SymbolDType], MutAnyOrigin],
    substitutions: Pointer[Scalar[SubstitutionDType], MutAnyOrigin],
    plans: Pointer[Scalar[PlanDType], MutAnyOrigin],
    block_best: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    block_place: Pointer[Scalar[DType.int64], MutAnyOrigin],
    forward_scores: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    forward_deletes: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    reverse_scores: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    reverse_deletes: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    left_scores: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    left_inserts: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    corner_scores: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    tile_anti_diagonal: Int32,
    widest_tiles: Int32,
    alphabet_size: Int32,
    open: Int32,
    extend: Int32,
):
    """One warp per tile, one tile-anti-diagonal per launch, every sweep of a level at once.

    Tiles sharing an anti-diagonal are independent, so a launch boundary supplies the barrier that
    Mojo 1.0 has no grid-wide primitive for. The flat block index carries both the sweep and the
    tile, which is what lets independent subproblems share a launch instead of taking turns. It is
    flat rather than two-dimensional because a deep recursion level holds more sweeps than the
    65535 a grid's second dimension allows.

    Inside a tile the sweep is a skewed register strip: lane `t` owns `STRIP_COLUMNS` columns and
    sits on row `s - t`, so the left-neighbour dependency travels by one `shuffle_up` and the tile
    runs to completion without a single barrier.
    """
    var lanes_wide = Int(widest_tiles)
    var sweep_index = Int(block_idx.x) // lanes_wide
    var tile_slot = Int(block_idx.x) % lanes_wide
    var sweep = plans.unsafe_bitcast[SweepPlan]()[unsafe_offset=sweep_index]
    var rows = Int(sweep.rows)
    var columns = Int(sweep.columns)
    var first_from = Int(sweep.first_from)
    var second_from = Int(sweep.second_from)
    var row_base = Int(sweep.row_base)
    var column_base = Int(sweep.column_base)
    var corner_base = Int(sweep.corner_base)
    var extends = sweep.entering_run == GapRun.EXTENDS
    var reversed_order = sweep.half == SweepHalf.REVERSE
    var tile_rows_count = Int(sweep.tile_rows_count)
    var tile_columns_count = Int(sweep.tile_columns_count)
    comptime local = mode == AlignmentMode.LOCAL
    var tile_height = Int(sweep.tile_height)
    var corner_rows = tile_rows_count + 1
    """
    A corner is written on one tile-anti-diagonal and read on the next but one, by exactly one tile, so three rotating
    buffers of one entering_run per tile row hold every live corner at once.
    """
    var slot = Int(block_idx.x)

    var tile_row_low = Int(tile_anti_diagonal) - min(Int(tile_anti_diagonal), tile_columns_count - 1)
    var tile_row_high = min(Int(tile_anti_diagonal), tile_rows_count - 1)
    var tile_row = tile_row_low + tile_slot
    if tile_row > tile_row_high:
        return
    var tile_column = Int(tile_anti_diagonal) - tile_row
    var row_begin = tile_row * tile_height
    var column_begin = tile_column * TILE_SIDE
    var height = min(tile_height, rows - row_begin)
    var width_span = min(TILE_SIDE, columns - column_begin)
    if height <= 0 or width_span <= 0:
        return

    var width = Int(alphabet_size)
    var scoring = AffineGapCosts(open, extend)
    var lane = Int(lane_id())
    var first_column = lane * STRIP_COLUMNS
    var owned = min(max(width_span - first_column, 0), STRIP_COLUMNS)
    var top_scores = reverse_scores if reversed_order else forward_scores
    var top_deletes = reverse_deletes if reversed_order else forward_deletes

    var table = stack_allocation[
        MAX_ALPHABET_SIZE * MAX_ALPHABET_SIZE,
        Scalar[SubstitutionDType],
        address_space=AddressSpace.SHARED,
    ]()
    for index in range(Int(thread_idx.x), width * width, Int(block_dim.x)):
        table[unsafe_offset=index] = substitutions[unsafe_offset=index]

    var edge_scores = stack_allocation[TILE_SIDE, Scalar[ScoreDType], address_space=AddressSpace.SHARED]()
    """
    The tile's left column, staged once by the whole warp. Lane zero consumes one entering_run per anti_diagonal, and a global load
    there would sit on the dependency chain that feeds every shuffle. Staging also decouples the read of the
    neighbour's frontier from this tile's write of its own, which land in the same slots.
    """
    var edge_inserts = stack_allocation[TILE_SIDE, Scalar[ScoreDType], address_space=AddressSpace.SHARED]()
    for index in range(Int(thread_idx.x), height, Int(block_dim.x)):
        var row = row_begin + index + 1
        if column_begin == 0:
            var border = Int32(0) if local else (Int32(row) * extend if extends else open + Int32(row - 1) * extend)
            edge_scores[unsafe_offset=index] = border
            edge_inserts[unsafe_offset=index] = border + open + extend
        else:
            edge_scores[unsafe_offset=index] = left_scores[unsafe_offset=row_base + row]
            edge_inserts[unsafe_offset=index] = left_inserts[unsafe_offset=row_base + row]
    barrier()

    var symbols = InlineArray[Int32, STRIP_COLUMNS](fill=0)
    """
    The tile's top row, for the columns this lane owns. Outside the matrix it is the affine ramp, or zero for a local
    sweep; inside it is whatever the tile above left behind.
    """
    var h = InlineArray[Int32, STRIP_COLUMNS](fill=0)
    var e = InlineArray[Int32, STRIP_COLUMNS](fill=0)
    comptime for k in range(STRIP_COLUMNS):
        var column = column_begin + min(first_column + k, width_span - 1) + 1
        var right_index = second_from + columns - column if reversed_order else second_from + column - 1
        symbols[k] = Int32(sequences[unsafe_offset=right_index])
        if local and row_begin == 0:
            h[k] = 0
            e[k] = open + extend
        elif row_begin == 0:
            h[k] = open + Int32(column - 1) * extend
            e[k] = h[k] + open + extend
        else:
            h[k] = top_scores[unsafe_offset=column_base + column]
            e[k] = top_deletes[unsafe_offset=column_base + column]

    var edge_h = h[STRIP_COLUMNS - 1]
    var edge_i = edge_h + open + extend

    var above_left_carry = Int32(0)
    """
    The cell above and to the left of this lane's first one. For lane zero that is the tile's own corner; for every
    other lane it is a cell of the top row.
    """
    if lane == 0:
        if local and (row_begin == 0 or column_begin == 0):
            above_left_carry = 0
        elif row_begin == 0 and column_begin == 0:
            above_left_carry = 0
        elif row_begin == 0:
            above_left_carry = open + Int32(column_begin - 1) * extend
        elif column_begin == 0:
            above_left_carry = Int32(row_begin) * extend if extends else open + Int32(row_begin - 1) * extend
        else:
            above_left_carry = corner_scores[
                unsafe_offset=corner_base + (Int(tile_anti_diagonal) % 3) * corner_rows + tile_row
            ]
    else:
        var column = column_begin + first_column
        if local and row_begin == 0:
            above_left_carry = 0
        elif row_begin == 0:
            above_left_carry = open + Int32(column - 1) * extend
        else:
            above_left_carry = top_scores[unsafe_offset=column_base + column]

    var best_cell = Int32(0)
    var best_at = Int64(0)

    for anti_diagonal in range(1, height + STRIP_LANES + 1):
        var local_row = anti_diagonal - lane
        var left_h = shuffle_up(edge_h, 1)
        var left_i = shuffle_up(edge_i, 1)
        if lane == 0:
            var index = min(max(local_row, 1), height) - 1
            left_h = edge_scores[unsafe_offset=index]
            left_i = edge_inserts[unsafe_offset=index]
        var above_left_score = above_left_carry
        above_left_carry = left_h

        if local_row >= 1 and local_row <= height:
            var row = row_begin + local_row
            var left_index = first_from + rows - row if reversed_order else first_from + row - 1
            var symbol = Int(sequences[unsafe_offset=left_index])
            var above_left = above_left_score
            var running_h = left_h
            var running_i = left_i
            comptime for k in range(STRIP_COLUMNS):
                var substitution = Int32(table[unsafe_offset=symbol * width + Int(symbols[k])])
                var computed = gotoh_cell[mode](above_left, h[k], e[k], running_h, running_i, substitution, scoring)
                var cell = computed.score
                if local and k < owned and cell > best_cell:
                    best_cell = cell
                    best_at = Int64(row) * Int64(columns + 1) + Int64(column_begin + first_column + k + 1)
                above_left = h[k]
                h[k] = cell
                e[k] = computed.deletion
                running_h = cell
                running_i = computed.insertion
            edge_h = running_h
            edge_i = running_i

            # A tile narrower than the full side is against the matrix's right edge, and nobody
            # reads its right column, so the gate is exact rather than conservative.
            if lane == STRIP_LANES - 1 and width_span == TILE_SIDE:
                left_scores[unsafe_offset=row_base + row] = running_h
                left_inserts[unsafe_offset=row_base + row] = running_i
            if local_row == height:
                # Column zero is the matrix border, so no lane computes it, but the crossing
                # reduction reads it as a candidate cut. Only the leftmost tile may publish it;
                # elsewhere that slot belongs to the tile on the left.
                if lane == 0 and column_begin == 0:
                    var border = Int32(0)
                    var border_delete = open + extend
                    if not local:
                        border = Int32(row) * extend if extends else open + Int32(row - 1) * extend
                        border_delete = border
                    top_scores[unsafe_offset=column_base] = border
                    top_deletes[unsafe_offset=column_base] = border_delete
                comptime for k in range(STRIP_COLUMNS):
                    if k < owned:
                        var column = column_begin + first_column + k + 1
                        top_scores[unsafe_offset=column_base + column] = h[k]
                        top_deletes[unsafe_offset=column_base + column] = e[k]
                if lane == STRIP_LANES - 1 and width_span == TILE_SIDE:
                    corner_scores[
                        unsafe_offset=corner_base + ((Int(tile_anti_diagonal) + 2) % 3) * corner_rows + tile_row + 1
                    ] = h[STRIP_COLUMNS - 1]

    # A local sweep reports the best cell this warp saw, for the host to reduce across blocks.
    if local:
        var span = UInt32(1)
        while span < UInt32(STRIP_LANES):
            var theirs = shuffle_xor(best_cell, span)
            var their_place = shuffle_xor(best_at, span)
            if theirs > best_cell or (theirs == best_cell and theirs != 0 and their_place < best_at):
                best_cell = theirs
                best_at = their_place
            span *= 2
        if thread_idx.x == 0:
            var held = block_best[unsafe_offset=slot]
            """One launch per tile-anti-diagonal, so this slot accumulates rather than replaces."""
            if best_cell > held or (best_cell == held and best_cell != 0 and best_at < block_place[unsafe_offset=slot]):
                block_best[unsafe_offset=slot] = best_cell
                block_place[unsafe_offset=slot] = best_at


@fieldwise_init
struct SweepBuffers(Movable):
    """Device scratch reused across every sweep of one alignment."""

    var sequences: DeviceBuffer[SymbolDType]
    """Both sequences, concatenated."""
    var substitutions: DeviceBuffer[SubstitutionDType]
    """The substitution table, staged once per alignment."""
    var top_scores: DeviceBuffer[ScoreDType]
    """Score frontier along the top edge of each frame."""
    var top_deletes: DeviceBuffer[ScoreDType]
    """Deletion frontier along the same edge."""
    var reverse_scores: DeviceBuffer[ScoreDType]
    """Score frontier of the reverse half."""
    var reverse_deletes: DeviceBuffer[ScoreDType]
    """Deletion frontier of the reverse half."""
    var crossing: DeviceBuffer[ScoreDType]
    """Where the two halves meet, which is what the join reads."""
    var left_scores: DeviceBuffer[ScoreDType]
    """Score carry down the left edge of a strip."""
    var left_inserts: DeviceBuffer[ScoreDType]
    """Insertion carry down the same edge."""
    var corner_scores: DeviceBuffer[ScoreDType]
    """Tile corners, which is how one tile hands off to the next."""
    var corner_span: Int
    """How much of each shared array a whole recursion level may claim."""
    var left_span: Int
    """How many entries the left carry holds."""
    var frontier_span: Int
    """How many entries a frontier holds."""
    var block_best: DeviceBuffer[ScoreDType]
    """Best score each block found, for the local-alignment scan."""
    var block_place: DeviceBuffer[DType.int64]
    """Where each block found it, so ties break on position."""
    var block_slots: Int
    """How many blocks the scan reduces over."""


def device_sweep_buffers(
    ctx: DeviceContext,
    rows: Int,
    columns: Int,
    sequences: ImmSpan[Scalar[SymbolDType], _],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
) raises -> SweepBuffers:
    """Device scratch for one alignment, sized so a whole recursion level fits side by side.

    Every sweep of a level claims its own slice, and a deep level is many tiny sweeps, so each
    array carries the real extent plus a constant per sweep. A level never holds more than two
    sweeps per row.

    The column extent is doubled because a frame's forward and reverse halves split its rows but
    both span all of its columns, so one level's sweeps cover the column axis twice.
    """
    var frontier_span = 2 * columns + 4 * rows + 32
    var left_span = 5 * rows + 32
    var tile_columns_count = ceildiv(columns, TILE_SIDE) + 2
    var corner_span = 3 * ceildiv(rows, MIN_TILE_HEIGHT) + 6 * rows + 32
    """
    Square-in-count tiling makes a sweep's tile grid as wide as it is tall, and a level's frames partition the
    columns, so the corner arrays of one level are bounded by twice the square of the full column count. Three
    rotating buffers per sweep, each one entering_run per tile row. Linear in sequence length, where the full tile grid was
    quadratic and overflowed its own int32 plan field.
    """
    var block_slots = min(ceildiv(rows, MIN_TILE_HEIGHT), tile_columns_count) + 4
    """Only a local scan writes here, and that is one sweep, so the grid is one tile-above_left wide."""
    var buffers = SweepBuffers(
        ctx.enqueue_create_buffer[SymbolDType](max(rows + columns, 1)),
        ctx.enqueue_create_buffer[SubstitutionDType](len(substitutions)),
        ctx.enqueue_create_buffer[ScoreDType](frontier_span),
        ctx.enqueue_create_buffer[ScoreDType](frontier_span),
        ctx.enqueue_create_buffer[ScoreDType](frontier_span),
        ctx.enqueue_create_buffer[ScoreDType](frontier_span),
        ctx.enqueue_create_buffer[ScoreDType](4 * max(rows, 1)),
        ctx.enqueue_create_buffer[ScoreDType](left_span),
        ctx.enqueue_create_buffer[ScoreDType](left_span),
        ctx.enqueue_create_buffer[ScoreDType](corner_span),
        corner_span,
        left_span,
        frontier_span,
        zeroed[ScoreDType](ctx, block_slots),
        zeroed[DType.int64](ctx, block_slots),
        block_slots,
    )
    ctx.enqueue_copy(buffers.sequences, sequences)
    ctx.enqueue_copy(buffers.substitutions, substitutions)
    return buffers^


def crossing_kernel(
    forward_scores: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    forward_deletes: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    reverse_scores: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    reverse_deletes: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    joins: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    crossing: Pointer[Scalar[ScoreDType], MutAnyOrigin],
    refund: Int32,
):
    """Picks where each split of a level crosses its cut, one block per split.

    Reading the frontiers back per split costs more than the sweeps once the recursion is
    thousands of nodes deep, so the reduction happens here and four numbers per split travel.
    """
    var split = Int(block_idx.x)
    var forward_base = Int(joins[unsafe_offset=split * 3])
    var reverse_base = Int(joins[unsafe_offset=split * 3 + 1])
    var span = Int(joins[unsafe_offset=split * 3 + 2])

    var plain_scores = stack_allocation[THREADS_PER_BLOCK, Scalar[ScoreDType], address_space=AddressSpace.SHARED]()
    var plain_places = stack_allocation[THREADS_PER_BLOCK, Scalar[DType.int64], address_space=AddressSpace.SHARED]()
    var gapped_scores = stack_allocation[THREADS_PER_BLOCK, Scalar[ScoreDType], address_space=AddressSpace.SHARED]()
    var gapped_places = stack_allocation[THREADS_PER_BLOCK, Scalar[DType.int64], address_space=AddressSpace.SHARED]()

    var best_plain = NEGATIVE_INFINITY
    var best_plain_column = Int64(0)
    var best_gapped = NEGATIVE_INFINITY
    var best_gapped_column = Int64(0)
    for offset in range(Int(thread_idx.x), span + 1, Int(block_dim.x)):
        var near = forward_base + offset
        var far = reverse_base + span - offset
        var plain = forward_scores[unsafe_offset=near] + reverse_scores[unsafe_offset=far]
        if plain > best_plain:
            best_plain = plain
            best_plain_column = Int64(offset)
        var gapped = forward_deletes[unsafe_offset=near] + reverse_deletes[unsafe_offset=far] + refund
        if gapped > best_gapped:
            best_gapped = gapped
            best_gapped_column = Int64(offset)

    block_argmax(plain_scores, plain_places, best_plain, best_plain_column)
    block_argmax(gapped_scores, gapped_places, best_gapped, best_gapped_column)
    if thread_idx.x == 0:
        crossing[unsafe_offset=split * 4] = plain_scores[unsafe_offset=0]
        crossing[unsafe_offset=split * 4 + 1] = Scalar[ScoreDType](plain_places[unsafe_offset=0])
        crossing[unsafe_offset=split * 4 + 2] = gapped_scores[unsafe_offset=0]
        crossing[unsafe_offset=split * 4 + 3] = Scalar[ScoreDType](gapped_places[unsafe_offset=0])


@fieldwise_init
struct Sweep(ImplicitlyCopyable, TrivialRegisterPassable):
    """One independent sub-rectangle sweep, as the plan-driven kernel needs to see it."""

    var rows: Int
    """How many rows this sub-rectangle covers."""
    var columns: Int
    """How many columns it covers."""
    var first_from: Int
    """First row of the sub-rectangle."""
    var second_from: Int
    """First column of the sub-rectangle."""
    var row_base: Int
    """Where its row-indexed scratch begins."""
    var column_base: Int
    """Where its column-indexed scratch begins."""
    var entering_run: GapRun
    """Whether a gap run is already open where the sweep starts."""
    var half: SweepHalf
    """Which direction the sweep runs."""

    def tile_height(self, sweeps_in_level: Int) -> Int:
        """The tallest tile that still fills the machine, given how many sweeps share the level.

        Concurrency is `sweeps_in_level * min(tile_rows, tile_columns)`, so shorter tiles buy
        parallelism only until the machine is full; past that they cost skew, since a tile `h`
        rows tall runs `h + 31` steps. Once `tile_columns` alone caps the anti-diagonal, the best
        a height can do is make the grid square in count.
        """
        var wide = self.tile_columns()
        var want = max(TARGET_TILES // max(sweeps_in_level, 1), 1)
        var enough = min(want, wide)
        var fair = ceildiv(self.rows, enough)
        return min(max(fair, MIN_TILE_HEIGHT), TILE_SIDE)

    def tile_rows(self, sweeps_in_level: Int) -> Int:
        var height = self.tile_height(sweeps_in_level)
        return max(ceildiv(self.rows, height), 1)

    def tile_columns(self) -> Int:
        return max(ceildiv(self.columns, TILE_SIDE), 1)

    def tile_anti_diagonals(self, sweeps_in_level: Int) -> Int:
        return self.tile_rows(sweeps_in_level) + self.tile_columns() - 1


def device_local_extremum[
    half: SweepHalf
](
    ctx: DeviceContext,
    mut buffers: SweepBuffers,
    rows: Int,
    first_to: Int,
    second_to: Int,
    alphabet_size: Int,
    scoring: AffineGapCosts,
) raises -> Tuple[Int, Int, Int32]:
    """Finds where the best local alignment ends, tiled across the device rather than one block.

    Local alignment has unknown endpoints, so this runs twice: forwards it says where the best
    alignment ends, and backwards over those prefixes how far the same alignment reaches back.
    Ties resolve to the earliest cell in row-major order, matching the reference scan.
    """
    var sweeps = List[Sweep]()
    sweeps.append(Sweep(first_to, second_to, 0, rows, 0, 0, GapRun.OPENS, half))
    # Blocks that fall outside their sweep return without writing, and the buffers outlive the
    # scan, so anything left from an earlier one would be read as a candidate.
    ctx.enqueue_memset(buffers.block_best, Scalar[ScoreDType](0))
    ctx.enqueue_memset(buffers.block_place, Scalar[DType.int64](0))
    device_sweep_level[AlignmentMode.LOCAL](ctx, buffers, sweeps, alphabet_size, scoring)

    var best = Int32(0)
    var best_place = Int64(0)
    var slots = min(sweeps[0].tile_rows(len(sweeps)), sweeps[0].tile_columns())
    with buffers.block_best.map_to_host() as scores, buffers.block_place.map_to_host() as places:
        for slot in range(min(slots, buffers.block_slots)):
            var candidate = scores[slot]
            if candidate > best or (candidate == best and candidate != 0 and places[slot] < best_place):
                best = candidate
                best_place = places[slot]
    var stride = Int64(second_to + 1)
    return (Int(best_place // stride), Int(best_place % stride), best)


def device_sweep_level[
    mode: AlignmentMode
](
    ctx: DeviceContext,
    mut buffers: SweepBuffers,
    mut sweeps: List[Sweep],
    alphabet_size: Int,
    scoring: AffineGapCosts,
) raises:
    """Sweeps every independent sub-rectangle of one recursion level together.

    Depth-first recursion offers the parallelism of a single frame, which halves as the recursion
    deepens while its work halves too, so utilization falls as fast as the work does. Frames at a
    level partition both axes, so they share the frontier arrays without touching, and one flat
    block index carries both the sweep and its tile.
    """
    if len(sweeps) == 0:
        return

    var plan = List[SweepPlan]()
    var widest_tiles = 1
    var deepest = 0
    var corner_base = 0
    var top_base = 0
    var left_base = 0
    for index in range(len(sweeps)):
        var sweep = sweeps[index]
        var tile_rows = sweep.tile_rows(len(sweeps))
        var tile_columns = sweep.tile_columns()
        plan.append(
            SweepPlan(
                Int64(sweep.rows),
                Int64(sweep.columns),
                Int64(sweep.first_from),
                Int64(sweep.second_from),
                Int64(left_base),
                Int64(top_base),
                Int64(corner_base),
                Int64(tile_rows),
                Int64(tile_columns),
                Int64(sweep.tile_height(len(sweeps))),
                sweep.entering_run,
                sweep.half,
            )
        )
        corner_base += 3 * (tile_rows + 1)
        sweeps[index].row_base = left_base
        sweeps[index].column_base = top_base
        left_base += sweep.rows + 2
        top_base += sweep.columns + 2
        widest_tiles = max(widest_tiles, min(tile_rows, tile_columns))
        deepest = max(deepest, sweep.tile_anti_diagonals(len(sweeps)))

    if corner_base > buffers.corner_span or left_base > buffers.left_span or top_base > buffers.frontier_span:
        raise AffineGapsError(
            ErrorKind.SCRATCH_TOO_SMALL,
            String(
                "corner ",
                corner_base,
                "/",
                buffers.corner_span,
                " left ",
                left_base,
                "/",
                buffers.left_span,
                " top ",
                top_base,
                "/",
                buffers.frontier_span,
            ),
        )

    var words = List[Scalar[PlanDType]](length=len(plan) * PLAN_WORDS, fill=Scalar[PlanDType](0))
    """The plan crosses to the device as raw words, because a kernel takes a pointer, not a `List`."""
    var source = plan.unsafe_ptr().unsafe_bitcast[Scalar[PlanDType]]()
    for index in range(len(words)):
        words[index] = source[unsafe_offset=index]
    var plan_buffer = upload(ctx, Span(words))
    for tile_anti_diagonal in range(deepest):
        ctx.enqueue_function[tiled_sweep_kernel[mode]](
            buffers.sequences.unsafe_ptr(),
            buffers.substitutions.unsafe_ptr(),
            plan_buffer.unsafe_ptr(),
            buffers.block_best.unsafe_ptr(),
            buffers.block_place.unsafe_ptr(),
            buffers.top_scores.unsafe_ptr(),
            buffers.top_deletes.unsafe_ptr(),
            buffers.reverse_scores.unsafe_ptr(),
            buffers.reverse_deletes.unsafe_ptr(),
            buffers.left_scores.unsafe_ptr(),
            buffers.left_inserts.unsafe_ptr(),
            buffers.corner_scores.unsafe_ptr(),
            Int32(tile_anti_diagonal),
            Int32(widest_tiles),
            Int32(alphabet_size),
            scoring.open,
            scoring.extend,
            grid_dim=widest_tiles * len(sweeps),
            block_dim=STRIP_LANES,
        )
    ctx.synchronize()


def device_align[
    mode: AlignmentMode
](
    ctx: DeviceContext,
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: AffineGapCosts,
    alphabet: String,
    leaf_cells: Int,
    placement: Placement,
) raises -> AlignmentResult:
    """One pair aligned with every sweep on the device, in linear space.

    The device counterpart of `serial_align`, so a caller chooses where the work runs without
    also choosing how the traceback stores its state.
    """
    var rows = len(first)
    var columns = len(second)
    var path_columns = List[Int32](length=rows + 1, fill=Int32(0))
    var path_layers = List[Layer](length=rows + 1, fill=Layer.ALIGNING)

    comptime if mode == AlignmentMode.GLOBAL:
        device_hirschberg(
            ctx,
            first,
            second,
            0,
            rows,
            0,
            columns,
            substitutions,
            alphabet_size,
            scoring,
            leaf_cells,
            placement,
            path_columns,
            path_layers,
        )
        var reached = score_path(first, second, path_columns, path_layers, substitutions, alphabet_size, scoring, rows)
        var whole = expand_path(first, second, path_columns, path_layers, alphabet, mode, 0, rows)
        return AlignmentResult(reached, whole[0], whole[1])

    var sequences = List[Scalar[SymbolDType]](capacity=rows + columns)
    sequences.extend(first)
    sequences.extend(second)
    var buffers = device_sweep_buffers(ctx, rows, columns, Span(sequences), substitutions)
    var last_row, last_column, score = device_local_extremum[SweepHalf.FORWARD](
        ctx, buffers, rows, rows, columns, alphabet_size, scoring
    )

    var first_row = last_row
    """
    A non-positive best means the local alignment is empty, so the walk never runs and the two
    ends stay where the forward scan left them.
    """
    if score > 0:
        var back_rows, back_columns, _ = device_local_extremum[SweepHalf.REVERSE](
            ctx, buffers, rows, last_row, last_column, alphabet_size, scoring
        )
        first_row = last_row - back_rows
        var first_column = last_column - back_columns
        path_columns[last_row] = Int32(last_column)
        device_hirschberg(
            ctx,
            first,
            second,
            first_row,
            last_row,
            first_column,
            last_column,
            substitutions,
            alphabet_size,
            scoring,
            leaf_cells,
            placement,
            path_columns,
            path_layers,
        )

    var window = expand_path(first, second, path_columns, path_layers, alphabet, mode, first_row, last_row)
    return AlignmentResult(score, window[0], window[1])


# endregion GPU Wavefront


# region Presentation


def colorize(first_gapped: String, second_gapped: String) raises AffineGapsError -> Tuple[String, String]:
    """Green for a match, red for a mismatch, dim for a gap, mirroring `colorize_alignment`."""
    comptime green = "\x1b[32m"
    comptime red = "\x1b[31m"
    comptime white = "\x1b[37m"
    comptime reset = "\x1b[0m"
    var top = first_gapped.as_bytes()
    var bottom = second_gapped.as_bytes()
    if len(top) != len(bottom):
        raise AffineGapsError(ErrorKind.LENGTH_MISMATCH, "colorized alignment")

    comptime painted_column = len(green.as_bytes()) + len(reset.as_bytes()) + 1
    var painted_first = List[Byte](capacity=len(top) * painted_column)
    var painted_second = List[Byte](capacity=len(bottom) * painted_column)
    """
    Painted byte by byte into two buffers and turned into strings once, because a column is three
    escape sequences and a megabase alignment is millions of them.
    """
    for index in range(len(top)):
        var color = red
        if top[index] == bottom[index] and top[index] != GAP_BYTE:
            color = green
        elif top[index] == GAP_BYTE or bottom[index] == GAP_BYTE:
            color = white
        var opening = color.as_bytes()
        var closing = reset.as_bytes()
        painted_first.extend(opening)
        painted_first.append(top[index])
        painted_first.extend(closing)
        painted_second.extend(opening)
        painted_second.append(bottom[index])
        painted_second.extend(closing)
    return (String(unsafe_from_utf8=painted_first), String(unsafe_from_utf8=painted_second))


# endregion Presentation
