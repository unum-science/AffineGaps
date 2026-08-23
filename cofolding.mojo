"""
Sankoff simultaneous alignment and folding for CPU and GPU, exact and with traceback.

Sankoff's 1985 recurrence aligns two sequences and folds them together, crediting a base pair only
when both sequences can form it. The signal is covariation rather than thermodynamics, so the
scoring is a substitution matrix and a pair table, with no energy model anywhere.

The table is indexed by `(start, length)` on each sequence rather than by four endpoints. A
bifurcation then reads strictly smaller lengths in both dimensions, which makes every cell on the
anti-diagonal `length_first + length_second` independent and turns $O(n^6)$ work into `n + m + 1`
dependent layers with full parallelism inside each.

Memory is $O(n^2 m^2)$ and that is inherent, not an implementation limit: a bifurcation at one
layer reads every layer beneath it, so nothing can be retired and no Hirschberg-style band exists.
Measured at n = 24, a perfect freeing oracle still leaves 75.5% of the table live at peak. The
consolation is that traceback costs nothing extra, since the whole table is resident regardless.

Gaps are linear rather than affine. Affine needs the open state of both ends of both windows,
which multiplies a table that is already what bounds the sequence length.

The recurrence, the tie-breaking and the traceback are transcribed from `cofolding.py`, which
stays the parity oracle.
"""

from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.memory.pointer import AddressSpace

from max.gpu import barrier
from max.gpu.host import DeviceContext

from errors import AffineGapsError, ErrorKind
from common import (
    CLOSE_BYTE, DEFAULT_RNA_ALPHABET, GAP_BYTE, NEGATIVE_INFINITY, OPEN_BYTE, SCORE_DTYPE,
    SUBSTITUTION_DTYPE, SYMBOL_DTYPE, THREADS_PER_BLOCK, UNPAIRED_BYTE, translate, uniform_matrix,
    upload, zeroed,
)

# region Scoring

comptime DEFAULT_RNA_ALPHABET_SIZE = 4
comptime DEFAULT_MATCH = Int32(2)
comptime DEFAULT_MISMATCH = Int32(-1)
comptime DEFAULT_GAP = Int32(-2)


@fieldwise_init
struct SankoffScoring(ImplicitlyCopyable, TrivialRegisterPassable):
    """The linear gap cost. Substitution and pair tables travel separately, as spans."""

    var gap: Int32
    """Cost of leaving one position unaligned. Linear, not affine."""


@fieldwise_init
struct Neighbours(ImplicitlyCopyable, TrivialRegisterPassable):
    """The seven cells one Sankoff cell reads outside its bifurcation."""

    var head_aligned: Int32
    """Both windows give up their first symbol and the two are aligned."""
    var tail_aligned: Int32
    """Both windows give up their last symbol and the two are aligned."""
    var head_gap_in_second: Int32
    """The first window's head aligns to a gap."""
    var head_gap_in_first: Int32
    """The second window's head aligns to a gap."""
    var tail_gap_in_second: Int32
    """The first window's tail aligns to a gap."""
    var tail_gap_in_first: Int32
    """The second window's tail aligns to a gap."""
    var paired_inner: Int32
    """What remains once both windows close a base pair across their two ends."""


@always_inline
def read_neighbours(
    table: Pointer[Scalar[SCORE_DTYPE], _],
    start_first: Int,
    length_first: Int,
    start_second: Int,
    length_second: Int,
    rows: Int,
    columns: Int,
) -> Neighbours:
    """The seven cells one Sankoff cell reads, in the order the recurrence tries them.

    One reader serves the host sweep, the device sweep and the traceback, so a case can never be
    tested against a cell the fill did not use.
    """
    var paired_inner = Int32(0)
    if length_first >= 2 and length_second >= 2:
        paired_inner = table[
            unsafe_offset = cell_index(
                start_first + 1, length_first - 2, start_second + 1, length_second - 2, rows, columns
            )
        ]
    return Neighbours(
        table[
            unsafe_offset = cell_index(
                start_first + 1, length_first - 1, start_second + 1, length_second - 1, rows, columns
            )
        ],
        table[
            unsafe_offset = cell_index(
                start_first, length_first - 1, start_second, length_second - 1, rows, columns
            )
        ],
        table[
            unsafe_offset = cell_index(
                start_first + 1, length_first - 1, start_second, length_second, rows, columns
            )
        ],
        table[
            unsafe_offset = cell_index(
                start_first, length_first, start_second + 1, length_second - 1, rows, columns
            )
        ],
        table[
            unsafe_offset = cell_index(
                start_first, length_first - 1, start_second, length_second, rows, columns
            )
        ],
        table[
            unsafe_offset = cell_index(
                start_first, length_first, start_second, length_second - 1, rows, columns
            )
        ],
        paired_inner,
    )


@always_inline
def sankoff_cell(
    neighbours: Neighbours,
    head_substitution: Int32,
    tail_substitution: Int32,
    closing_first: Int32,
    closing_second: Int32,
    length_first: Int,
    length_second: Int,
    scoring: SankoffScoring,
) -> Int32:
    """The constant-work cases of one Sankoff cell, without the bifurcation.

    Pure arithmetic over values the caller already read, so the host sweep and the device sweep
    share this transcription unchanged. The bifurcation stays with the caller because its shape is
    a serial loop on one and a block reduction on the other.
    """
    var best = neighbours.head_aligned + head_substitution
    best = max(best, neighbours.tail_aligned + tail_substitution)
    best = max(best, neighbours.head_gap_in_second + scoring.gap)
    best = max(best, neighbours.head_gap_in_first + scoring.gap)
    best = max(best, neighbours.tail_gap_in_second + scoring.gap)
    best = max(best, neighbours.tail_gap_in_first + scoring.gap)
    if length_first >= 2 and length_second >= 2 and closing_first > 0 and closing_second > 0:
        best = max(
            best,
            neighbours.paired_inner
            + closing_first
            + closing_second
            + head_substitution
            + tail_substitution,
        )
    return best


@always_inline
def cell_index(
    start_first: Int, length_first: Int, start_second: Int, length_second: Int, rows: Int, columns: Int
) -> Int:
    """Row-major over `(start_first, length_first, start_second, length_second)`."""
    var stride = columns + 1
    return ((start_first * (rows + 1) + length_first) * stride + start_second) * stride + length_second


@always_inline
def table_cells(rows: Int, columns: Int) -> Int:
    return (rows + 1) * (rows + 1) * (columns + 1) * (columns + 1)

# endregion Scoring

# region Serial Reference


@always_inline
def cofold_cell(
    table: Pointer[Scalar[SCORE_DTYPE], _],
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    pairs: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: SankoffScoring,
    start_first: Int,
    length_first: Int,
    start_second: Int,
    length_second: Int,
) -> Int32:
    """Every case of one Sankoff cell, including the bifurcation scan.

    An empty window on either side can only be gapped through, which is what makes the base cases
    a pair of early returns rather than a branch wrapped around the whole body.
    """
    var rows = len(first)
    var columns = len(second)
    if length_first == 0:
        return Int32(length_second) * scoring.gap
    if length_second == 0:
        return Int32(length_first) * scoring.gap

    var head_first = Int(first[start_first])
    var head_second = Int(second[start_second])
    var tail_first = Int(first[start_first + length_first - 1])
    var tail_second = Int(second[start_second + length_second - 1])

    var closing_first = Int32(0)
    var closing_second = Int32(0)
    if length_first >= 2 and length_second >= 2:
        closing_first = Int32(pairs[head_first * alphabet_size + tail_first])
        closing_second = Int32(pairs[head_second * alphabet_size + tail_second])

    var neighbours = read_neighbours(
        table, start_first, length_first, start_second, length_second, rows, columns
    )
    var best = sankoff_cell(
        neighbours,
        Int32(substitutions[head_first * alphabet_size + head_second]),
        Int32(substitutions[tail_first * alphabet_size + tail_second]),
        closing_first,
        closing_second,
        length_first,
        length_second,
        scoring,
    )
    return max(best, bifurcation_best(table, start_first, length_first, start_second, length_second, rows, columns))


@always_inline
def bifurcation_best(
    table: Pointer[Scalar[SCORE_DTYPE], _],
    start_first: Int,
    length_first: Int,
    start_second: Int,
    length_second: Int,
    rows: Int,
    columns: Int,
) -> Int32:
    """The best split of both windows into two adjacent halves, or nothing when neither can split."""
    var best = NEGATIVE_INFINITY
    for cut_first in range(1, length_first):
        for cut_second in range(1, length_second):
            var left = table[
                unsafe_offset = cell_index(start_first, cut_first, start_second, cut_second, rows, columns)
            ]
            var right = table[
                unsafe_offset = cell_index(
                    start_first + cut_first,
                    length_first - cut_first,
                    start_second + cut_second,
                    length_second - cut_second,
                    rows,
                    columns,
                )
            ]
            best = max(best, left + right)
    return best


def serial_cofold_table(
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    pairs: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: SankoffScoring,
) raises -> List[Scalar[SCORE_DTYPE]]:
    """Fills the whole table on the host, in increasing order of each window's length.

    Kept as the definition the device sweep is differentially tested against, so it favours
    following `cofolding.py` line for line over being fast.
    """
    var rows = len(first)
    var columns = len(second)
    var table = List[Scalar[SCORE_DTYPE]](length=table_cells(rows, columns), fill=Scalar[SCORE_DTYPE](0))
    var cells = table.unsafe_ptr()

    for length_first in range(rows + 1):
        for length_second in range(columns + 1):
            if length_first == 0 and length_second == 0:
                continue
            for start_first in range(rows - length_first + 1):
                for start_second in range(columns - length_second + 1):
                    var here = cell_index(
                        start_first, length_first, start_second, length_second, rows, columns
                    )
                    cells[unsafe_offset=here] = cofold_cell(
                        cells, first, second, substitutions, pairs, alphabet_size, scoring,
                        start_first, length_first, start_second, length_second,
                    )
    return table^

# endregion Serial Reference

# region GPU Sweep


@always_inline
def block_max(
    reduction: Pointer[Scalar[SCORE_DTYPE], MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    best: Int32,
) -> Int32:
    """Block-wide maximum. No place is carried, because a Sankoff cell keeps only its score.

    Every thread holds the answer afterwards, which saves the caller a second barrier.
    """
    reduction[unsafe_offset = Int(thread_idx.x)] = best
    barrier()
    var span = THREADS_PER_BLOCK // 2
    while span > 0:
        if Int(thread_idx.x) < span:
            reduction[unsafe_offset = Int(thread_idx.x)] = max(
                reduction[unsafe_offset = Int(thread_idx.x)],
                reduction[unsafe_offset = Int(thread_idx.x) + span],
            )
        barrier()
        span //= 2
    return reduction[unsafe_offset=0]


def cofold_layer_kernel(
    first: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    second: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    substitutions: Pointer[Scalar[SUBSTITUTION_DTYPE], MutAnyOrigin],
    pairs: Pointer[Scalar[SUBSTITUTION_DTYPE], MutAnyOrigin],
    table: Pointer[Scalar[SCORE_DTYPE], MutAnyOrigin],
    rows_in: Int32,
    columns_in: Int32,
    alphabet_size: Int32,
    diagonal: Int32,
    smallest_length_first: Int32,
    gap: Int32,
):
    """One block per cell of the anti-diagonal `length_first + length_second`.

    Cells on a diagonal read only strictly smaller lengths, so the whole layer is independent and
    one kernel launch per diagonal is all the synchronization the recurrence needs.
    """
    var rows = Int(rows_in)
    var columns = Int(columns_in)
    var width = Int(alphabet_size)
    var length_first = Int(smallest_length_first) + Int(block_idx.x)
    var length_second = Int(diagonal) - length_first
    var start_first = Int(block_idx.y)
    var start_second = Int(block_idx.z)

    if length_second < 0 or length_first > rows or length_second > columns:
        return
    if start_first + length_first > rows or start_second + length_second > columns:
        return

    var reduction = stack_allocation[
        THREADS_PER_BLOCK, Scalar[SCORE_DTYPE], address_space = AddressSpace.SHARED
    ]()
    var here = cell_index(start_first, length_first, start_second, length_second, rows, columns)
    var scoring = SankoffScoring(gap)

    if length_first == 0 and length_second == 0:
        return
    if length_first == 0:
        if thread_idx.x == 0:
            table[unsafe_offset=here] = Int32(length_second) * gap
        return
    if length_second == 0:
        if thread_idx.x == 0:
            table[unsafe_offset=here] = Int32(length_first) * gap
        return

    var best = NEGATIVE_INFINITY
    """
    Thread zero owns the constant cases, so only it pays for their seven reads. Every other thread goes straight to
    its slice of the bifurcation.
    """
    if thread_idx.x == 0:
        var head_first = Int(first[unsafe_offset=start_first])
        var head_second = Int(second[unsafe_offset=start_second])
        var tail_first = Int(first[unsafe_offset = start_first + length_first - 1])
        var tail_second = Int(second[unsafe_offset = start_second + length_second - 1])

        var closing_first = Int32(0)
        var closing_second = Int32(0)
        if length_first >= 2 and length_second >= 2:
            closing_first = Int32(pairs[unsafe_offset = head_first * width + tail_first])
            closing_second = Int32(pairs[unsafe_offset = head_second * width + tail_second])
        var neighbours = read_neighbours(
            table, start_first, length_first, start_second, length_second, rows, columns
        )
        best = sankoff_cell(
            neighbours,
            Int32(substitutions[unsafe_offset = head_first * width + head_second]),
            Int32(substitutions[unsafe_offset = tail_first * width + tail_second]),
            closing_first,
            closing_second,
            length_first,
            length_second,
            scoring,
        )

    if length_first >= 2 and length_second >= 2:
        var splits_second = length_second - 1
        var split_count = (length_first - 1) * splits_second
        for flat in range(Int(thread_idx.x), split_count, THREADS_PER_BLOCK):
            var cut_first = flat // splits_second + 1
            var cut_second = flat % splits_second + 1
            var left = table[
                unsafe_offset = cell_index(
                    start_first, cut_first, start_second, cut_second, rows, columns
                )
            ]
            var right = table[
                unsafe_offset = cell_index(
                    start_first + cut_first,
                    length_first - cut_first,
                    start_second + cut_second,
                    length_second - cut_second,
                    rows,
                    columns,
                )
            ]
            best = max(best, left + right)

    var answer = block_max(reduction, best)
    if thread_idx.x == 0:
        table[unsafe_offset=here] = answer


def device_cofold_table(
    ctx: DeviceContext,
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    pairs: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: SankoffScoring,
) raises -> List[Scalar[SCORE_DTYPE]]:
    """Sweeps the table on the device, one launch per anti-diagonal of lengths."""
    var rows = len(first)
    var columns = len(second)
    var cells = table_cells(rows, columns)

    var first_buffer = upload[SYMBOL_DTYPE](ctx, first)
    var second_buffer = upload[SYMBOL_DTYPE](ctx, second)
    var substitutions_buffer = upload[SUBSTITUTION_DTYPE](ctx, substitutions)
    var pairs_buffer = upload[SUBSTITUTION_DTYPE](ctx, pairs)
    var table_buffer = zeroed[SCORE_DTYPE](ctx, cells)

    for diagonal in range(0, rows + columns + 1):
        var smallest = max(0, diagonal - columns)
        var largest = min(rows, diagonal)
        if smallest > largest:
            continue
        ctx.enqueue_function[cofold_layer_kernel](
            first_buffer.unsafe_ptr(),
            second_buffer.unsafe_ptr(),
            substitutions_buffer.unsafe_ptr(),
            pairs_buffer.unsafe_ptr(),
            table_buffer.unsafe_ptr(),
            Int32(rows),
            Int32(columns),
            Int32(alphabet_size),
            Int32(diagonal),
            Int32(smallest),
            scoring.gap,
            grid_dim=(largest - smallest + 1, rows + 1, columns + 1),
            block_dim=THREADS_PER_BLOCK,
        )
    ctx.synchronize()

    var table = List[Scalar[SCORE_DTYPE]](length=cells, fill=Scalar[SCORE_DTYPE](0))
    """
    The whole table comes back because the traceback walks arbitrary cells of it. Copying element by element costs
    more than the sweep at any interesting size, so this is one move.
    """
    ctx.enqueue_copy(table.unsafe_ptr(), table_buffer)
    ctx.synchronize()
    return table^

# endregion GPU Sweep

# region Traceback

@fieldwise_init
struct SankoffCase(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which decomposition produced a cell's score, in the order the recurrence tries them."""

    var identifier: UInt8
    """Which decomposition this names."""
    comptime HEAD_ALIGNED = Self(0)
    """Both heads consumed and aligned to each other."""
    comptime TAIL_ALIGNED = Self(1)
    """Both tails consumed and aligned to each other."""
    comptime HEAD_GAP_IN_SECOND = Self(2)
    """The first head consumed against a gap."""
    comptime HEAD_GAP_IN_FIRST = Self(3)
    """The second head consumed against a gap."""
    comptime TAIL_GAP_IN_SECOND = Self(4)
    """The first tail consumed against a gap."""
    comptime TAIL_GAP_IN_FIRST = Self(5)
    """The second tail consumed against a gap."""
    comptime PAIRED = Self(6)
    """Both windows close a base pair, which is where covariation is credited."""
    comptime BIFURCATION = Self(7)
    """Both windows split into two adjacent halves."""


@fieldwise_init
struct Decision(ImplicitlyCopyable, TrivialRegisterPassable):
    """A winning case, with the split point when it bifurcated."""

    var outcome: SankoffCase
    """Which case reproduced the cell's stored score."""
    var cut_first: Int32
    """Where the first window split, or zero when the case did not bifurcate."""
    var cut_second: Int32
    """Where the second window split, or zero when the case did not bifurcate."""


def winning_case(
    table: ImmSpan[Scalar[SCORE_DTYPE], _],
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    pairs: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    scoring: SankoffScoring,
    start_first: Int,
    length_first: Int,
    start_second: Int,
    length_second: Int,
) raises AffineGapsError -> Decision:
    """Which case reproduces a cell's stored score, tried in the order the sweep tried them.

    Re-derived rather than recorded: the table is resident anyway, so a parallel array of
    decisions would cost as much again for nothing.
    """
    var rows = len(first)
    var columns = len(second)
    var stored = table[cell_index(start_first, length_first, start_second, length_second, rows, columns)]
    var head_first = Int(first[start_first])
    var head_second = Int(second[start_second])
    var tail_first = Int(first[start_first + length_first - 1])
    var tail_second = Int(second[start_second + length_second - 1])
    var head_substitution = Int32(substitutions[head_first * alphabet_size + head_second])
    var tail_substitution = Int32(substitutions[tail_first * alphabet_size + tail_second])

    var near = read_neighbours(
        table.unsafe_ptr(), start_first, length_first, start_second, length_second, rows, columns
    )
    if near.head_aligned + head_substitution == stored:
        return Decision(SankoffCase.HEAD_ALIGNED, 0, 0)
    if near.tail_aligned + tail_substitution == stored:
        return Decision(SankoffCase.TAIL_ALIGNED, 0, 0)
    if near.head_gap_in_second + scoring.gap == stored:
        return Decision(SankoffCase.HEAD_GAP_IN_SECOND, 0, 0)
    if near.head_gap_in_first + scoring.gap == stored:
        return Decision(SankoffCase.HEAD_GAP_IN_FIRST, 0, 0)
    if near.tail_gap_in_second + scoring.gap == stored:
        return Decision(SankoffCase.TAIL_GAP_IN_SECOND, 0, 0)
    if near.tail_gap_in_first + scoring.gap == stored:
        return Decision(SankoffCase.TAIL_GAP_IN_FIRST, 0, 0)

    if length_first >= 2 and length_second >= 2:
        var closing_first = Int32(pairs[head_first * alphabet_size + tail_first])
        var closing_second = Int32(pairs[head_second * alphabet_size + tail_second])
        if closing_first > 0 and closing_second > 0:
            if (
                near.paired_inner + closing_first + closing_second + head_substitution + tail_substitution
                == stored
            ):
                return Decision(SankoffCase.PAIRED, 0, 0)
        for cut_first in range(1, length_first):
            for cut_second in range(1, length_second):
                var left = table[
                    cell_index(start_first, cut_first, start_second, cut_second, rows, columns)
                ]
                var right = table[
                    cell_index(
                        start_first + cut_first,
                        length_first - cut_first,
                        start_second + cut_second,
                        length_second - cut_second,
                        rows,
                        columns,
                    )
                ]
                if left + right == stored:
                    return Decision(SankoffCase.BIFURCATION, Int32(cut_first), Int32(cut_second))
    raise AffineGapsError(ErrorKind.INCONSISTENT_TABLE, "Sankoff traceback")


def expand_window(
    table: ImmSpan[Scalar[SCORE_DTYPE], _],
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    pairs: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    letters: ImmSpan[Byte, _],
    alphabet_size: Int,
    scoring: SankoffScoring,
    start_first: Int,
    length_first: Int,
    start_second: Int,
    length_second: Int,
    mut gapped_first: List[Byte],
    mut gapped_second: List[Byte],
    mut structure: List[Byte],
) raises:
    """Appends the columns one window contributes, left to right.

    Head cases append before recursing and tail cases after, which is what keeps a bifurcation's
    two halves in order without a separate ordering pass.
    """
    if length_first == 0 and length_second == 0:
        return
    if length_first == 0:
        for offset in range(length_second):
            gapped_first.append(GAP_BYTE)
            gapped_second.append(letters[Int(second[start_second + offset])])
            structure.append(UNPAIRED_BYTE)
        return
    if length_second == 0:
        for offset in range(length_first):
            gapped_first.append(letters[Int(first[start_first + offset])])
            gapped_second.append(GAP_BYTE)
            structure.append(UNPAIRED_BYTE)
        return

    var decision = winning_case(
        table, first, second, substitutions, pairs, alphabet_size, scoring,
        start_first, length_first, start_second, length_second,
    )
    var head_first = letters[Int(first[start_first])]
    var head_second = letters[Int(second[start_second])]
    var tail_first = letters[Int(first[start_first + length_first - 1])]
    var tail_second = letters[Int(second[start_second + length_second - 1])]

    if decision.outcome == SankoffCase.HEAD_ALIGNED:
        gapped_first.append(head_first)
        gapped_second.append(head_second)
        structure.append(UNPAIRED_BYTE)
        expand_window(
            table, first, second, substitutions, pairs, letters, alphabet_size, scoring,
            start_first + 1, length_first - 1, start_second + 1, length_second - 1,
            gapped_first, gapped_second, structure,
        )
    elif decision.outcome == SankoffCase.HEAD_GAP_IN_SECOND:
        gapped_first.append(head_first)
        gapped_second.append(GAP_BYTE)
        structure.append(UNPAIRED_BYTE)
        expand_window(
            table, first, second, substitutions, pairs, letters, alphabet_size, scoring,
            start_first + 1, length_first - 1, start_second, length_second,
            gapped_first, gapped_second, structure,
        )
    elif decision.outcome == SankoffCase.HEAD_GAP_IN_FIRST:
        gapped_first.append(GAP_BYTE)
        gapped_second.append(head_second)
        structure.append(UNPAIRED_BYTE)
        expand_window(
            table, first, second, substitutions, pairs, letters, alphabet_size, scoring,
            start_first, length_first, start_second + 1, length_second - 1,
            gapped_first, gapped_second, structure,
        )
    elif decision.outcome == SankoffCase.PAIRED:
        gapped_first.append(head_first)
        gapped_second.append(head_second)
        structure.append(OPEN_BYTE)
        expand_window(
            table, first, second, substitutions, pairs, letters, alphabet_size, scoring,
            start_first + 1, length_first - 2, start_second + 1, length_second - 2,
            gapped_first, gapped_second, structure,
        )
        gapped_first.append(tail_first)
        gapped_second.append(tail_second)
        structure.append(CLOSE_BYTE)
    elif decision.outcome == SankoffCase.BIFURCATION:
        var cut_first = Int(decision.cut_first)
        var cut_second = Int(decision.cut_second)
        expand_window(
            table, first, second, substitutions, pairs, letters, alphabet_size, scoring,
            start_first, cut_first, start_second, cut_second,
            gapped_first, gapped_second, structure,
        )
        expand_window(
            table, first, second, substitutions, pairs, letters, alphabet_size, scoring,
            start_first + cut_first, length_first - cut_first,
            start_second + cut_second, length_second - cut_second,
            gapped_first, gapped_second, structure,
        )
    elif decision.outcome == SankoffCase.TAIL_ALIGNED:
        expand_window(
            table, first, second, substitutions, pairs, letters, alphabet_size, scoring,
            start_first, length_first - 1, start_second, length_second - 1,
            gapped_first, gapped_second, structure,
        )
        gapped_first.append(tail_first)
        gapped_second.append(tail_second)
        structure.append(UNPAIRED_BYTE)
    elif decision.outcome == SankoffCase.TAIL_GAP_IN_SECOND:
        expand_window(
            table, first, second, substitutions, pairs, letters, alphabet_size, scoring,
            start_first, length_first - 1, start_second, length_second,
            gapped_first, gapped_second, structure,
        )
        gapped_first.append(tail_first)
        gapped_second.append(GAP_BYTE)
        structure.append(UNPAIRED_BYTE)
    else:
        expand_window(
            table, first, second, substitutions, pairs, letters, alphabet_size, scoring,
            start_first, length_first, start_second, length_second - 1,
            gapped_first, gapped_second, structure,
        )
        gapped_first.append(GAP_BYTE)
        gapped_second.append(tail_second)
        structure.append(UNPAIRED_BYTE)

# endregion Traceback

# region Interface


def default_rna_pair_matrix() -> List[Scalar[SUBSTITUTION_DTYPE]]:
    """The six chemically possible pairings over `DEFAULT_RNA_ALPHABET`, strongest first.

    Watson-Crick pairs score above the wobble pair; everything else is zero and cannot close.
    These rank pairings, they are not measured energies.
    """
    var pairs = List[Scalar[SUBSTITUTION_DTYPE]](
        length=DEFAULT_RNA_ALPHABET_SIZE * DEFAULT_RNA_ALPHABET_SIZE, fill=Scalar[SUBSTITUTION_DTYPE](0)
    )
    var adenine = 0
    var cytosine = 1
    var guanine = 2
    var uracil = 3
    pairs[cytosine * DEFAULT_RNA_ALPHABET_SIZE + guanine] = 3
    pairs[guanine * DEFAULT_RNA_ALPHABET_SIZE + cytosine] = 3
    pairs[adenine * DEFAULT_RNA_ALPHABET_SIZE + uracil] = 2
    pairs[uracil * DEFAULT_RNA_ALPHABET_SIZE + adenine] = 2
    pairs[guanine * DEFAULT_RNA_ALPHABET_SIZE + uracil] = 1
    pairs[uracil * DEFAULT_RNA_ALPHABET_SIZE + guanine] = 1
    return pairs^


@fieldwise_init
struct CofoldResult(Copyable, Movable):
    """Both gapped sequences, the structure they agree on, and the score of that decomposition."""

    var gapped_first: String
    """The first sequence with gaps inserted, one character per alignment column."""
    var gapped_second: String
    """The second sequence, gapped to the same columns."""
    var structure: String
    """Dot-bracket over those columns, so one string serves both sequences."""
    var score: Int32
    """The optimum of the recurrence, which is not a free energy."""


def cofold_from_table(
    table: ImmSpan[Scalar[SCORE_DTYPE], _],
    first: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    second: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    substitutions: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    pairs: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet: String,
    scoring: SankoffScoring,
) raises -> CofoldResult:
    """Walks a filled table into the two gapped strings and their consensus structure."""
    var rows = len(first)
    var columns = len(second)
    var letters = alphabet.as_bytes()
    var gapped_first = List[Byte]()
    var gapped_second = List[Byte]()
    var structure = List[Byte]()
    expand_window(
        table, first, second, substitutions, pairs, letters, alphabet.byte_length(), scoring,
        0, rows, 0, columns, gapped_first, gapped_second, structure,
    )
    return CofoldResult(
        String(unsafe_from_utf8=gapped_first),
        String(unsafe_from_utf8=gapped_second),
        String(unsafe_from_utf8=structure),
        table[cell_index(0, rows, 0, columns, rows, columns)],
    )


def serial_cofold(
    first_text: String,
    second_text: String,
    alphabet: String,
    scoring: SankoffScoring,
    match_score: Int,
    mismatch_score: Int,
) raises -> CofoldResult:
    """Aligns and folds two sequences on the host."""
    var first = translate(first_text, alphabet)
    var second = translate(second_text, alphabet)
    var substitutions = uniform_matrix(alphabet.byte_length(), match_score, mismatch_score)
    var pairs = default_rna_pair_matrix()
    var table = serial_cofold_table(
        Span(first), Span(second), Span(substitutions), Span(pairs), alphabet.byte_length(), scoring
    )
    return cofold_from_table(
        Span(table), Span(first), Span(second), Span(substitutions), Span(pairs), alphabet, scoring
    )


def device_cofold(
    ctx: DeviceContext,
    first_text: String,
    second_text: String,
    alphabet: String,
    scoring: SankoffScoring,
    match_score: Int,
    mismatch_score: Int,
) raises -> CofoldResult:
    """Aligns and folds two sequences with the sweep on the device and the walk on the host.

    Traceback stays on the host because it is a serial walk over a table the device already
    filled, and copying the table back costs less than a kernel that cannot use its threads.
    """
    var first = translate(first_text, alphabet)
    var second = translate(second_text, alphabet)
    var substitutions = uniform_matrix(alphabet.byte_length(), match_score, mismatch_score)
    var pairs = default_rna_pair_matrix()
    var table = device_cofold_table(
        ctx, Span(first), Span(second), Span(substitutions), Span(pairs), alphabet.byte_length(), scoring
    )
    return cofold_from_table(
        Span(table), Span(first), Span(second), Span(substitutions), Span(pairs), alphabet, scoring
    )

# endregion Interface
