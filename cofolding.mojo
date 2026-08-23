"""
Sankoff simultaneous alignment and folding for CPU and GPU, exact and with traceback.

Sankoff's 1985 recurrence aligns two sequences and folds them together, crediting a base pair only
when both sequences can form it. The signal is covariation rather than thermodynamics, so the
scoring is a substitution matrix and a pair table, with no energy model anywhere.

The table is indexed by `(start, length)` on each sequence rather than by four endpoints. Each
cell decides only what its two heads do, so the recurrence reads strictly smaller lengths in both
dimensions, every cell on the anti-diagonal `length_first + length_second` is independent, and
$O(n^6)$ work becomes `n + m + 1` dependent layers with full parallelism inside each.

The heads are gapped, aligned and unpaired, or aligned and paired with a later column, and that
last case is the whole scan: it names the partner of each head and covers both a helix closing the
window and a helix followed by more structure. Restricting the scan to partners the pair table
actually allows is what keeps it affordable, since only six of the sixteen letter pairs can close.
Partners are tried from the furthest back, so a tie resolves to the longest helix and a stem comes
out nested rather than chopped into neighbours.

Memory is $O(n^2 m^2)$ and that is inherent, not an implementation limit: a helix at one layer
reads every layer beneath it, so nothing can be retired and no Hirschberg-style band exists.
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
    CLOSE_BYTE,
    DEFAULT_RNA_ALPHABET,
    GAP_BYTE,
    NEGATIVE_INFINITY,
    OPEN_BYTE,
    SCORE_DTYPE,
    SUBSTITUTION_DTYPE,
    SYMBOL_DTYPE,
    THREADS_PER_BLOCK,
    UNPAIRED_BYTE,
    translate,
    uniform_matrix,
    upload,
    zeroed,
)

# region Scoring

comptime DEFAULT_RNA_ALPHABET_SIZE = 4
comptime DEFAULT_MATCH = Int32(2)
comptime DEFAULT_MISMATCH = Int32(-1)
comptime DEFAULT_GAP = Int32(-2)
comptime POSITION_DTYPE = DType.int32


@fieldwise_init
struct SankoffScoring(ImplicitlyCopyable, TrivialRegisterPassable):
    """The linear gap cost. Substitution and pair tables travel separately, as spans."""

    var gap: Int32
    """Cost of leaving one position unaligned. Linear, not affine."""


@fieldwise_init
struct Neighbours(ImplicitlyCopyable, TrivialRegisterPassable):
    """The three cells one Sankoff cell reads outside its helix scan."""

    var aligned: Int32
    """Both windows give up their first symbol and the two are aligned."""
    var gap_in_second: Int32
    """The first window's head aligns to a gap."""
    var gap_in_first: Int32
    """The second window's head aligns to a gap."""


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
    """The three cells one Sankoff cell reads, in the order the recurrence tries them.

    One reader serves the host sweep, the device sweep and the traceback, so a case can never be
    tested against a cell the fill did not use.
    """
    return Neighbours(
        table[
            unsafe_offset=cell_index(
                start_first + 1, length_first - 1, start_second + 1, length_second - 1, rows, columns
            )
        ],
        table[unsafe_offset=cell_index(start_first + 1, length_first - 1, start_second, length_second, rows, columns)],
        table[unsafe_offset=cell_index(start_first, length_first, start_second + 1, length_second - 1, rows, columns)],
    )


@always_inline
def sankoff_cell(neighbours: Neighbours, head_substitution: Int32, scoring: SankoffScoring) -> Int32:
    """The constant-work cases of one Sankoff cell, without the helix scan.

    Pure arithmetic over values the caller already read, so the host sweep and the device sweep
    share this transcription unchanged. The helix scan stays with the caller because its shape is
    a serial loop on one and a block reduction on the other.
    """
    var best = neighbours.aligned + head_substitution
    best = max(best, neighbours.gap_in_second + scoring.gap)
    best = max(best, neighbours.gap_in_first + scoring.gap)
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

# region Partners


@fieldwise_init
struct PartnerIndex(Copyable, Movable):
    """Where each letter's possible partners sit in one sequence, as one ascending run per letter."""

    var positions: List[Scalar[POSITION_DTYPE]]
    """Every position that can close a pair, the letters' runs laid end to end."""
    var bounds: List[Scalar[POSITION_DTYPE]]
    """Letter `c`'s run at its first entry from `t` onwards, held at `c * (length + 1) + t`."""


@fieldwise_init
struct PartnerRange(ImplicitlyCopyable, TrivialRegisterPassable):
    """Half-open slice of one letter's run, covering the partners one window can reach."""

    var low: Int
    """First entry of the run that lands inside the window."""
    var high: Int
    """One past the run's last entry inside the window."""


def partner_index(
    sequence: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    pairs: ImmSpan[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
) -> PartnerIndex:
    """Lists, per letter, every position in the sequence that letter can close a pair with.

    The runs are ascending, so a window's candidates are a contiguous slice and the scan carries
    no test inside it. Six of the sixteen RNA letter pairs can close, which is what makes the
    slice worth building rather than filtering the window on the fly.
    """
    var length = len(sequence)
    var positions = List[Scalar[POSITION_DTYPE]]()
    var bounds = List[Scalar[POSITION_DTYPE]](length=alphabet_size * (length + 1), fill=Scalar[POSITION_DTYPE](0))
    for letter in range(alphabet_size):
        for position in range(length):
            bounds[letter * (length + 1) + position] = Scalar[POSITION_DTYPE](len(positions))
            if pairs[letter * alphabet_size + Int(sequence[position])] > 0:
                positions.append(Scalar[POSITION_DTYPE](position))
        bounds[letter * (length + 1) + length] = Scalar[POSITION_DTYPE](len(positions))
    return PartnerIndex(positions^, bounds^)


@always_inline
def partner_range(
    bounds: Pointer[Scalar[POSITION_DTYPE], _],
    head: Int,
    sequence_length: Int,
    start: Int,
    window: Int,
) -> PartnerRange:
    """Which of `head`'s partners a window can reach, as a slice of that letter's run.

    A head cannot pair with itself, so the slice runs from the position after it to the window's
    last, and a window of one position yields an empty slice rather than a case.
    """
    var run = head * (sequence_length + 1)
    return PartnerRange(
        Int(bounds[unsafe_offset=run + start + 1]),
        Int(bounds[unsafe_offset=run + start + window]),
    )


@always_inline
def helix_candidate(
    table: Pointer[Scalar[SCORE_DTYPE], _],
    first: Pointer[Scalar[SYMBOL_DTYPE], _],
    second: Pointer[Scalar[SYMBOL_DTYPE], _],
    substitutions: Pointer[Scalar[SUBSTITUTION_DTYPE], _],
    pairs: Pointer[Scalar[SUBSTITUTION_DTYPE], _],
    alphabet_size: Int,
    start_first: Int,
    length_first: Int,
    start_second: Int,
    length_second: Int,
    span_first: Int,
    span_second: Int,
    rows: Int,
    columns: Int,
) -> Int32:
    """What one helix scores: its two closing columns, what they enclose, and what follows it.

    One transcription serves the host sweep, the device sweep and the traceback, so a helix can
    never be tested against a score the fill did not use.
    """
    var head_first = Int(first[unsafe_offset=start_first])
    var head_second = Int(second[unsafe_offset=start_second])
    var partner_first = Int(first[unsafe_offset=start_first + span_first])
    var partner_second = Int(second[unsafe_offset=start_second + span_second])
    var inside = table[
        unsafe_offset=cell_index(start_first + 1, span_first - 1, start_second + 1, span_second - 1, rows, columns)
    ]
    var after = table[
        unsafe_offset=cell_index(
            start_first + span_first + 1,
            length_first - span_first - 1,
            start_second + span_second + 1,
            length_second - span_second - 1,
            rows,
            columns,
        )
    ]
    return (
        inside
        + after
        + Int32(pairs[unsafe_offset=head_first * alphabet_size + partner_first])
        + Int32(pairs[unsafe_offset=head_second * alphabet_size + partner_second])
        + Int32(substitutions[unsafe_offset=head_first * alphabet_size + head_second])
        + Int32(substitutions[unsafe_offset=partner_first * alphabet_size + partner_second])
    )


# endregion Partners

# region Serial Reference


@always_inline
def helix_best(
    table: Pointer[Scalar[SCORE_DTYPE], _],
    first: Pointer[Scalar[SYMBOL_DTYPE], _],
    second: Pointer[Scalar[SYMBOL_DTYPE], _],
    substitutions: Pointer[Scalar[SUBSTITUTION_DTYPE], _],
    pairs: Pointer[Scalar[SUBSTITUTION_DTYPE], _],
    positions_first: Pointer[Scalar[POSITION_DTYPE], _],
    bounds_first: Pointer[Scalar[POSITION_DTYPE], _],
    positions_second: Pointer[Scalar[POSITION_DTYPE], _],
    bounds_second: Pointer[Scalar[POSITION_DTYPE], _],
    alphabet_size: Int,
    start_first: Int,
    length_first: Int,
    start_second: Int,
    length_second: Int,
    rows: Int,
    columns: Int,
) -> Int32:
    """The best helix the two heads can open, over every partner both windows can reach."""
    var best = NEGATIVE_INFINITY
    var head_first = Int(first[unsafe_offset=start_first])
    var head_second = Int(second[unsafe_offset=start_second])
    var reachable_first = partner_range(bounds_first, head_first, rows, start_first, length_first)
    var reachable_second = partner_range(bounds_second, head_second, columns, start_second, length_second)
    for index_first in range(reachable_first.high - 1, reachable_first.low - 1, -1):
        var partner_first = Int(positions_first[unsafe_offset=index_first])
        for index_second in range(reachable_second.high - 1, reachable_second.low - 1, -1):
            best = max(
                best,
                helix_candidate(
                    table,
                    first,
                    second,
                    substitutions,
                    pairs,
                    alphabet_size,
                    start_first,
                    length_first,
                    start_second,
                    length_second,
                    partner_first - start_first,
                    Int(positions_second[unsafe_offset=index_second]) - start_second,
                    rows,
                    columns,
                ),
            )
    return best


@always_inline
def cofold_cell(
    table: Pointer[Scalar[SCORE_DTYPE], _],
    first: Pointer[Scalar[SYMBOL_DTYPE], _],
    second: Pointer[Scalar[SYMBOL_DTYPE], _],
    substitutions: Pointer[Scalar[SUBSTITUTION_DTYPE], _],
    pairs: Pointer[Scalar[SUBSTITUTION_DTYPE], _],
    positions_first: Pointer[Scalar[POSITION_DTYPE], _],
    bounds_first: Pointer[Scalar[POSITION_DTYPE], _],
    positions_second: Pointer[Scalar[POSITION_DTYPE], _],
    bounds_second: Pointer[Scalar[POSITION_DTYPE], _],
    alphabet_size: Int,
    scoring: SankoffScoring,
    start_first: Int,
    length_first: Int,
    start_second: Int,
    length_second: Int,
    rows: Int,
    columns: Int,
) -> Int32:
    """Every case of one Sankoff cell, including the helix scan.

    An empty window on either side can only be gapped through, which is what makes the base cases
    a pair of early returns rather than a branch wrapped around the whole body.
    """
    if length_first == 0:
        return Int32(length_second) * scoring.gap
    if length_second == 0:
        return Int32(length_first) * scoring.gap

    var head_first = Int(first[unsafe_offset=start_first])
    var head_second = Int(second[unsafe_offset=start_second])
    var neighbours = read_neighbours(table, start_first, length_first, start_second, length_second, rows, columns)
    var best = sankoff_cell(
        neighbours, Int32(substitutions[unsafe_offset=head_first * alphabet_size + head_second]), scoring
    )
    return max(
        best,
        helix_best(
            table,
            first,
            second,
            substitutions,
            pairs,
            positions_first,
            bounds_first,
            positions_second,
            bounds_second,
            alphabet_size,
            start_first,
            length_first,
            start_second,
            length_second,
            rows,
            columns,
        ),
    )


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
    var partners_first = partner_index(first, pairs, alphabet_size)
    var partners_second = partner_index(second, pairs, alphabet_size)
    var table = List[Scalar[SCORE_DTYPE]](length=table_cells(rows, columns), fill=Scalar[SCORE_DTYPE](0))
    var cells = table.unsafe_ptr()
    var first_cursor = first.unsafe_ptr()
    var second_cursor = second.unsafe_ptr()
    var substitutions_cursor = substitutions.unsafe_ptr()
    var pairs_cursor = pairs.unsafe_ptr()
    var positions_first = partners_first.positions.unsafe_ptr()
    var bounds_first = partners_first.bounds.unsafe_ptr()
    var positions_second = partners_second.positions.unsafe_ptr()
    var bounds_second = partners_second.bounds.unsafe_ptr()

    for length_first in range(rows + 1):
        for length_second in range(columns + 1):
            if length_first == 0 and length_second == 0:
                continue
            for start_first in range(rows - length_first + 1):
                for start_second in range(columns - length_second + 1):
                    var here = cell_index(start_first, length_first, start_second, length_second, rows, columns)
                    cells[unsafe_offset=here] = cofold_cell(
                        cells,
                        first_cursor,
                        second_cursor,
                        substitutions_cursor,
                        pairs_cursor,
                        positions_first,
                        bounds_first,
                        positions_second,
                        bounds_second,
                        alphabet_size,
                        scoring,
                        start_first,
                        length_first,
                        start_second,
                        length_second,
                        rows,
                        columns,
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
    reduction[unsafe_offset=Int(thread_idx.x)] = best
    barrier()
    var span = THREADS_PER_BLOCK // 2
    while span > 0:
        if Int(thread_idx.x) < span:
            reduction[unsafe_offset=Int(thread_idx.x)] = max(
                reduction[unsafe_offset=Int(thread_idx.x)],
                reduction[unsafe_offset=Int(thread_idx.x) + span],
            )
        barrier()
        span //= 2
    return reduction[unsafe_offset=0]


def cofold_layer_kernel(
    first: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    second: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    substitutions: Pointer[Scalar[SUBSTITUTION_DTYPE], MutAnyOrigin],
    pairs: Pointer[Scalar[SUBSTITUTION_DTYPE], MutAnyOrigin],
    positions_first: Pointer[Scalar[POSITION_DTYPE], MutAnyOrigin],
    bounds_first: Pointer[Scalar[POSITION_DTYPE], MutAnyOrigin],
    positions_second: Pointer[Scalar[POSITION_DTYPE], MutAnyOrigin],
    bounds_second: Pointer[Scalar[POSITION_DTYPE], MutAnyOrigin],
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

    var reduction = stack_allocation[THREADS_PER_BLOCK, Scalar[SCORE_DTYPE], address_space=AddressSpace.SHARED]()
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

    var head_first = Int(first[unsafe_offset=start_first])
    var head_second = Int(second[unsafe_offset=start_second])
    var best = NEGATIVE_INFINITY
    """
    Thread zero owns the three head cases, so only it pays for their reads. Every other thread goes straight to its
    slice of the helix scan.
    """
    if thread_idx.x == 0:
        var neighbours = read_neighbours(table, start_first, length_first, start_second, length_second, rows, columns)
        best = sankoff_cell(neighbours, Int32(substitutions[unsafe_offset=head_first * width + head_second]), scoring)

    var reachable_first = partner_range(bounds_first, head_first, rows, start_first, length_first)
    var reachable_second = partner_range(bounds_second, head_second, columns, start_second, length_second)
    var reachable_second_count = reachable_second.high - reachable_second.low
    var candidates = (reachable_first.high - reachable_first.low) * reachable_second_count
    for flat in range(Int(thread_idx.x), candidates, THREADS_PER_BLOCK):
        var partner_first = Int(positions_first[unsafe_offset=reachable_first.low + flat // reachable_second_count])
        var partner_second = Int(positions_second[unsafe_offset=reachable_second.low + flat % reachable_second_count])
        best = max(
            best,
            helix_candidate(
                table,
                first,
                second,
                substitutions,
                pairs,
                width,
                start_first,
                length_first,
                start_second,
                length_second,
                partner_first - start_first,
                partner_second - start_second,
                rows,
                columns,
            ),
        )

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
    var partners_first = partner_index(first, pairs, alphabet_size)
    var partners_second = partner_index(second, pairs, alphabet_size)

    var first_buffer = upload[SYMBOL_DTYPE](ctx, first)
    var second_buffer = upload[SYMBOL_DTYPE](ctx, second)
    var substitutions_buffer = upload[SUBSTITUTION_DTYPE](ctx, substitutions)
    var pairs_buffer = upload[SUBSTITUTION_DTYPE](ctx, pairs)
    var positions_first_buffer = upload[POSITION_DTYPE](ctx, Span(partners_first.positions))
    var bounds_first_buffer = upload[POSITION_DTYPE](ctx, Span(partners_first.bounds))
    var positions_second_buffer = upload[POSITION_DTYPE](ctx, Span(partners_second.positions))
    var bounds_second_buffer = upload[POSITION_DTYPE](ctx, Span(partners_second.bounds))
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
            positions_first_buffer.unsafe_ptr(),
            bounds_first_buffer.unsafe_ptr(),
            positions_second_buffer.unsafe_ptr(),
            bounds_second_buffer.unsafe_ptr(),
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
    comptime ALIGNED = Self(0)
    """Both heads consumed and aligned to each other, leaving that column unpaired."""
    comptime GAP_IN_SECOND = Self(1)
    """The first head consumed against a gap."""
    comptime GAP_IN_FIRST = Self(2)
    """The second head consumed against a gap."""
    comptime HELIX = Self(3)
    """Both heads consumed, aligned and paired with a later column, which is where covariation is credited."""


@fieldwise_init
struct Decision(ImplicitlyCopyable, TrivialRegisterPassable):
    """A winning case, with the two spans when it opened a helix."""

    var outcome: SankoffCase
    """Which case reproduced the cell's stored score."""
    var span_first: Int32
    """How far the first head reaches to its partner, or zero when no helix opened."""
    var span_second: Int32
    """How far the second head reaches to its partner, or zero when no helix opened."""


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
    decisions would cost as much again for nothing. The walk is one path rather than the whole
    table, so it tests each span for a possible pair instead of carrying the sweep's index.
    """
    var rows = len(first)
    var columns = len(second)
    var stored = table[cell_index(start_first, length_first, start_second, length_second, rows, columns)]
    var head_first = Int(first[start_first])
    var head_second = Int(second[start_second])
    var head_substitution = Int32(substitutions[head_first * alphabet_size + head_second])

    var near = read_neighbours(
        table.unsafe_ptr(), start_first, length_first, start_second, length_second, rows, columns
    )
    if near.aligned + head_substitution == stored:
        return Decision(SankoffCase.ALIGNED, 0, 0)
    if near.gap_in_second + scoring.gap == stored:
        return Decision(SankoffCase.GAP_IN_SECOND, 0, 0)
    if near.gap_in_first + scoring.gap == stored:
        return Decision(SankoffCase.GAP_IN_FIRST, 0, 0)

    for span_first in range(length_first - 1, 0, -1):
        if pairs[head_first * alphabet_size + Int(first[start_first + span_first])] <= 0:
            continue
        for span_second in range(length_second - 1, 0, -1):
            if pairs[head_second * alphabet_size + Int(second[start_second + span_second])] <= 0:
                continue
            var candidate = helix_candidate(
                table.unsafe_ptr(),
                first.unsafe_ptr(),
                second.unsafe_ptr(),
                substitutions.unsafe_ptr(),
                pairs.unsafe_ptr(),
                alphabet_size,
                start_first,
                length_first,
                start_second,
                length_second,
                span_first,
                span_second,
                rows,
                columns,
            )
            if candidate == stored:
                return Decision(SankoffCase.HELIX, Int32(span_first), Int32(span_second))
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

    Every case consumes the two heads first, so the columns arrive in order without a separate
    ordering pass, and a helix emits its opening column, what it encloses, its closing column and
    then whatever follows it.
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
        table,
        first,
        second,
        substitutions,
        pairs,
        alphabet_size,
        scoring,
        start_first,
        length_first,
        start_second,
        length_second,
    )
    var head_first = letters[Int(first[start_first])]
    var head_second = letters[Int(second[start_second])]

    if decision.outcome == SankoffCase.ALIGNED:
        gapped_first.append(head_first)
        gapped_second.append(head_second)
        structure.append(UNPAIRED_BYTE)
        expand_window(
            table,
            first,
            second,
            substitutions,
            pairs,
            letters,
            alphabet_size,
            scoring,
            start_first + 1,
            length_first - 1,
            start_second + 1,
            length_second - 1,
            gapped_first,
            gapped_second,
            structure,
        )
    elif decision.outcome == SankoffCase.GAP_IN_SECOND:
        gapped_first.append(head_first)
        gapped_second.append(GAP_BYTE)
        structure.append(UNPAIRED_BYTE)
        expand_window(
            table,
            first,
            second,
            substitutions,
            pairs,
            letters,
            alphabet_size,
            scoring,
            start_first + 1,
            length_first - 1,
            start_second,
            length_second,
            gapped_first,
            gapped_second,
            structure,
        )
    elif decision.outcome == SankoffCase.GAP_IN_FIRST:
        gapped_first.append(GAP_BYTE)
        gapped_second.append(head_second)
        structure.append(UNPAIRED_BYTE)
        expand_window(
            table,
            first,
            second,
            substitutions,
            pairs,
            letters,
            alphabet_size,
            scoring,
            start_first,
            length_first,
            start_second + 1,
            length_second - 1,
            gapped_first,
            gapped_second,
            structure,
        )
    else:
        var span_first = Int(decision.span_first)
        var span_second = Int(decision.span_second)
        gapped_first.append(head_first)
        gapped_second.append(head_second)
        structure.append(OPEN_BYTE)
        expand_window(
            table,
            first,
            second,
            substitutions,
            pairs,
            letters,
            alphabet_size,
            scoring,
            start_first + 1,
            span_first - 1,
            start_second + 1,
            span_second - 1,
            gapped_first,
            gapped_second,
            structure,
        )
        gapped_first.append(letters[Int(first[start_first + span_first])])
        gapped_second.append(letters[Int(second[start_second + span_second])])
        structure.append(CLOSE_BYTE)
        expand_window(
            table,
            first,
            second,
            substitutions,
            pairs,
            letters,
            alphabet_size,
            scoring,
            start_first + span_first + 1,
            length_first - span_first - 1,
            start_second + span_second + 1,
            length_second - span_second - 1,
            gapped_first,
            gapped_second,
            structure,
        )


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
        table,
        first,
        second,
        substitutions,
        pairs,
        letters,
        alphabet.byte_length(),
        scoring,
        0,
        rows,
        0,
        columns,
        gapped_first,
        gapped_second,
        structure,
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
