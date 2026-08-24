"""
Sankoff simultaneous alignment and folding for CPU and GPU, exact and with traceback.

Sankoff's 1985 recurrence aligns two sequences and folds them together, crediting a base pair only
when both sequences can form it. The signal is covariation rather than thermodynamics, so the
scoring is a substitution matrix and a pair table, with no energy model anywhere.

The table is indexed by `(start, length)` on each sequence rather than by four endpoints. Each
cell decides only what its two heads do, so the recurrence reads strictly smaller lengths in both
dimensions, every cell on the anti-diagonal `window_first + window_second` is independent, and
$O(n^6)$ work becomes `n + m + 1` dependent layers with full parallelism inside each.

The heads are gapped, aligned and unpaired, or aligned and paired with a later column, and that
last case is the whole scan: it names the partner of each head and covers both a pair closing the
window and a pair followed by more structure. Restricting the scan to partners the pair table
actually allows is what keeps it affordable, since only six of the sixteen letter pairs can close.
The traceback tries partners from the furthest back, so a tie resolves to the longest helix and a
stem comes out nested rather than chopped into neighbours. The fill's own order does not matter,
because it keeps only a maximum.

Memory is $O(n^2 m^2)$ and that is inherent, not an implementation limit: a pairing at one layer
reads every layer beneath it, so nothing can be retired and no Hirschberg-style band exists.
Measured at n = 24, a perfect freeing oracle still leaves 75.5% of the table live at peak. The
consolation is that traceback costs nothing extra, since the whole table is resident regardless.

Gaps are linear rather than affine. Affine needs the open state of both ends of both windows,
which multiplies a table that is already what bounds the sequence length.

The recurrence, the tie-breaking and the traceback are transcribed from `cofolding.py`, which
stays the parity oracle.
"""

from std.gpu import block_idx, thread_idx
from std.math import ceildiv
from std.gpu.primitives.warp import WARP_SIZE, max as warp_max

from max.gpu.host import DeviceContext

from errors import AffineGapsError, ErrorKind
from common import (
    CLOSE_BYTE,
    DEFAULT_RNA_ALPHABET,
    GAP_BYTE,
    NEGATIVE_INFINITY,
    OPEN_BYTE,
    SubstitutionDType,
    SymbolDType,
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
comptime MIN_TURN = 3
"""Fewest bases any pair must enclose, the same floor `folding.mojo` applies."""
comptime MIN_CLOSING_REACH = MIN_TURN + 1
"""The turn plus the partner past it: the shortest head-to-partner distance."""
comptime CellDType = DType.int16
"""Storage for one table cell. Narrower than the arithmetic, because the table is what binds."""
comptime PositionDType = DType.int32
comptime WARPS_PER_BLOCK = THREADS_PER_BLOCK // WARP_SIZE


@fieldwise_init
struct SankoffScoring(ImplicitlyCopyable, TrivialRegisterPassable):
    """The linear gap cost. Substitution and pair tables travel separately, as spans."""

    var gap: Int32
    """Cost of leaving one position unaligned. Linear, not affine."""


@fieldwise_init
struct Neighbours(ImplicitlyCopyable, TrivialRegisterPassable):
    """The three cells one Sankoff cell reads outside its pairing scan."""

    var aligned: Int32
    """Both windows give up their first symbol and the two are aligned."""
    var gap_in_second: Int32
    """The first window's head aligns to a gap."""
    var gap_in_first: Int32
    """The second window's head aligns to a gap."""


@always_inline
def read_neighbours(
    table: Pointer[Scalar[CellDType], _],
    windows: WindowPair,
    shape: TableShape,
) -> Neighbours:
    """The three cells one Sankoff cell reads, in the order the recurrence tries them.

    One reader serves the host sweep, the device sweep and the traceback, so a case can never be
    tested against a cell the fill did not use.
    """
    return Neighbours(
        Int32(
            table[
                unsafe_offset=cell_index(
                    WindowPair(
                        windows.start_first + 1,
                        windows.window_first - 1,
                        windows.start_second + 1,
                        windows.window_second - 1,
                    ),
                    shape,
                )
            ]
        ),
        Int32(
            table[
                unsafe_offset=cell_index(
                    WindowPair(
                        windows.start_first + 1, windows.window_first - 1, windows.start_second, windows.window_second
                    ),
                    shape,
                )
            ]
        ),
        Int32(
            table[
                unsafe_offset=cell_index(
                    WindowPair(
                        windows.start_first, windows.window_first, windows.start_second + 1, windows.window_second - 1
                    ),
                    shape,
                )
            ]
        ),
    )


@always_inline
def neighbours_best(neighbours: Neighbours, head_substitution: Int32, scoring: SankoffScoring) -> Int32:
    """The constant-work cases of one Sankoff cell, without the pairing scan.

    Pure arithmetic over values the caller already read, so the host sweep and the device sweep
    share this transcription unchanged. The pairing scan stays with the caller because its shape is
    a serial loop on one and a block reduction on the other.
    """
    var best = neighbours.aligned + head_substitution
    best = max(best, neighbours.gap_in_second + scoring.gap)
    best = max(best, neighbours.gap_in_first + scoring.gap)
    return best


@always_inline
def cell_index(windows: WindowPair, shape: TableShape) -> Int:
    """Row-major over `(start_first, window_first, start_second, window_second)`."""
    var stride = shape.columns + 1
    return (
        (windows.start_first * (shape.rows + 1) + windows.window_first) * stride + windows.start_second
    ) * stride + windows.window_second


@always_inline
def table_cells(rows: Int, columns: Int) -> Int:
    return (rows + 1) * (rows + 1) * (columns + 1) * (columns + 1)


# endregion Scoring

# region Partners


@fieldwise_init
struct PartnerIndex(Copyable, Movable):
    """Where each letter's possible partners sit in one sequence, as one ascending run per letter."""

    var positions: List[Scalar[PositionDType]]
    """Every position that can close a pair, the letters' runs laid end to end."""
    var bounds: List[Scalar[PositionDType]]
    """Letter `c`'s run at its first entry from `t` onwards, held at `c * (length + 1) + t`."""


@fieldwise_init
struct PartnerRange(ImplicitlyCopyable, TrivialRegisterPassable):
    """Half-open slice of one letter's run, covering the partners one window can reach."""

    var low: Int
    """First entry of the run that lands inside the window."""
    var high: Int
    """One past the run's last entry inside the window."""


@fieldwise_init
struct WindowPair(ImplicitlyCopyable, TrivialRegisterPassable):
    """One window into each sequence, which together address one cell."""

    var start_first: Int
    """First position of the window into the first sequence."""
    var window_first: Int
    """How many positions that window covers."""
    var start_second: Int
    """First position of the window into the second sequence."""
    var window_second: Int
    """How many positions that window covers."""


@fieldwise_init
struct TableShape(ImplicitlyCopyable, TrivialRegisterPassable):
    """The two sequence lengths, which are what stride the four-dimensional table."""

    var rows: Int
    """Length of the first sequence."""
    var columns: Int
    """Length of the second sequence."""


def partner_index(
    sequence: ImmSpan[Scalar[SymbolDType], _],
    pairs: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
) -> PartnerIndex:
    """Lists, per letter, every position in the sequence that letter can close a pair with.

    The runs are ascending, so a window's candidates are a contiguous slice and the scan carries
    no test inside it. Six of the sixteen RNA letter pairs can close, which is what makes the
    slice worth building rather than filtering the window on the fly.
    """
    var length = len(sequence)
    var positions = List[Scalar[PositionDType]]()
    var bounds = List[Scalar[PositionDType]](length=alphabet_size * (length + 1), fill=Scalar[PositionDType](0))
    for letter in range(alphabet_size):
        for position in range(length):
            bounds[letter * (length + 1) + position] = Scalar[PositionDType](len(positions))
            if pairs[letter * alphabet_size + Int(sequence[position])] > 0:
                positions.append(Scalar[PositionDType](position))
        bounds[letter * (length + 1) + length] = Scalar[PositionDType](len(positions))
    return PartnerIndex(positions^, bounds^)


@always_inline
def partner_range(
    bounds: Pointer[Scalar[PositionDType], _],
    head: Int,
    sequence_length: Int,
    start: Int,
    window: Int,
) -> PartnerRange:
    """Which of `head`'s partners a window can reach, as a slice of that letter's run.

    A pair has to enclose enough bases to turn the backbone around, so the slice starts past that
    floor, clamped to the window's own end. A window too short to hold a hairpin yields an empty
    slice rather than a partner nobody can pair with.
    """
    var run = head * (sequence_length + 1)
    var floor = min(MIN_CLOSING_REACH, window)
    return PartnerRange(
        Int(bounds[unsafe_offset=run + start + floor]),
        Int(bounds[unsafe_offset=run + start + window]),
    )


@fieldwise_init
struct PairedHeads(ImplicitlyCopyable, TrivialRegisterPassable):
    """What a cell's two heads contribute to every candidate they can open.

    None of it depends on the partners, so a cell reads it once instead of once per candidate, and
    the partner scan is the hottest loop in the recurrence.
    """

    var first: Int
    """Encoded head of the first sequence."""
    var second: Int
    """Encoded head of the second sequence."""
    var substitution: Int32
    """What aligning the two heads scores, fixed for the cell."""


@always_inline
def paired_heads(
    first: Pointer[Scalar[SymbolDType], _],
    second: Pointer[Scalar[SymbolDType], _],
    substitutions: Pointer[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    start_first: Int,
    start_second: Int,
) -> PairedHeads:
    """Everything one cell's two heads contribute, hoisted out of the partner scan."""
    var head_first = Int(first[unsafe_offset=start_first])
    var head_second = Int(second[unsafe_offset=start_second])
    return PairedHeads(
        head_first,
        head_second,
        Int32(substitutions[unsafe_offset=head_first * alphabet_size + head_second]),
    )


@always_inline
def paired_candidate(
    table: Pointer[Scalar[CellDType], _],
    first: Pointer[Scalar[SymbolDType], _],
    second: Pointer[Scalar[SymbolDType], _],
    substitutions: Pointer[Scalar[SubstitutionDType], _],
    pairs: Pointer[Scalar[SubstitutionDType], _],
    heads: PairedHeads,
    alphabet_size: Int,
    windows: WindowPair,
    reach_first: Int,
    reach_second: Int,
    shape: TableShape,
) -> Int32:
    """What one pairing scores: its two closing columns, what they enclose, and what follows it.

    One transcription serves the host sweep, the device sweep and the traceback, so a pairing can
    never be tested against a score the fill did not use.
    """
    var partner_first = Int(first[unsafe_offset=windows.start_first + reach_first])
    var partner_second = Int(second[unsafe_offset=windows.start_second + reach_second])
    var inside = Int32(
        table[
            unsafe_offset=cell_index(
                WindowPair(windows.start_first + 1, reach_first - 1, windows.start_second + 1, reach_second - 1),
                shape,
            )
        ]
    )
    var after = Int32(
        table[
            unsafe_offset=cell_index(
                WindowPair(
                    windows.start_first + reach_first + 1,
                    windows.window_first - reach_first - 1,
                    windows.start_second + reach_second + 1,
                    windows.window_second - reach_second - 1,
                ),
                shape,
            )
        ]
    )
    return (
        inside
        + after
        + Int32(pairs[unsafe_offset=heads.first * alphabet_size + partner_first])
        + Int32(pairs[unsafe_offset=heads.second * alphabet_size + partner_second])
        + heads.substitution
        + Int32(substitutions[unsafe_offset=partner_first * alphabet_size + partner_second])
    )


# endregion Partners

# region Serial Reference


@always_inline
def paired_best(
    table: Pointer[Scalar[CellDType], _],
    first: Pointer[Scalar[SymbolDType], _],
    second: Pointer[Scalar[SymbolDType], _],
    substitutions: Pointer[Scalar[SubstitutionDType], _],
    pairs: Pointer[Scalar[SubstitutionDType], _],
    positions_first: Pointer[Scalar[PositionDType], _],
    bounds_first: Pointer[Scalar[PositionDType], _],
    positions_second: Pointer[Scalar[PositionDType], _],
    bounds_second: Pointer[Scalar[PositionDType], _],
    alphabet_size: Int,
    windows: WindowPair,
    shape: TableShape,
) -> Int32:
    """The best pairing the two heads can open, over every partner both windows can reach."""
    var best = NEGATIVE_INFINITY
    var heads = paired_heads(first, second, substitutions, alphabet_size, windows.start_first, windows.start_second)
    var head_first = heads.first
    var head_second = heads.second
    var reachable_first = partner_range(bounds_first, head_first, shape.rows, windows.start_first, windows.window_first)
    var reachable_second = partner_range(
        bounds_second, head_second, shape.columns, windows.start_second, windows.window_second
    )
    for index_first in range(reachable_first.high - 1, reachable_first.low - 1, -1):
        var partner_first = Int(positions_first[unsafe_offset=index_first])
        for index_second in range(reachable_second.high - 1, reachable_second.low - 1, -1):
            best = max(
                best,
                paired_candidate(
                    table,
                    first,
                    second,
                    substitutions,
                    pairs,
                    heads,
                    alphabet_size,
                    windows,
                    partner_first - windows.start_first,
                    Int(positions_second[unsafe_offset=index_second]) - windows.start_second,
                    shape,
                ),
            )
    return best


@always_inline
def cofold_cell(
    table: Pointer[Scalar[CellDType], _],
    first: Pointer[Scalar[SymbolDType], _],
    second: Pointer[Scalar[SymbolDType], _],
    substitutions: Pointer[Scalar[SubstitutionDType], _],
    pairs: Pointer[Scalar[SubstitutionDType], _],
    positions_first: Pointer[Scalar[PositionDType], _],
    bounds_first: Pointer[Scalar[PositionDType], _],
    positions_second: Pointer[Scalar[PositionDType], _],
    bounds_second: Pointer[Scalar[PositionDType], _],
    alphabet_size: Int,
    scoring: SankoffScoring,
    windows: WindowPair,
    shape: TableShape,
) -> Int32:
    """Every case of one Sankoff cell, including the pairing scan.

    An empty window on either side can only be gapped through, which is what makes the base cases
    a pair of early returns rather than a branch wrapped around the whole body.
    """
    if windows.window_first == 0:
        return Int32(windows.window_second) * scoring.gap
    if windows.window_second == 0:
        return Int32(windows.window_first) * scoring.gap

    var head_first = Int(first[unsafe_offset=windows.start_first])
    var head_second = Int(second[unsafe_offset=windows.start_second])
    var neighbours = read_neighbours(table, windows, shape)
    var best = neighbours_best(
        neighbours, Int32(substitutions[unsafe_offset=head_first * alphabet_size + head_second]), scoring
    )
    return max(
        best,
        paired_best(
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
            windows,
            shape,
        ),
    )


@always_inline
def check_cell_range(
    shape: TableShape,
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    pairs: ImmSpan[Scalar[SubstitutionDType], _],
    scoring: SankoffScoring,
) raises AffineGapsError:
    """Refuses scoring that could drive a cell past what one cell can hold.

    An alignment spans at most `rows + columns` columns, and a column earns either a substitution
    or a gap, plus at most one closing pair's share, so those two magnitudes bound the table. The
    default scoring leaves room for sequences far longer than the recurrence can reach.
    """
    var widest_column = Int(abs(scoring.gap))
    for index in range(len(substitutions)):
        widest_column = max(widest_column, Int(abs(Int32(substitutions[index]))))
    var widest_pair = 0
    for index in range(len(pairs)):
        widest_pair = max(widest_pair, Int(abs(Int32(pairs[index]))))
    if (shape.rows + shape.columns) * (widest_column + widest_pair) > Int(Scalar[CellDType].MAX):
        raise AffineGapsError(ErrorKind.INVALID_SCORING, "Sankoff scoring overflows a table cell")


def serial_cofold_table(
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    pairs: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: SankoffScoring,
) raises -> List[Scalar[CellDType]]:
    """Fills the whole table on the host, in increasing order of each window's length.

    Kept as the definition the device sweep is differentially tested against, so it favours
    following `cofolding.py` line for line over being fast.
    """
    var rows = len(first)
    var columns = len(second)
    var shape = TableShape(rows, columns)
    check_cell_range(shape, substitutions, pairs, scoring)
    var partners_first = partner_index(first, pairs, alphabet_size)
    var partners_second = partner_index(second, pairs, alphabet_size)
    var table = List[Scalar[CellDType]](length=table_cells(rows, columns), fill=Scalar[CellDType](0))
    var cells = table.unsafe_ptr()
    var first_cursor = first.unsafe_ptr()
    var second_cursor = second.unsafe_ptr()
    var substitutions_cursor = substitutions.unsafe_ptr()
    var pairs_cursor = pairs.unsafe_ptr()
    var positions_first = partners_first.positions.unsafe_ptr()
    var bounds_first = partners_first.bounds.unsafe_ptr()
    var positions_second = partners_second.positions.unsafe_ptr()
    var bounds_second = partners_second.bounds.unsafe_ptr()

    for window_first in range(rows + 1):
        for window_second in range(columns + 1):
            if window_first == 0 and window_second == 0:
                continue
            for start_first in range(rows - window_first + 1):
                for start_second in range(columns - window_second + 1):
                    var windows = WindowPair(start_first, window_first, start_second, window_second)
                    var here = cell_index(windows, shape)
                    cells[unsafe_offset=here] = Scalar[CellDType](
                        cofold_cell(
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
                            windows,
                            shape,
                        )
                    )
    return table^


# endregion Serial Reference

# region GPU Sweep


@fieldwise_init
struct AntiDiagonalPlan(Copyable, Movable):
    """Where every cell of every anti-diagonal sits in one flat numbering per layer.

    A launch sized from the lengths alone spends most of its blocks on start positions the window
    cannot reach. Numbering the cells that exist instead lets the grid be exactly as wide as the
    layer, at the cost of one small search to undo the numbering.
    """

    var offsets: List[Scalar[PositionDType]]
    """Cells preceding each window length within its layer, the layers laid end to end."""
    var bases: List[Scalar[PositionDType]]
    """Where a layer's run of offsets begins, one entry per anti-diagonal."""


def anti_diagonal_plan(rows: Int, columns: Int) -> AntiDiagonalPlan:
    """Counts the cells of each anti-diagonal, cumulatively over the first window's length."""
    var offsets = List[Scalar[PositionDType]]()
    var bases = List[Scalar[PositionDType]]()
    for anti_diagonal in range(rows + columns + 1):
        bases.append(Scalar[PositionDType](len(offsets)))
        var running = 0
        for window_first in range(max(0, anti_diagonal - columns), min(rows, anti_diagonal) + 1):
            offsets.append(Scalar[PositionDType](running))
            running += (rows - window_first + 1) * (columns - (anti_diagonal - window_first) + 1)
        offsets.append(Scalar[PositionDType](running))
    return AntiDiagonalPlan(offsets^, bases^)


@always_inline
def anti_diagonal_cells(plan: AntiDiagonalPlan, anti_diagonal: Int, span: Int) -> Int:
    """How many cells one anti-diagonal holds, which is how wide its launch has to be."""
    return Int(plan.offsets[Int(plan.bases[anti_diagonal]) + span])


def cofold_anti_diagonal_kernel(
    first: Pointer[Scalar[SymbolDType], MutAnyOrigin],
    second: Pointer[Scalar[SymbolDType], MutAnyOrigin],
    substitutions: Pointer[Scalar[SubstitutionDType], MutAnyOrigin],
    pairs: Pointer[Scalar[SubstitutionDType], MutAnyOrigin],
    positions_first: Pointer[Scalar[PositionDType], MutAnyOrigin],
    bounds_first: Pointer[Scalar[PositionDType], MutAnyOrigin],
    positions_second: Pointer[Scalar[PositionDType], MutAnyOrigin],
    bounds_second: Pointer[Scalar[PositionDType], MutAnyOrigin],
    offsets: Pointer[Scalar[PositionDType], MutAnyOrigin],
    table: Pointer[Scalar[CellDType], MutAnyOrigin],
    rows_in: Int32,
    columns_in: Int32,
    alphabet_size: Int32,
    anti_diagonal: Int32,
    smallest_window_first: Int32,
    anti_diagonal_base: Int32,
    window_count: Int32,
    gap: Int32,
):
    """One warp per cell of the anti-diagonal `window_first + window_second`.

    Cells on an anti-diagonal read only strictly smaller lengths, so the whole layer is independent and
    one kernel launch per anti-diagonal is all the synchronization the recurrence needs. A warp rather
    than a block owns a cell, so the scan reduces through registers and a cell too small to fill
    the warp wastes lanes rather than a whole block.
    """
    var shape = TableShape(Int(rows_in), Int(columns_in))
    var width = Int(alphabet_size)
    var base = Int(anti_diagonal_base)
    var span = Int(window_count)
    var cell = Int(block_idx.x) * WARPS_PER_BLOCK + Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    if cell >= Int(offsets[unsafe_offset=base + span]):
        return

    var low = 0
    var high = span
    """
    Which window length owns this cell, by bisecting the layer's cumulative counts. The lengths run
    to a few hundred, so this is a handful of loads every lane of the warp makes together.
    """
    while low + 1 < high:
        var middle = (low + high) // 2
        if Int(offsets[unsafe_offset=base + middle]) <= cell:
            low = middle
        else:
            high = middle

    var window_first = Int(smallest_window_first) + low
    var window_second = Int(anti_diagonal) - window_first
    var within = cell - Int(offsets[unsafe_offset=base + low])
    var starts_second = shape.columns - window_second + 1
    var windows = WindowPair(within // starts_second, window_first, within % starts_second, window_second)
    var here = cell_index(windows, shape)
    var scoring = SankoffScoring(gap)

    if window_first == 0 and window_second == 0:
        return
    if window_first == 0:
        if lane == 0:
            table[unsafe_offset=here] = Scalar[CellDType](Int32(window_second) * gap)
        return
    if window_second == 0:
        if lane == 0:
            table[unsafe_offset=here] = Scalar[CellDType](Int32(window_first) * gap)
        return

    var heads = paired_heads(first, second, substitutions, width, windows.start_first, windows.start_second)
    var head_first = heads.first
    var head_second = heads.second
    var best = NEGATIVE_INFINITY
    """
    Lane zero owns the three head cases, so only it pays for their reads. Every other lane goes straight to its slice
    of the pairing scan.
    """
    if lane == 0:
        var neighbours = read_neighbours(table, windows, shape)
        best = neighbours_best(neighbours, heads.substitution, scoring)

    var reachable_first = partner_range(bounds_first, head_first, shape.rows, windows.start_first, windows.window_first)
    var reachable_second = partner_range(
        bounds_second, head_second, shape.columns, windows.start_second, windows.window_second
    )
    var reachable_second_count = reachable_second.high - reachable_second.low
    var candidates = (reachable_first.high - reachable_first.low) * reachable_second_count
    for candidate in range(lane, candidates, WARP_SIZE):
        var partner_first = Int(
            positions_first[unsafe_offset=reachable_first.low + candidate // reachable_second_count]
        )
        var partner_second = Int(
            positions_second[unsafe_offset=reachable_second.low + candidate % reachable_second_count]
        )
        best = max(
            best,
            paired_candidate(
                table,
                first,
                second,
                substitutions,
                pairs,
                heads,
                width,
                windows,
                partner_first - windows.start_first,
                partner_second - windows.start_second,
                shape,
            ),
        )

    var answer = warp_max(best)
    if lane == 0:
        table[unsafe_offset=here] = Scalar[CellDType](answer)


def device_cofold_table(
    ctx: DeviceContext,
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    pairs: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: SankoffScoring,
) raises -> List[Scalar[CellDType]]:
    """Sweeps the table on the device, one launch per anti-diagonal of lengths."""
    var rows = len(first)
    var columns = len(second)
    var cells = table_cells(rows, columns)
    check_cell_range(TableShape(rows, columns), substitutions, pairs, scoring)
    var partners_first = partner_index(first, pairs, alphabet_size)
    var partners_second = partner_index(second, pairs, alphabet_size)
    var plan = anti_diagonal_plan(rows, columns)

    var first_buffer = upload[SymbolDType](ctx, first)
    var second_buffer = upload[SymbolDType](ctx, second)
    var substitutions_buffer = upload[SubstitutionDType](ctx, substitutions)
    var pairs_buffer = upload[SubstitutionDType](ctx, pairs)
    var positions_first_buffer = upload[PositionDType](ctx, Span(partners_first.positions))
    var bounds_first_buffer = upload[PositionDType](ctx, Span(partners_first.bounds))
    var positions_second_buffer = upload[PositionDType](ctx, Span(partners_second.positions))
    var bounds_second_buffer = upload[PositionDType](ctx, Span(partners_second.bounds))
    var offsets_buffer = upload[PositionDType](ctx, Span(plan.offsets))
    var table_buffer = zeroed[CellDType](ctx, cells)

    for anti_diagonal in range(rows + columns + 1):
        var smallest = max(0, anti_diagonal - columns)
        var span = min(rows, anti_diagonal) - smallest + 1
        var layer = anti_diagonal_cells(plan, anti_diagonal, span)
        if layer == 0:
            continue
        ctx.enqueue_function[cofold_anti_diagonal_kernel](
            first_buffer.unsafe_ptr(),
            second_buffer.unsafe_ptr(),
            substitutions_buffer.unsafe_ptr(),
            pairs_buffer.unsafe_ptr(),
            positions_first_buffer.unsafe_ptr(),
            bounds_first_buffer.unsafe_ptr(),
            positions_second_buffer.unsafe_ptr(),
            bounds_second_buffer.unsafe_ptr(),
            offsets_buffer.unsafe_ptr(),
            table_buffer.unsafe_ptr(),
            Int32(rows),
            Int32(columns),
            Int32(alphabet_size),
            Int32(anti_diagonal),
            Int32(smallest),
            plan.bases[anti_diagonal],
            Int32(span),
            scoring.gap,
            grid_dim=ceildiv(layer, WARPS_PER_BLOCK),
            block_dim=THREADS_PER_BLOCK,
        )
    ctx.synchronize()

    # The copy overwrites every byte, so filling this first is a memset of the whole footprint.
    var table = List[Scalar[CellDType]](unsafe_uninit_length=cells)
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
    comptime PAIRED = Self(3)
    """Both heads consumed, aligned and paired with a later column, which is where covariation is credited."""


@fieldwise_init
struct Decision(ImplicitlyCopyable, TrivialRegisterPassable):
    """A winning case, with the two reaches when it opened a pair."""

    var chosen: SankoffCase
    """Which case reproduced the cell's stored score."""
    var reach_first: Int32
    """How far the first head reaches to its partner, or zero when no pair opened."""
    var reach_second: Int32
    """How far the second head reaches to its partner, or zero when no pair opened."""


def winning_case(
    table: ImmSpan[Scalar[CellDType], _],
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    pairs: ImmSpan[Scalar[SubstitutionDType], _],
    alphabet_size: Int,
    scoring: SankoffScoring,
    windows: WindowPair,
) raises AffineGapsError -> Decision:
    """Which case reproduces a cell's stored score, tried in the order the sweep tried them.

    Re-derived rather than recorded: the table is resident anyway, so a parallel array of
    decisions would cost as much again for nothing. The walk is one path rather than the whole
    table, so it tests each span for a possible pair instead of carrying the sweep's index.
    """
    var shape = TableShape(len(first), len(second))
    var stored = Int32(table[cell_index(windows, shape)])
    var heads = paired_heads(
        first.unsafe_ptr(),
        second.unsafe_ptr(),
        substitutions.unsafe_ptr(),
        alphabet_size,
        windows.start_first,
        windows.start_second,
    )
    var head_first = heads.first
    var head_second = heads.second
    var head_substitution = heads.substitution

    var near = read_neighbours(table.unsafe_ptr(), windows, shape)
    if near.aligned + head_substitution == stored:
        return Decision(SankoffCase.ALIGNED, 0, 0)
    if near.gap_in_second + scoring.gap == stored:
        return Decision(SankoffCase.GAP_IN_SECOND, 0, 0)
    if near.gap_in_first + scoring.gap == stored:
        return Decision(SankoffCase.GAP_IN_FIRST, 0, 0)

    for reach_first in range(windows.window_first - 1, MIN_TURN, -1):
        if pairs[head_first * alphabet_size + Int(first[windows.start_first + reach_first])] <= 0:
            continue
        for reach_second in range(windows.window_second - 1, MIN_TURN, -1):
            if pairs[head_second * alphabet_size + Int(second[windows.start_second + reach_second])] <= 0:
                continue
            var candidate = paired_candidate(
                table.unsafe_ptr(),
                first.unsafe_ptr(),
                second.unsafe_ptr(),
                substitutions.unsafe_ptr(),
                pairs.unsafe_ptr(),
                heads,
                alphabet_size,
                windows,
                reach_first,
                reach_second,
                shape,
            )
            if candidate == stored:
                return Decision(SankoffCase.PAIRED, Int32(reach_first), Int32(reach_second))
    raise AffineGapsError(ErrorKind.INCONSISTENT_TABLE, "Sankoff traceback")


def expand_window(
    table: ImmSpan[Scalar[CellDType], _],
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    pairs: ImmSpan[Scalar[SubstitutionDType], _],
    letters: ImmSpan[Byte, _],
    alphabet_size: Int,
    scoring: SankoffScoring,
    windows: WindowPair,
    mut gapped_first: List[Byte],
    mut gapped_second: List[Byte],
    mut structure: List[Byte],
) raises:
    """Appends the columns one window contributes, left to right.

    Every case consumes the two heads first, so the columns arrive in order without a separate
    ordering pass, and a pairing emits its opening column, what it encloses, its closing column and
    then whatever follows it.
    """
    if windows.window_first == 0 and windows.window_second == 0:
        return
    if windows.window_first == 0:
        for offset in range(windows.window_second):
            gapped_first.append(GAP_BYTE)
            gapped_second.append(letters[Int(second[windows.start_second + offset])])
            structure.append(UNPAIRED_BYTE)
        return
    if windows.window_second == 0:
        for offset in range(windows.window_first):
            gapped_first.append(letters[Int(first[windows.start_first + offset])])
            gapped_second.append(GAP_BYTE)
            structure.append(UNPAIRED_BYTE)
        return

    var decision = winning_case(table, first, second, substitutions, pairs, alphabet_size, scoring, windows)
    var head_first = letters[Int(first[windows.start_first])]
    var head_second = letters[Int(second[windows.start_second])]

    if decision.chosen == SankoffCase.ALIGNED:
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
            WindowPair(
                windows.start_first + 1,
                windows.window_first - 1,
                windows.start_second + 1,
                windows.window_second - 1,
            ),
            gapped_first,
            gapped_second,
            structure,
        )
    elif decision.chosen == SankoffCase.GAP_IN_SECOND:
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
            WindowPair(windows.start_first + 1, windows.window_first - 1, windows.start_second, windows.window_second),
            gapped_first,
            gapped_second,
            structure,
        )
    elif decision.chosen == SankoffCase.GAP_IN_FIRST:
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
            WindowPair(windows.start_first, windows.window_first, windows.start_second + 1, windows.window_second - 1),
            gapped_first,
            gapped_second,
            structure,
        )
    else:
        var reach_first = Int(decision.reach_first)
        var reach_second = Int(decision.reach_second)
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
            WindowPair(windows.start_first + 1, reach_first - 1, windows.start_second + 1, reach_second - 1),
            gapped_first,
            gapped_second,
            structure,
        )
        gapped_first.append(letters[Int(first[windows.start_first + reach_first])])
        gapped_second.append(letters[Int(second[windows.start_second + reach_second])])
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
            WindowPair(
                windows.start_first + reach_first + 1,
                windows.window_first - reach_first - 1,
                windows.start_second + reach_second + 1,
                windows.window_second - reach_second - 1,
            ),
            gapped_first,
            gapped_second,
            structure,
        )


# endregion Traceback

# region Interface


def default_rna_pair_matrix() -> List[Scalar[SubstitutionDType]]:
    """The six chemically possible pairings over `DEFAULT_RNA_ALPHABET`, strongest first.

    Watson-Crick pairs score above the wobble pair; everything else is zero and cannot close.
    These rank pairings, they are not measured energies.
    """
    var pairs = List[Scalar[SubstitutionDType]](
        length=DEFAULT_RNA_ALPHABET_SIZE * DEFAULT_RNA_ALPHABET_SIZE, fill=Scalar[SubstitutionDType](0)
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
    table: ImmSpan[Scalar[CellDType], _],
    first: ImmSpan[Scalar[SymbolDType], _],
    second: ImmSpan[Scalar[SymbolDType], _],
    substitutions: ImmSpan[Scalar[SubstitutionDType], _],
    pairs: ImmSpan[Scalar[SubstitutionDType], _],
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
        WindowPair(0, rows, 0, columns),
        gapped_first,
        gapped_second,
        structure,
    )
    return CofoldResult(
        String(unsafe_from_utf8=gapped_first),
        String(unsafe_from_utf8=gapped_second),
        String(unsafe_from_utf8=structure),
        Int32(table[cell_index(WindowPair(0, rows, 0, columns), TableShape(rows, columns))]),
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
