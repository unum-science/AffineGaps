"""
Zuker minimum free energy folding for CPU and GPU, exact and with traceback.

Three tables are filled in step: `paired` for a subsequence whose ends pair, `multiloop` for one
inside a multibranched loop, and `external` for one that is not enclosed. Both `multiloop` and
`external` read `paired` at the same span, so a span is swept in that order and the device runs
three launches per span rather than one.

Everything is indexed by `(start, length)`, so a bifurcation reads strictly smaller lengths and
every cell of a span is independent. Memory is `O(n^2)` and time is `O(n^3)`, the interior-loop
term bounded by capping a loop at `MAX_LOOP` unpaired bases as this recurrence always is.

Energies are integer decikilocalories per mole, so a fold is reproducible bit for bit rather than
depending on floating-point association order.

The energy model is the subset described in `turner.mojo`. It reproduces RNAstructure's `efn2`
exactly for structures built from stacks and hairpins, and differs where dangling ends, coaxial
stacking, tetraloop bonuses or the special small-internal-loop tables would apply.

The recurrence, the tie-breaking and the traceback are transcribed from `folding.py`, the oracle.
"""

from std.gpu import block_idx, thread_idx
from std.math import log
from std.memory import stack_allocation
from std.memory.pointer import AddressSpace

from max.gpu import barrier
from max.gpu.host import DeviceContext

from common import (
    CLOSE_BYTE, DEFAULT_RNA_ALPHABET, OPEN_BYTE, SYMBOL_DTYPE, THREADS_PER_BLOCK, UNPAIRED_BYTE,
    translate, upload, zeroed,
)
from turner import (
    BULGE_INITIATION, DANGLE_AFTER, DANGLE_BEFORE, ENERGY_DTYPE, FORBIDDEN, HAIRPIN_INITIATION,
    HEXALOOP_ENERGIES, HEXALOOP_KEYS, INTERNAL_INITIATION, LOOP_LIMIT, MULTILOOP_OFFSET,
    MULTILOOP_PER_HELIX, MULTILOOP_PER_UNPAIRED, NINIO_CAP, NINIO_PER_ASYMMETRY, PAIR_INDEX,
    PAIR_TYPES, STACK, TERMINAL_AU, TERMINAL_MISMATCH_HAIRPIN,
    TERMINAL_MISMATCH_INTERNAL, TETRALOOP_ENERGIES, TETRALOOP_KEYS, TRILOOP_ENERGIES, TRILOOP_KEYS,
)

# region Energy Model

comptime MAX_LOOP = LOOP_LIMIT
comptime MIN_HAIRPIN = 3

comptime PAIR_CG = 1
"""The two Watson-Crick pairs that are not charged a helix-end penalty."""
comptime PAIR_GC = 2

comptime TRILOOP_COUNT = len(TRILOOP_KEYS)
"""One buffer carries every table, so a kernel takes one pointer rather than six."""
comptime TETRALOOP_COUNT = len(TETRALOOP_KEYS)
comptime HEXALOOP_COUNT = len(HEXALOOP_KEYS)

comptime STACK_OFFSET = 0
comptime MISMATCH_OFFSET = STACK_OFFSET + PAIR_TYPES * PAIR_TYPES
comptime MISMATCH_INTERNAL_OFFSET = MISMATCH_OFFSET + PAIR_TYPES * 16
comptime DANGLE_AFTER_OFFSET = MISMATCH_INTERNAL_OFFSET + PAIR_TYPES * 16
comptime DANGLE_BEFORE_OFFSET = DANGLE_AFTER_OFFSET + PAIR_TYPES * 4
comptime HAIRPIN_OFFSET = DANGLE_BEFORE_OFFSET + PAIR_TYPES * 4
comptime BULGE_OFFSET = HAIRPIN_OFFSET + LOOP_LIMIT + 1
comptime INTERNAL_OFFSET = BULGE_OFFSET + LOOP_LIMIT + 1
comptime PAIR_INDEX_OFFSET = INTERNAL_OFFSET + LOOP_LIMIT + 1
comptime TRILOOP_KEY_OFFSET = PAIR_INDEX_OFFSET + 16
comptime TRILOOP_ENERGY_OFFSET = TRILOOP_KEY_OFFSET + TRILOOP_COUNT
comptime TETRALOOP_KEY_OFFSET = TRILOOP_ENERGY_OFFSET + TRILOOP_COUNT
comptime TETRALOOP_ENERGY_OFFSET = TETRALOOP_KEY_OFFSET + TETRALOOP_COUNT
comptime HEXALOOP_KEY_OFFSET = TETRALOOP_ENERGY_OFFSET + TETRALOOP_COUNT
comptime HEXALOOP_ENERGY_OFFSET = HEXALOOP_KEY_OFFSET + HEXALOOP_COUNT
comptime TABLE_CELLS = HEXALOOP_ENERGY_OFFSET + HEXALOOP_COUNT


def packed_turner_tables() -> List[Scalar[ENERGY_DTYPE]]:
    """Every table this recurrence reads, concatenated at the offsets above."""
    var stack = materialize[STACK]()
    var hairpin_mismatch = materialize[TERMINAL_MISMATCH_HAIRPIN]()
    var internal_mismatch = materialize[TERMINAL_MISMATCH_INTERNAL]()
    var after = materialize[DANGLE_AFTER]()
    var before = materialize[DANGLE_BEFORE]()
    var hairpin = materialize[HAIRPIN_INITIATION]()
    var bulge = materialize[BULGE_INITIATION]()
    var internal = materialize[INTERNAL_INITIATION]()
    var pairs = materialize[PAIR_INDEX]()
    var triloop_keys = materialize[TRILOOP_KEYS]()
    var triloop_energies = materialize[TRILOOP_ENERGIES]()
    var tetraloop_keys = materialize[TETRALOOP_KEYS]()
    var tetraloop_energies = materialize[TETRALOOP_ENERGIES]()
    var hexaloop_keys = materialize[HEXALOOP_KEYS]()
    var hexaloop_energies = materialize[HEXALOOP_ENERGIES]()

    var packed = List[Scalar[ENERGY_DTYPE]](length=TABLE_CELLS, fill=Scalar[ENERGY_DTYPE](0))
    for index in range(PAIR_TYPES * PAIR_TYPES):
        packed[STACK_OFFSET + index] = stack[index]
    for index in range(PAIR_TYPES * 16):
        packed[MISMATCH_OFFSET + index] = hairpin_mismatch[index]
        packed[MISMATCH_INTERNAL_OFFSET + index] = internal_mismatch[index]
    for index in range(PAIR_TYPES * 4):
        packed[DANGLE_AFTER_OFFSET + index] = after[index]
        packed[DANGLE_BEFORE_OFFSET + index] = before[index]
    for index in range(LOOP_LIMIT + 1):
        packed[HAIRPIN_OFFSET + index] = hairpin[index]
        packed[BULGE_OFFSET + index] = bulge[index]
        packed[INTERNAL_OFFSET + index] = internal[index]
    for index in range(16):
        packed[PAIR_INDEX_OFFSET + index] = pairs[index]
    for index in range(TRILOOP_COUNT):
        packed[TRILOOP_KEY_OFFSET + index] = triloop_keys[index]
        packed[TRILOOP_ENERGY_OFFSET + index] = triloop_energies[index]
    for index in range(TETRALOOP_COUNT):
        packed[TETRALOOP_KEY_OFFSET + index] = tetraloop_keys[index]
        packed[TETRALOOP_ENERGY_OFFSET + index] = tetraloop_energies[index]
    for index in range(HEXALOOP_COUNT):
        packed[HEXALOOP_KEY_OFFSET + index] = hexaloop_keys[index]
        packed[HEXALOOP_ENERGY_OFFSET + index] = hexaloop_energies[index]
    return packed^


comptime NO_PAIR = -1
"""What `pair_of` answers when two bases form none of the six pairs."""


@always_inline
def pairs(index: Int) -> Bool:
    """Whether a `pair_of` answer names a real pair."""
    return index != NO_PAIR


@always_inline
def pair_of(
    tables: Pointer[Scalar[ENERGY_DTYPE], _], left: Int, right: Int
) -> Int:
    """Which of the six pair types two bases form, or a negative value for none."""
    return Int(tables[unsafe_offset = PAIR_INDEX_OFFSET + left * 4 + right])


@always_inline
def terminal_penalty(pair: Int) -> Int32:
    """The helix-end penalty, charged to everything but a Watson-Crick CG or GC pair."""
    if pair == PAIR_CG or pair == PAIR_GC:
        return Int32(0)
    return TERMINAL_AU


@always_inline
def packed_key(
    sequence: Pointer[Scalar[SYMBOL_DTYPE], _], start: Int, end: Int
) -> Int32:
    """The closing pair and loop bases packed base-four, most significant first."""
    var key = Int32(0)
    for index in range(start, end + 1):
        key = key * 4 + Int32(Int(sequence[unsafe_offset=index]))
    return key


@always_inline
def special_hairpin(
    sequence: Pointer[Scalar[SYMBOL_DTYPE], _],
    tables: Pointer[Scalar[ENERGY_DTYPE], _],
    start: Int,
    end: Int,
    size: Int,
) -> Int32:
    """A tabulated hairpin energy for this exact loop, or `FORBIDDEN` when there is none.

    These replace initiation and terminal mismatch rather than adding to them, which is how the
    tables are defined and how `efn2` applies them.
    """
    var keys: Int
    var energies: Int
    var count: Int
    if size == 3:
        keys, energies, count = TRILOOP_KEY_OFFSET, TRILOOP_ENERGY_OFFSET, TRILOOP_COUNT
    elif size == 4:
        keys, energies, count = TETRALOOP_KEY_OFFSET, TETRALOOP_ENERGY_OFFSET, TETRALOOP_COUNT
    elif size == 6:
        keys, energies, count = HEXALOOP_KEY_OFFSET, HEXALOOP_ENERGY_OFFSET, HEXALOOP_COUNT
    else:
        return FORBIDDEN
    var key = packed_key(sequence, start, end)
    for index in range(count):
        if tables[unsafe_offset = keys + index] == key:
            return tables[unsafe_offset = energies + index]
    return FORBIDDEN


@always_inline
def dangle_energy(
    sequence: Pointer[Scalar[SYMBOL_DTYPE], _],
    tables: Pointer[Scalar[ENERGY_DTYPE], _],
    start: Int,
    end: Int,
    length: Int,
) -> Int32:
    """Both dangles on a helix placed in an exterior loop or a multiloop.

    The pair is read from outside the helix, and both neighbours are charged whenever they exist,
    which is the simple treatment that needs no extra states in the recurrence.
    """
    var outward = pair_of(
        tables, Int(sequence[unsafe_offset=end]), Int(sequence[unsafe_offset=start])
    )
    if not pairs(outward):
        return Int32(0)
    var total = Int32(0)
    if start > 0:
        total += tables[
            unsafe_offset = DANGLE_BEFORE_OFFSET
            + outward * 4
            + Int(sequence[unsafe_offset = start - 1])
        ]
    if end < length - 1:
        total += tables[
            unsafe_offset = DANGLE_AFTER_OFFSET
            + outward * 4
            + Int(sequence[unsafe_offset = end + 1])
        ]
    return total


@always_inline
def hairpin_energy(
    sequence: Pointer[Scalar[SYMBOL_DTYPE], _],
    tables: Pointer[Scalar[ENERGY_DTYPE], _],
    start: Int,
    end: Int,
) -> Int32:
    """A hairpin closed by `(start, end)`, with everything between it unpaired."""
    var size = end - start - 1
    if size < MIN_HAIRPIN:
        return FORBIDDEN
    var pair = pair_of(tables, Int(sequence[unsafe_offset=start]), Int(sequence[unsafe_offset=end]))
    if not pairs(pair):
        return FORBIDDEN
    var tabulated = special_hairpin(sequence, tables, start, end, size)
    if tabulated < FORBIDDEN:
        return tabulated
    if size > LOOP_LIMIT:
        var ratio = Float64(size) / Float64(LOOP_LIMIT)
        """
        Beyond the tabulated sizes the model extrapolates by polymer theory, which needs a logarithm; capping instead
        would make long loops artificially cheap.
        """
        var extrapolated = Int32(round(10.79 * log(ratio)))
        return tables[unsafe_offset = HAIRPIN_OFFSET + LOOP_LIMIT] + extrapolated
    var initiation = tables[unsafe_offset = HAIRPIN_OFFSET + size]
    if size == MIN_HAIRPIN:
        return initiation + terminal_penalty(pair)
    var first_unpaired = Int(sequence[unsafe_offset = start + 1])
    var last_unpaired = Int(sequence[unsafe_offset = end - 1])
    return initiation + tables[
        unsafe_offset = MISMATCH_OFFSET + pair * 16 + first_unpaired * 4 + last_unpaired
    ]


@always_inline
def interior_energy(
    sequence: Pointer[Scalar[SYMBOL_DTYPE], _],
    tables: Pointer[Scalar[ENERGY_DTYPE], _],
    start: Int,
    end: Int,
    inner_start: Int,
    inner_end: Int,
) -> Int32:
    """The loop between an outer pair and the pair nested directly inside it.

    Nothing unpaired on either side is a stack, nothing on one side is a bulge, and anything else
    is an internal loop carrying Ninio's asymmetry correction.
    """
    var outer = pair_of(tables, Int(sequence[unsafe_offset=start]), Int(sequence[unsafe_offset=end]))
    var inner = pair_of(
        tables, Int(sequence[unsafe_offset=inner_start]), Int(sequence[unsafe_offset=inner_end])
    )
    if not pairs(outer) or not pairs(inner):
        return FORBIDDEN
    var left = inner_start - start - 1
    var right = end - inner_end - 1
    if left + right > MAX_LOOP:
        return FORBIDDEN
    if left == 0 and right == 0:
        return tables[unsafe_offset = STACK_OFFSET + outer * PAIR_TYPES + inner]
    var size = left + right
    if left == 0 or right == 0:
        if size == 1:
            # A single-base bulge keeps the helix stacked across it.
            return (
                tables[unsafe_offset = BULGE_OFFSET + size]
                + tables[unsafe_offset = STACK_OFFSET + outer * PAIR_TYPES + inner]
            )
        return (
            tables[unsafe_offset = BULGE_OFFSET + size]
            + terminal_penalty(outer)
            + terminal_penalty(inner)
        )
    var asymmetry = left - right if left > right else right - left
    var correction = min(Int32(asymmetry) * NINIO_PER_ASYMMETRY, NINIO_CAP)
    var reversed_inner = pair_of(
        tables, Int(sequence[unsafe_offset=inner_end]), Int(sequence[unsafe_offset=inner_start])
    )
    """
    The loop sees the inner helix from outside, so that pair is read reversed, the same way the dangle tables are
    indexed.
    """
    var outer_mismatch = tables[
        unsafe_offset = MISMATCH_INTERNAL_OFFSET
        + outer * 16
        + Int(sequence[unsafe_offset = start + 1]) * 4
        + Int(sequence[unsafe_offset = end - 1])
    ]
    var inner_mismatch = tables[
        unsafe_offset = MISMATCH_INTERNAL_OFFSET
        + reversed_inner * 16
        + Int(sequence[unsafe_offset = inner_end + 1]) * 4
        + Int(sequence[unsafe_offset = inner_start - 1])
    ]
    return (
        tables[unsafe_offset = INTERNAL_OFFSET + size] + correction + outer_mismatch + inner_mismatch
    )


@always_inline
def cell_index(start: Int, span: Int, length: Int) -> Int:
    """Row-major over `(start, span)`, with one column of slack so `span = length` fits."""
    return start * (length + 2) + span


@always_inline
def interior_end_floor(start: Int, end: Int, inner_start: Int) -> Int:
    """The earliest inner end that keeps the loop within `MAX_LOOP` unpaired bases."""
    var left = inner_start - start - 1
    var floor = end - 1 - (MAX_LOOP - left)
    return max(floor, inner_start + MIN_HAIRPIN + 1)

# endregion Energy Model

# region Serial Reference


@fieldwise_init
struct FoldTables(Copyable, Movable):
    """The three tables one fold fills, flat and indexed by `cell_index`."""

    var paired: List[Scalar[ENERGY_DTYPE]]
    """Best energy of a span whose two ends pair with each other."""
    var multiloop: List[Scalar[ENERGY_DTYPE]]
    """Best energy of a span sitting inside a multibranched loop, holding at least one helix."""
    var external: List[Scalar[ENERGY_DTYPE]]
    """Best energy of a span with nothing enclosing it, where unpaired bases are free."""


@always_inline
def paired_cell(
    sequence: Pointer[Scalar[SYMBOL_DTYPE], _],
    tables: Pointer[Scalar[ENERGY_DTYPE], _],
    paired: Pointer[Scalar[ENERGY_DTYPE], _],
    multiloop: Pointer[Scalar[ENERGY_DTYPE], _],
    length: Int,
    start: Int,
    span: Int,
) -> Int32:
    """The hairpin and interior-loop cases of one `paired` cell, and optionally its multiloop.

    The multiloop closure is a linear scan over split points, which the device gives to its whole
    block; `interior_only` lets that caller take the bounded interior scan alone.
    """
    var end = start + span - 1
    var pair = pair_of(tables, Int(sequence[unsafe_offset=start]), Int(sequence[unsafe_offset=end]))
    if not pairs(pair) or span < MIN_HAIRPIN + 2:
        return FORBIDDEN

    var best = hairpin_energy(sequence, tables, start, end)
    for inner_start in range(start + 1, end):
        if inner_start - start - 1 > MAX_LOOP:
            break
        for inner_end in range(interior_end_floor(start, end, inner_start), end):
            var inner_span = inner_end - inner_start + 1
            if inner_span < MIN_HAIRPIN + 2:
                continue
            var nested = paired[unsafe_offset = cell_index(inner_start, inner_span, length)]
            if nested >= FORBIDDEN:
                continue
            var loop = interior_energy(sequence, tables, start, end, inner_start, inner_end)
            if loop < FORBIDDEN:
                best = min(best, loop + nested)

    var closure = MULTILOOP_OFFSET + MULTILOOP_PER_HELIX + terminal_penalty(pair)
    for split in range(start + 2, end - 1):
        var left = multiloop[unsafe_offset = cell_index(start + 1, split - start - 1, length)]
        var right = multiloop[unsafe_offset = cell_index(split, end - split, length)]
        if left < FORBIDDEN and right < FORBIDDEN:
            best = min(best, left + right + closure)
    return best


@always_inline
def multiloop_seed(
    sequence: Pointer[Scalar[SYMBOL_DTYPE], _],
    tables: Pointer[Scalar[ENERGY_DTYPE], _],
    paired: Pointer[Scalar[ENERGY_DTYPE], _],
    multiloop: Pointer[Scalar[ENERGY_DTYPE], _],
    length: Int,
    start: Int,
    span: Int,
) -> Int32:
    """A `multiloop` cell before its split scan: this span closing a helix, or trimmed by a base.

    The split scan stays with the caller because its shape is a serial loop on the host and a
    strided block reduction on the device.
    """
    var end = start + span - 1
    var here = cell_index(start, span, length)
    var best = FORBIDDEN
    var closed = paired[unsafe_offset=here]
    if closed < FORBIDDEN:
        var pair = pair_of(tables, Int(sequence[unsafe_offset=start]), Int(sequence[unsafe_offset=end]))
        best = closed + MULTILOOP_PER_HELIX + terminal_penalty(pair)
        best += dangle_energy(sequence, tables, start, end, length)
    if span > 1:
        var left_trimmed = multiloop[unsafe_offset = cell_index(start + 1, span - 1, length)]
        if left_trimmed < FORBIDDEN:
            best = min(best, left_trimmed + MULTILOOP_PER_UNPAIRED)
        var right_trimmed = multiloop[unsafe_offset = cell_index(start, span - 1, length)]
        if right_trimmed < FORBIDDEN:
            best = min(best, right_trimmed + MULTILOOP_PER_UNPAIRED)
    return best


@always_inline
def external_seed(
    sequence: Pointer[Scalar[SYMBOL_DTYPE], _],
    tables: Pointer[Scalar[ENERGY_DTYPE], _],
    paired: Pointer[Scalar[ENERGY_DTYPE], _],
    external: Pointer[Scalar[ENERGY_DTYPE], _],
    length: Int,
    start: Int,
    span: Int,
) -> Int32:
    """An `external` cell before its split scan, where unpaired bases are free."""
    var end = start + span - 1
    var here = cell_index(start, span, length)
    var best = Int32(0)
    if span > 1:
        best = min(
            external[unsafe_offset = cell_index(start + 1, span - 1, length)],
            external[unsafe_offset = cell_index(start, span - 1, length)],
        )
    var closed = paired[unsafe_offset=here]
    if closed < FORBIDDEN:
        var pair = pair_of(tables, Int(sequence[unsafe_offset=start]), Int(sequence[unsafe_offset=end]))
        var placed = closed + terminal_penalty(pair)
        placed += dangle_energy(sequence, tables, start, end, length)
        best = min(best, placed)
    return best


def serial_fold_tables(
    sequence: ImmSpan[Scalar[SYMBOL_DTYPE], _], tables: ImmSpan[Scalar[ENERGY_DTYPE], _]
) raises -> FoldTables:
    """Fills the three tables on the host, in increasing span and in dependency order inside one."""
    var length = len(sequence)
    var cells = (length + 1) * (length + 2)
    var paired = List[Scalar[ENERGY_DTYPE]](length=cells, fill=FORBIDDEN)
    var multiloop = List[Scalar[ENERGY_DTYPE]](length=cells, fill=FORBIDDEN)
    var external = List[Scalar[ENERGY_DTYPE]](length=cells, fill=Scalar[ENERGY_DTYPE](0))
    var sequence_pointer = sequence.unsafe_ptr()
    var tables_pointer = tables.unsafe_ptr()
    var paired_pointer = paired.unsafe_ptr()
    var multiloop_pointer = multiloop.unsafe_ptr()
    var external_pointer = external.unsafe_ptr()

    for span in range(1, length + 1):
        for start in range(0, length - span + 1):
            var here = cell_index(start, span, length)
            paired[here] = paired_cell(
                sequence_pointer, tables_pointer, paired_pointer, multiloop_pointer,
                length, start, span,
            )

            var best = multiloop_seed(
                sequence_pointer, tables_pointer, paired_pointer, multiloop_pointer, length, start, span
            )
            if span > 1:
                for split in range(1, span):
                    var left = multiloop[cell_index(start, split, length)]
                    var right = multiloop[cell_index(start + split, span - split, length)]
                    if left < FORBIDDEN and right < FORBIDDEN:
                        best = min(best, left + right)
            multiloop[here] = best

            var outside = external_seed(
                sequence_pointer, tables_pointer, paired_pointer, external_pointer, length, start, span
            )
            for split in range(1, span):
                outside = min(
                    outside,
                    external[cell_index(start, split, length)]
                    + external[cell_index(start + split, span - split, length)],
                )
            external[here] = outside

    return FoldTables(paired^, multiloop^, external^)

# endregion Serial Reference

# region GPU Sweep

comptime INTERIOR_COMBINATIONS = (MAX_LOOP + 1) * (MAX_LOOP + 1)
"""
The interior-loop scan is a triangle of unpaired counts on each side; flattening it to a square and discarding the
corner keeps the thread mapping a division rather than a search.
"""


@always_inline
def block_min(
    reduction: Pointer[Scalar[ENERGY_DTYPE], MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    best: Int32,
) -> Int32:
    """Block-wide minimum, left in every thread so the caller needs no second barrier."""
    reduction[unsafe_offset = Int(thread_idx.x)] = best
    barrier()
    var span = THREADS_PER_BLOCK // 2
    while span > 0:
        if Int(thread_idx.x) < span:
            reduction[unsafe_offset = Int(thread_idx.x)] = min(
                reduction[unsafe_offset = Int(thread_idx.x)],
                reduction[unsafe_offset = Int(thread_idx.x) + span],
            )
        barrier()
        span //= 2
    return reduction[unsafe_offset=0]


def paired_kernel(
    sequence: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    tables: Pointer[Scalar[ENERGY_DTYPE], MutAnyOrigin],
    paired: Pointer[Scalar[ENERGY_DTYPE], MutAnyOrigin],
    multiloop: Pointer[Scalar[ENERGY_DTYPE], MutAnyOrigin],
    length_in: Int32,
    span_in: Int32,
):
    """One block per `paired` cell of the span, threads splitting both of its scans."""
    var length = Int(length_in)
    var span = Int(span_in)
    var start = Int(block_idx.x)
    if start + span > length:
        return
    var end = start + span - 1
    var here = cell_index(start, span, length)

    var reduction = stack_allocation[
        THREADS_PER_BLOCK, Scalar[ENERGY_DTYPE], address_space = AddressSpace.SHARED
    ]()
    var pair = pair_of(tables, Int(sequence[unsafe_offset=start]), Int(sequence[unsafe_offset=end]))
    if not pairs(pair) or span < MIN_HAIRPIN + 2:
        if thread_idx.x == 0:
            paired[unsafe_offset=here] = FORBIDDEN
        return

    var best = FORBIDDEN
    if thread_idx.x == 0:
        best = hairpin_energy(sequence, tables, start, end)

    for combined in range(Int(thread_idx.x), INTERIOR_COMBINATIONS, THREADS_PER_BLOCK):
        var left = combined // (MAX_LOOP + 1)
        var right = combined % (MAX_LOOP + 1)
        if left + right > MAX_LOOP:
            continue
        var inner_start = start + 1 + left
        var inner_end = end - 1 - right
        var inner_span = inner_end - inner_start + 1
        if inner_span < MIN_HAIRPIN + 2 or inner_start >= inner_end:
            continue
        var nested = paired[unsafe_offset = cell_index(inner_start, inner_span, length)]
        if nested >= FORBIDDEN:
            continue
        var loop = interior_energy(sequence, tables, start, end, inner_start, inner_end)
        if loop < FORBIDDEN:
            best = min(best, loop + nested)

    var closure = MULTILOOP_OFFSET + MULTILOOP_PER_HELIX + terminal_penalty(pair)
    for split in range(start + 2 + Int(thread_idx.x), end - 1, THREADS_PER_BLOCK):
        var left = multiloop[unsafe_offset = cell_index(start + 1, split - start - 1, length)]
        var right = multiloop[unsafe_offset = cell_index(split, end - split, length)]
        if left < FORBIDDEN and right < FORBIDDEN:
            best = min(best, left + right + closure)

    var answer = block_min(reduction, best)
    if thread_idx.x == 0:
        paired[unsafe_offset=here] = answer


def multiloop_kernel(
    sequence: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    tables: Pointer[Scalar[ENERGY_DTYPE], MutAnyOrigin],
    paired: Pointer[Scalar[ENERGY_DTYPE], MutAnyOrigin],
    multiloop: Pointer[Scalar[ENERGY_DTYPE], MutAnyOrigin],
    length_in: Int32,
    span_in: Int32,
):
    """One block per `multiloop` cell, which reads `paired` at the same span."""
    var length = Int(length_in)
    var span = Int(span_in)
    var start = Int(block_idx.x)
    if start + span > length:
        return
    var end = start + span - 1
    var here = cell_index(start, span, length)

    var reduction = stack_allocation[
        THREADS_PER_BLOCK, Scalar[ENERGY_DTYPE], address_space = AddressSpace.SHARED
    ]()
    var best = FORBIDDEN
    if thread_idx.x == 0:
        best = multiloop_seed(sequence, tables, paired, multiloop, length, start, span)

    for split in range(1 + Int(thread_idx.x), span, THREADS_PER_BLOCK):
        var left = multiloop[unsafe_offset = cell_index(start, split, length)]
        var right = multiloop[unsafe_offset = cell_index(start + split, span - split, length)]
        if left < FORBIDDEN and right < FORBIDDEN:
            best = min(best, left + right)

    var answer = block_min(reduction, best)
    if thread_idx.x == 0:
        multiloop[unsafe_offset=here] = answer


def external_kernel(
    sequence: Pointer[Scalar[SYMBOL_DTYPE], MutAnyOrigin],
    tables: Pointer[Scalar[ENERGY_DTYPE], MutAnyOrigin],
    paired: Pointer[Scalar[ENERGY_DTYPE], MutAnyOrigin],
    external: Pointer[Scalar[ENERGY_DTYPE], MutAnyOrigin],
    length_in: Int32,
    span_in: Int32,
):
    """One block per `external` cell, where unpaired bases are free."""
    var length = Int(length_in)
    var span = Int(span_in)
    var start = Int(block_idx.x)
    if start + span > length:
        return
    var end = start + span - 1
    var here = cell_index(start, span, length)

    var reduction = stack_allocation[
        THREADS_PER_BLOCK, Scalar[ENERGY_DTYPE], address_space = AddressSpace.SHARED
    ]()
    var best = FORBIDDEN
    if thread_idx.x == 0:
        best = external_seed(sequence, tables, paired, external, length, start, span)

    for split in range(1 + Int(thread_idx.x), span, THREADS_PER_BLOCK):
        best = min(
            best,
            external[unsafe_offset = cell_index(start, split, length)]
            + external[unsafe_offset = cell_index(start + split, span - split, length)],
        )

    var answer = block_min(reduction, best)
    if thread_idx.x == 0:
        external[unsafe_offset=here] = answer


def device_fold_tables(
    ctx: DeviceContext,
    sequence: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    tables: ImmSpan[Scalar[ENERGY_DTYPE], _],
) raises -> FoldTables:
    """Sweeps the three tables on the device, three launches per span.

    `multiloop` and `external` read `paired` at their own span, so they cannot share a launch
    with it; every other dependency is on a strictly shorter span and needs no ordering.
    """
    var length = len(sequence)
    var cells = (length + 1) * (length + 2)

    var sequence_buffer = upload[SYMBOL_DTYPE](ctx, sequence)
    var tables_buffer = upload[ENERGY_DTYPE](ctx, tables)
    var paired_buffer = ctx.enqueue_create_buffer[ENERGY_DTYPE](cells)
    var multiloop_buffer = ctx.enqueue_create_buffer[ENERGY_DTYPE](cells)
    var external_buffer = zeroed[ENERGY_DTYPE](ctx, cells)
    ctx.enqueue_memset(paired_buffer, FORBIDDEN)
    ctx.enqueue_memset(multiloop_buffer, FORBIDDEN)

    for span in range(1, length + 1):
        ctx.enqueue_function[paired_kernel](
            sequence_buffer.unsafe_ptr(), tables_buffer.unsafe_ptr(),
            paired_buffer.unsafe_ptr(), multiloop_buffer.unsafe_ptr(),
            Int32(length), Int32(span),
            grid_dim=length, block_dim=THREADS_PER_BLOCK,
        )
        ctx.enqueue_function[multiloop_kernel](
            sequence_buffer.unsafe_ptr(), tables_buffer.unsafe_ptr(),
            paired_buffer.unsafe_ptr(), multiloop_buffer.unsafe_ptr(),
            Int32(length), Int32(span),
            grid_dim=length, block_dim=THREADS_PER_BLOCK,
        )
        ctx.enqueue_function[external_kernel](
            sequence_buffer.unsafe_ptr(), tables_buffer.unsafe_ptr(),
            paired_buffer.unsafe_ptr(), external_buffer.unsafe_ptr(),
            Int32(length), Int32(span),
            grid_dim=length, block_dim=THREADS_PER_BLOCK,
        )
    ctx.synchronize()

    var paired = List[Scalar[ENERGY_DTYPE]](length=cells, fill=FORBIDDEN)
    var multiloop = List[Scalar[ENERGY_DTYPE]](length=cells, fill=FORBIDDEN)
    var external = List[Scalar[ENERGY_DTYPE]](length=cells, fill=Scalar[ENERGY_DTYPE](0))
    ctx.enqueue_copy(paired.unsafe_ptr(), paired_buffer)
    ctx.enqueue_copy(multiloop.unsafe_ptr(), multiloop_buffer)
    ctx.enqueue_copy(external.unsafe_ptr(), external_buffer)
    ctx.synchronize()
    return FoldTables(paired^, multiloop^, external^)

# endregion GPU Sweep

# region Traceback

@fieldwise_init
struct FoldTable(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which of the three tables a pending traceback item belongs to."""

    var identifier: UInt8
    """Which table this names."""
    comptime PAIRED = Self(0)
    """The span's two ends form a pair."""
    comptime MULTILOOP = Self(1)
    """The span lies inside a multibranched loop."""
    comptime EXTERNAL = Self(2)
    """The span is not enclosed by any pair."""


@fieldwise_init
struct FoldStep(ImplicitlyCopyable, TrivialRegisterPassable):
    """A span still to be decomposed, and which table it was reached through."""

    var table: FoldTable
    """Which table this span was reached through, and so which cases apply to it."""
    var start: Int32
    """First position of the span."""
    var span: Int32
    """How many positions it covers."""


@always_inline
def winning_interior(
    sequence: Pointer[Scalar[SYMBOL_DTYPE], _],
    tables: Pointer[Scalar[ENERGY_DTYPE], _],
    paired: Pointer[Scalar[ENERGY_DTYPE], _],
    length: Int,
    start: Int,
    span: Int,
) -> FoldStep:
    """The nested pair whose interior loop reproduces a `paired` cell, or a negative span for none."""
    var end = start + span - 1
    var stored = paired[unsafe_offset = cell_index(start, span, length)]
    for inner_start in range(start + 1, end):
        if inner_start - start - 1 > MAX_LOOP:
            break
        for inner_end in range(interior_end_floor(start, end, inner_start), end):
            var inner_span = inner_end - inner_start + 1
            if inner_span < MIN_HAIRPIN + 2:
                continue
            var nested = paired[unsafe_offset = cell_index(inner_start, inner_span, length)]
            if nested >= FORBIDDEN:
                continue
            if interior_energy(sequence, tables, start, end, inner_start, inner_end) + nested == stored:
                return FoldStep(FoldTable.PAIRED, Int32(inner_start), Int32(inner_span))
    return FoldStep(FoldTable.PAIRED, 0, -1)


def fold_traceback(
    sequence: ImmSpan[Scalar[SYMBOL_DTYPE], _],
    tables: ImmSpan[Scalar[ENERGY_DTYPE], _],
    folded: FoldTables,
) raises -> String:
    """Walks the tables into a dot-bracket structure.

    Nothing is recorded during the fill: the tables are resident anyway, so testing the cases in
    the order the fill tried them costs less than a parallel array of decisions.
    """
    var length = len(sequence)
    var sequence_pointer = sequence.unsafe_ptr()
    var tables_pointer = tables.unsafe_ptr()
    var paired = folded.paired.unsafe_ptr()
    var multiloop = folded.multiloop.unsafe_ptr()
    var external = folded.external.unsafe_ptr()

    var structure = List[Byte](length=length, fill=UNPAIRED_BYTE)
    var work = List[FoldStep]()
    work.append(FoldStep(FoldTable.EXTERNAL, 0, Int32(length)))

    while len(work) > 0:
        var item = work.pop()
        var start = Int(item.start)
        var span = Int(item.span)
        if span <= 0:
            continue
        var end = start + span - 1
        var here = cell_index(start, span, length)
        var pair = pair_of(
            tables_pointer, Int(sequence[start]), Int(sequence[end])
        )

        if item.table == FoldTable.EXTERNAL:
            var stored = external[unsafe_offset=here]
            if span > 1 and external[unsafe_offset = cell_index(start + 1, span - 1, length)] == stored:
                work.append(FoldStep(FoldTable.EXTERNAL, Int32(start + 1), Int32(span - 1)))
                continue
            if span > 1 and external[unsafe_offset = cell_index(start, span - 1, length)] == stored:
                work.append(FoldStep(FoldTable.EXTERNAL, Int32(start), Int32(span - 1)))
                continue
            var closed = paired[unsafe_offset=here]
            var placed = (
                closed
                + terminal_penalty(pair)
                + dangle_energy(sequence_pointer, tables_pointer, start, end, length)
            )
            if closed < FORBIDDEN and placed == stored:
                work.append(FoldStep(FoldTable.PAIRED, Int32(start), Int32(span)))
                continue
            for split in range(1, span):
                if (
                    external[unsafe_offset = cell_index(start, split, length)]
                    + external[unsafe_offset = cell_index(start + split, span - split, length)]
                    == stored
                ):
                    work.append(FoldStep(FoldTable.EXTERNAL, Int32(start), Int32(split)))
                    work.append(FoldStep(FoldTable.EXTERNAL, Int32(start + split), Int32(span - split)))
                    break
            continue

        if item.table == FoldTable.MULTILOOP:
            var stored = multiloop[unsafe_offset=here]
            var closed = paired[unsafe_offset=here]
            var placed = (
                closed
                + MULTILOOP_PER_HELIX
                + terminal_penalty(pair)
                + dangle_energy(sequence_pointer, tables_pointer, start, end, length)
            )
            if closed < FORBIDDEN and placed == stored:
                work.append(FoldStep(FoldTable.PAIRED, Int32(start), Int32(span)))
                continue
            if span > 1 and (
                multiloop[unsafe_offset = cell_index(start + 1, span - 1, length)]
                + MULTILOOP_PER_UNPAIRED
                == stored
            ):
                work.append(FoldStep(FoldTable.MULTILOOP, Int32(start + 1), Int32(span - 1)))
                continue
            if span > 1 and (
                multiloop[unsafe_offset = cell_index(start, span - 1, length)]
                + MULTILOOP_PER_UNPAIRED
                == stored
            ):
                work.append(FoldStep(FoldTable.MULTILOOP, Int32(start), Int32(span - 1)))
                continue
            for split in range(1, span):
                var left = multiloop[unsafe_offset = cell_index(start, split, length)]
                var right = multiloop[unsafe_offset = cell_index(start + split, span - split, length)]
                if left < FORBIDDEN and right < FORBIDDEN and left + right == stored:
                    work.append(FoldStep(FoldTable.MULTILOOP, Int32(start), Int32(split)))
                    work.append(FoldStep(FoldTable.MULTILOOP, Int32(start + split), Int32(span - split)))
                    break
            continue

        var stored = paired[unsafe_offset=here]
        structure[start] = OPEN_BYTE
        structure[end] = CLOSE_BYTE
        if hairpin_energy(sequence_pointer, tables_pointer, start, end) == stored:
            continue
        var nested_step = winning_interior(sequence_pointer, tables_pointer, paired, length, start, span)
        if nested_step.span > 0:
            work.append(nested_step)
            continue
        var closure = MULTILOOP_OFFSET + MULTILOOP_PER_HELIX + terminal_penalty(pair)
        for split in range(start + 2, end - 1):
            var left = multiloop[unsafe_offset = cell_index(start + 1, split - start - 1, length)]
            var right = multiloop[unsafe_offset = cell_index(split, end - split, length)]
            if left < FORBIDDEN and right < FORBIDDEN and left + right + closure == stored:
                work.append(FoldStep(FoldTable.MULTILOOP, Int32(start + 1), Int32(split - start - 1)))
                work.append(FoldStep(FoldTable.MULTILOOP, Int32(split), Int32(end - split)))
                break

    return String(unsafe_from_utf8=structure)

# endregion Traceback

# region Interface


@fieldwise_init
struct FoldResult(Copyable, Movable):
    """A dot-bracket structure and the free energy of the decomposition that produced it."""

    var structure: String
    """Dot-bracket over the sequence, one character per position."""
    var decikcal: Int32
    """Free energy in integer decikilocalories per mole."""


def serial_fold(sequence_text: String) raises -> FoldResult:
    """Folds one RNA sequence on the host."""
    var alphabet = String(DEFAULT_RNA_ALPHABET)
    var sequence = translate(sequence_text, alphabet)
    var tables = packed_turner_tables()
    var folded = serial_fold_tables(Span(sequence), Span(tables))
    var energy = folded.external[cell_index(0, len(sequence), len(sequence))]
    var structure = fold_traceback(Span(sequence), Span(tables), folded)
    return FoldResult(structure^, energy)


def device_fold(ctx: DeviceContext, sequence_text: String) raises -> FoldResult:
    """Folds one RNA sequence with the sweep on the device and the walk on the host.

    Traceback stays on the host because it is a serial walk over tables the device already filled,
    and they are only quadratic, so bringing them back is cheap.
    """
    var alphabet = String(DEFAULT_RNA_ALPHABET)
    var sequence = translate(sequence_text, alphabet)
    var tables = packed_turner_tables()
    var folded = device_fold_tables(ctx, Span(sequence), Span(tables))
    var energy = folded.external[cell_index(0, len(sequence), len(sequence))]
    var structure = fold_traceback(Span(sequence), Span(tables), folded)
    return FoldResult(structure^, energy)

# endregion Interface
