"""
Zuker minimum free energy folding for CPU and GPU, exact and with traceback.

Four tables are filled: `paired` for a subsequence whose ends pair, `multiloop` for one inside a
multibranched loop, `closable` for one holding two branches or more, and `exterior` for a suffix
nothing encloses. The first three read `paired` at their own window, so each window takes two
launches; `exterior` holds one live cell per window and follows in a single descending pass.

Everything is indexed by `(start, window)`, so a bifurcation reads strictly smaller lengths and
every cell of a window is independent. Memory is $O(n^2)$ and time is $O(n^3)$, the interior-loop
term bounded by capping a loop at `MAX_LOOP` unpaired bases as this recurrence always is.

Energies are integer decikilocalories per mole, so a fold is reproducible bit for bit rather than
depending on floating-point association order.

The energy model is the subset described in `turner.mojo`. It reproduces RNAstructure's `efn2`
exactly for structures built from stacks and hairpins, and differs where dangling ends, coaxial
stacking, tetraloop bonuses or the special small-internal-loop tables would apply.

The recurrence, the tie-breaking and the traceback are transcribed from `folding.py`, the oracle.
"""

from std.gpu import block_idx, thread_idx
from std.gpu.primitives.warp import WARP_SIZE, min as warp_min
from std.memory import stack_allocation
from std.memory.pointer import AddressSpace

from max.gpu import barrier
from std.math import log

from max.gpu.host import DeviceContext

from errors import AffineGapsError, ErrorKind
from common import (
    CLOSE_BYTE,
    DEFAULT_RNA_ALPHABET,
    OPEN_BYTE,
    SymbolDType,
    THREADS_PER_BLOCK,
    UNPAIRED_BYTE,
    translate,
    upload,
    zeroed,
)
from turner import (
    BULGE_INITIATION,
    DANGLE_AFTER,
    DANGLE_BEFORE,
    EnergyDType,
    FORBIDDEN,
    HAIRPIN_INITIATION,
    HEXALOOP_ENERGIES,
    HEXALOOP_KEYS,
    INTERNAL_INITIATION,
    LOOP_LIMIT,
    MULTILOOP_OFFSET,
    MULTILOOP_PER_HELIX,
    MULTILOOP_PER_UNPAIRED,
    NINIO_CAP,
    NINIO_PER_ASYMMETRY,
    PAIR_INDEX,
    PAIR_TYPES,
    STACK,
    TERMINAL_AU,
    TERMINAL_MISMATCH_HAIRPIN,
    TERMINAL_MISMATCH_INTERNAL,
    TETRALOOP_ENERGIES,
    TETRALOOP_KEYS,
    TRILOOP_ENERGIES,
    TRILOOP_KEYS,
)

# region Energy Model

comptime MAX_LOOP = LOOP_LIMIT
comptime MIN_TURN = 3
"""Fewest bases any pair must enclose, which is what the backbone can turn in."""
comptime MIN_CLOSING_REACH = MIN_TURN + 1
"""The turn plus the partner past it: the shortest head-to-partner distance."""
comptime MIN_PAIRED_WINDOW = MIN_CLOSING_REACH + 1
"""The reach plus the head: the shortest window `paired` can be finite on."""
comptime PositionDType = DType.int32
comptime RNA_ALPHABET_SIZE = 4
"""Letters the folding model knows, which is what sizes the per-letter partner runs."""
comptime WARPS_PER_BLOCK = THREADS_PER_BLOCK // WARP_SIZE

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
comptime ENERGY_MODEL_LENGTH = HEXALOOP_ENERGY_OFFSET + HEXALOOP_COUNT


def packed_energy_model() -> List[Scalar[EnergyDType]]:
    """Every table this recurrence reads, concatenated at the offsets above."""
    var stack = materialize[STACK]()
    var hairpin_mismatch = materialize[TERMINAL_MISMATCH_HAIRPIN]()
    var internal_mismatch = materialize[TERMINAL_MISMATCH_INTERNAL]()
    var after = materialize[DANGLE_AFTER]()
    var before = materialize[DANGLE_BEFORE]()
    var hairpin = materialize[HAIRPIN_INITIATION]()
    var bulge = materialize[BULGE_INITIATION]()
    var internal = materialize[INTERNAL_INITIATION]()
    var is_pair = materialize[PAIR_INDEX]()
    var triloop_keys = materialize[TRILOOP_KEYS]()
    var triloop_energies = materialize[TRILOOP_ENERGIES]()
    var tetraloop_keys = materialize[TETRALOOP_KEYS]()
    var tetraloop_energies = materialize[TETRALOOP_ENERGIES]()
    var hexaloop_keys = materialize[HEXALOOP_KEYS]()
    var hexaloop_energies = materialize[HEXALOOP_ENERGIES]()

    # Every entry below is written, so filling first would be a memset immediately overwritten.
    var packed = List[Scalar[EnergyDType]](unsafe_uninit_length=ENERGY_MODEL_LENGTH)
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
        packed[PAIR_INDEX_OFFSET + index] = is_pair[index]
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
def is_pair(index: Int) -> Bool:
    """Whether a `pair_of` answer names a real pair."""
    return index != NO_PAIR


@always_inline
def pair_of(energy_model: Pointer[Scalar[EnergyDType], _], first: Int, second: Int) -> Int:
    """Which of the six pair types two bases form, or a negative value for none."""
    return Int(energy_model[unsafe_offset=PAIR_INDEX_OFFSET + first * 4 + second])


@always_inline
def helix_end_penalty(pair: Int) -> Int32:
    """The helix-end penalty, charged to everything but a Watson-Crick CG or GC pair."""
    if pair == PAIR_CG or pair == PAIR_GC:
        return Int32(0)
    return TERMINAL_AU


@always_inline
def packed_key(sequence: Pointer[Scalar[SymbolDType], _], start: Int, end: Int) -> Int32:
    """The closing pair and loop bases packed base-four, most significant first."""
    var key = Int32(0)
    for index in range(start, end + 1):
        key = key * 4 + Int32(Int(sequence[unsafe_offset=index]))
    return key


@always_inline
def special_hairpin(
    sequence: Pointer[Scalar[SymbolDType], _],
    energy_model: Pointer[Scalar[EnergyDType], _],
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
        if energy_model[unsafe_offset=keys + index] == key:
            return energy_model[unsafe_offset=energies + index]
    return FORBIDDEN


@always_inline
def dangle_energy(
    sequence: Pointer[Scalar[SymbolDType], _],
    energy_model: Pointer[Scalar[EnergyDType], _],
    start: Int,
    end: Int,
    sequence_length: Int,
) -> Int32:
    """Both dangles on a helix placed in an exterior loop or a multiloop.

    The pair is read from outside the helix, and both neighbours are charged whenever they exist,
    which is the simple treatment that needs no extra states in the recurrence.
    """
    var outward = pair_of(energy_model, Int(sequence[unsafe_offset=end]), Int(sequence[unsafe_offset=start]))
    if not is_pair(outward):
        return Int32(0)
    var total = Int32(0)
    if start > 0:
        total += energy_model[unsafe_offset=DANGLE_BEFORE_OFFSET + outward * 4 + Int(sequence[unsafe_offset=start - 1])]
    if end < sequence_length - 1:
        total += energy_model[unsafe_offset=DANGLE_AFTER_OFFSET + outward * 4 + Int(sequence[unsafe_offset=end + 1])]
    return total


@always_inline
def hairpin_energy(
    sequence: Pointer[Scalar[SymbolDType], _],
    energy_model: Pointer[Scalar[EnergyDType], _],
    start: Int,
    end: Int,
) -> Int32:
    """A hairpin closed by `(start, end)`, with everything between it unpaired."""
    var size = end - start - 1
    if size < MIN_TURN:
        return FORBIDDEN
    var pair = pair_of(energy_model, Int(sequence[unsafe_offset=start]), Int(sequence[unsafe_offset=end]))
    if not is_pair(pair):
        return FORBIDDEN
    var tabulated = special_hairpin(sequence, energy_model, start, end, size)
    if tabulated < FORBIDDEN:
        # Tabulated with the helix end factored out, as the mismatch tables are.
        return tabulated + helix_end_penalty(pair)
    var initiation = energy_model[unsafe_offset=HAIRPIN_OFFSET + min(size, LOOP_LIMIT)]
    if size > LOOP_LIMIT:
        # Beyond the tabulated sizes the model extrapolates by polymer theory, which needs a logarithm; capping
        # instead would make long loops artificially cheap. Metal has no `f64`, and single precision rounds to the
        # same decikcal for every size a hairpin can reach.
        var ratio = Float32(size) / Float32(LOOP_LIMIT)
        initiation += Int32(round(Float32(10.79) * log(ratio)))
    if size == MIN_TURN:
        return initiation + helix_end_penalty(pair)
    var first_unpaired = Int(sequence[unsafe_offset=start + 1])
    var last_unpaired = Int(sequence[unsafe_offset=end - 1])
    return (
        initiation
        + helix_end_penalty(pair)
        + energy_model[unsafe_offset=MISMATCH_OFFSET + pair * 16 + first_unpaired * 4 + last_unpaired]
    )


@fieldwise_init
struct ClosingPair(ImplicitlyCopyable, TrivialRegisterPassable):
    """What a cell's outer pair contributes to every interior loop it can close.

    None of it depends on the nested pair, so a cell reads it once instead of once per candidate,
    and the interior scan is the hottest loop in the whole recurrence.
    """

    var pair: Int
    """Which of the six pair types the two closing bases form, or `NO_PAIR`."""
    var mismatch: Int32
    """The internal-loop mismatch this closing pair reads, fixed by the two bases inside it."""
    var penalty: Int32
    """The helix-end charge this closing pair pays."""


@always_inline
def closing_pair(
    sequence: Pointer[Scalar[SymbolDType], _],
    energy_model: Pointer[Scalar[EnergyDType], _],
    start: Int,
    end: Int,
) -> ClosingPair:
    """Everything one cell's closing pair contributes, hoisted out of the interior scan."""
    var pair = pair_of(energy_model, Int(sequence[unsafe_offset=start]), Int(sequence[unsafe_offset=end]))
    if not is_pair(pair):
        return ClosingPair(pair, 0, 0)
    var mismatch = energy_model[
        unsafe_offset=MISMATCH_INTERNAL_OFFSET
        + pair * 16
        + Int(sequence[unsafe_offset=start + 1]) * 4
        + Int(sequence[unsafe_offset=end - 1])
    ]
    return ClosingPair(pair, mismatch, helix_end_penalty(pair))


@always_inline
def interior_energy(
    sequence: Pointer[Scalar[SymbolDType], _],
    energy_model: Pointer[Scalar[EnergyDType], _],
    closing: ClosingPair,
    start: Int,
    end: Int,
    inner_start: Int,
    inner_end: Int,
) -> Int32:
    """The loop between an outer pair and the pair nested directly inside it.

    Nothing unpaired on either side is a stack, nothing on one side is a bulge, and anything else
    is an internal loop carrying Ninio's asymmetry correction.
    """
    var outer = closing.pair
    var inner = pair_of(energy_model, Int(sequence[unsafe_offset=inner_start]), Int(sequence[unsafe_offset=inner_end]))
    if not is_pair(outer) or not is_pair(inner):
        return FORBIDDEN
    var unpaired_before = inner_start - start - 1
    var unpaired_after = end - inner_end - 1
    if unpaired_before + unpaired_after > MAX_LOOP:
        return FORBIDDEN
    if unpaired_before == 0 and unpaired_after == 0:
        return energy_model[unsafe_offset=STACK_OFFSET + outer * PAIR_TYPES + inner]
    var size = unpaired_before + unpaired_after
    if unpaired_before == 0 or unpaired_after == 0:
        if size == 1:
            # A single-base bulge keeps the helix stacked across it.
            return (
                energy_model[unsafe_offset=BULGE_OFFSET + size]
                + energy_model[unsafe_offset=STACK_OFFSET + outer * PAIR_TYPES + inner]
            )
        return energy_model[unsafe_offset=BULGE_OFFSET + size] + closing.penalty + helix_end_penalty(inner)
    var asymmetry = (
        unpaired_before - unpaired_after if unpaired_before > unpaired_after else unpaired_after - unpaired_before
    )
    var correction = min(Int32(asymmetry) * NINIO_PER_ASYMMETRY, NINIO_CAP)
    var reversed_inner = pair_of(
        energy_model, Int(sequence[unsafe_offset=inner_end]), Int(sequence[unsafe_offset=inner_start])
    )
    """
    The loop sees the inner helix from outside, so that pair is read reversed, the same way the dangle tables are
    indexed.
    """
    var inner_mismatch = energy_model[
        unsafe_offset=MISMATCH_INTERNAL_OFFSET
        + reversed_inner * 16
        + Int(sequence[unsafe_offset=inner_end + 1]) * 4
        + Int(sequence[unsafe_offset=inner_start - 1])
    ]
    var closure = closing.penalty + helix_end_penalty(reversed_inner)
    return energy_model[unsafe_offset=INTERNAL_OFFSET + size] + correction + closure + closing.mismatch + inner_mismatch


@always_inline
def cell_index(start: Int, window: Int, sequence_length: Int) -> Int:
    """Row-major over `(start, window)`, with one column of slack so `window = sequence_length` fits."""
    return start * (sequence_length + 2) + window


@always_inline
def interior_end_floor(start: Int, end: Int, inner_start: Int) -> Int:
    """The earliest inner end that keeps the loop within `MAX_LOOP` unpaired bases."""
    var unpaired_before = inner_start - start - 1
    var floor = end - 1 - (MAX_LOOP - unpaired_before)
    return max(floor, inner_start + MIN_CLOSING_REACH)


# endregion Energy Model

# region Serial Reference


@fieldwise_init
struct PartnerRuns(Copyable, Movable):
    """Where each letter's possible partners sit in the sequence, one ascending run per letter."""

    var positions: List[Scalar[PositionDType]]
    """Every position that can close a pair, the letters' runs laid end to end."""
    var bounds: List[Scalar[PositionDType]]
    """Letter `c`'s run at its first entry from `t` onwards, held at `c * (sequence_length + 1) + t`."""


@fieldwise_init
struct PartnerRange(ImplicitlyCopyable, TrivialRegisterPassable):
    """Half-open slice of one letter's run, covering the partners one window can reach."""

    var low: Int
    """First entry of the run that lands inside the window."""
    var high: Int
    """One past the run's last entry inside the window."""


def partner_runs(
    sequence: ImmSpan[Scalar[SymbolDType], _], energy_model: ImmSpan[Scalar[EnergyDType], _], alphabet_size: Int
) -> PartnerRuns:
    """Lists, per letter, every position in the sequence that letter can close a pair with."""
    var sequence_length = len(sequence)
    var positions = List[Scalar[PositionDType]]()
    var bounds = List[Scalar[PositionDType]](
        length=alphabet_size * (sequence_length + 1), fill=Scalar[PositionDType](0)
    )
    var energy_model_pointer = energy_model.unsafe_ptr()
    for letter in range(alphabet_size):
        for position in range(sequence_length):
            bounds[letter * (sequence_length + 1) + position] = Scalar[PositionDType](len(positions))
            if is_pair(pair_of(energy_model_pointer, letter, Int(sequence[position]))):
                positions.append(Scalar[PositionDType](position))
        bounds[letter * (sequence_length + 1) + sequence_length] = Scalar[PositionDType](len(positions))
    return PartnerRuns(positions^, bounds^)


@always_inline
def reachable_partners(
    bounds: Pointer[Scalar[PositionDType], _], head: Int, sequence_length: Int, start: Int, window: Int
) -> PartnerRange:
    """The slice of `head`'s run this window can close a helix with.

    The short-window guard is load-bearing rather than an optimization: without it the threshold
    can run past the end of `bounds`.
    """
    if window < MIN_PAIRED_WINDOW:
        return PartnerRange(0, 0)
    var run = head * (sequence_length + 1)
    return PartnerRange(
        Int(bounds[unsafe_offset=run + start + MIN_CLOSING_REACH]),
        Int(bounds[unsafe_offset=run + start + window]),
    )


@always_inline
def branch_energy(
    sequence: Pointer[Scalar[SymbolDType], _],
    energy_model: Pointer[Scalar[EnergyDType], _],
    paired: Pointer[Scalar[EnergyDType], _],
    sequence_length: Int,
    start: Int,
    partner: Int,
) -> Int32:
    """One helix placed as a branch: what it encloses, its helix-end penalty and its dangles.

    A pure function of where the helix starts and ends, never of what encloses it, which is why
    the sweep stores it once per window instead of recomputing it once per candidate. The forbidden
    case returns early, because adding the penalties to the sentinel would drift it below itself.
    """
    var closed = paired[unsafe_offset=cell_index(start, partner - start + 1, sequence_length)]
    if closed >= FORBIDDEN:
        return FORBIDDEN
    var pair = pair_of(energy_model, Int(sequence[unsafe_offset=start]), Int(sequence[unsafe_offset=partner]))
    return closed + helix_end_penalty(pair) + dangle_energy(sequence, energy_model, start, partner, sequence_length)


@fieldwise_init
struct FoldTables(Copyable, Movable):
    """The four tables one fold fills, flat and indexed by `cell_index`."""

    var paired: List[Scalar[EnergyDType]]
    """Best energy of a window whose two ends pair with each other."""
    var multiloop: List[Scalar[EnergyDType]]
    """Best energy of a window inside a multibranched loop, holding at least one branch."""
    var closable: List[Scalar[EnergyDType]]
    """The same, holding at least two, which is what a closing pair may enclose."""
    var exterior: List[Scalar[EnergyDType]]
    """Best energy of a suffix with nothing enclosing it, indexed by its start alone."""


@always_inline
def multiloop_closure(pair: Int) -> Int32:
    """What a pair pays to close a multiloop, over and above what it encloses.

    One transcription serves the host sweep, the device sweep and the traceback, so a closure can
    never be tested against a price the fill did not use.
    """
    return MULTILOOP_OFFSET + MULTILOOP_PER_HELIX + helix_end_penalty(pair)


@always_inline
def paired_cell(
    sequence: Pointer[Scalar[SymbolDType], _],
    energy_model: Pointer[Scalar[EnergyDType], _],
    paired: Pointer[Scalar[EnergyDType], _],
    closable: Pointer[Scalar[EnergyDType], _],
    sequence_length: Int,
    start: Int,
    window: Int,
) -> Int32:
    """Every case of one `paired` cell: the hairpin, the interior loops, and the multiloop closure."""
    var end = start + window - 1
    var pair = pair_of(energy_model, Int(sequence[unsafe_offset=start]), Int(sequence[unsafe_offset=end]))
    if not is_pair(pair) or window < MIN_PAIRED_WINDOW:
        return FORBIDDEN

    var best = hairpin_energy(sequence, energy_model, start, end)
    var closing = closing_pair(sequence, energy_model, start, end)
    for inner_start in range(start + 1, end):
        if inner_start - start - 1 > MAX_LOOP:
            break
        for inner_end in range(interior_end_floor(start, end, inner_start), end):
            var inner_window = inner_end - inner_start + 1
            if inner_window < MIN_PAIRED_WINDOW:
                continue
            var nested = paired[unsafe_offset=cell_index(inner_start, inner_window, sequence_length)]
            if nested >= FORBIDDEN:
                continue
            var loop = interior_energy(sequence, energy_model, closing, start, end, inner_start, inner_end)
            if loop < FORBIDDEN:
                best = min(best, loop + nested)

    var enclosed = closable[unsafe_offset=cell_index(start + 1, window - 2, sequence_length)]
    if enclosed < FORBIDDEN:
        best = min(best, enclosed + multiloop_closure(pair))
    return best


@always_inline
def multiloop_cell(
    sequence: Pointer[Scalar[SymbolDType], _],
    branch_placed: Pointer[Scalar[EnergyDType], _],
    multiloop: Pointer[Scalar[EnergyDType], _],
    positions: Pointer[Scalar[PositionDType], _],
    bounds: Pointer[Scalar[PositionDType], _],
    sequence_length: Int,
    start: Int,
    window: Int,
) -> Int32:
    """One `multiloop` cell, at least one branch: the head unpaired inside the loop, or opening one."""
    var end = start + window - 1
    var best = FORBIDDEN
    if window > 1:
        var trimmed = multiloop[unsafe_offset=cell_index(start + 1, window - 1, sequence_length)]
        if trimmed < FORBIDDEN:
            best = min(best, trimmed + MULTILOOP_PER_UNPAIRED)
    var reach = reachable_partners(bounds, Int(sequence[unsafe_offset=start]), sequence_length, start, window)
    for index in range(reach.high - 1, reach.low - 1, -1):
        var partner = Int(positions[unsafe_offset=index])
        var branch = branch_placed[unsafe_offset=cell_index(start, partner - start + 1, sequence_length)]
        if branch >= FORBIDDEN:
            continue
        branch += MULTILOOP_PER_HELIX
        var tail = end - partner
        # Nothing more pairs after this branch, which also covers a tail of no bases at all.
        best = min(best, branch + MULTILOOP_PER_UNPAIRED * Int32(tail))
        var following = multiloop[unsafe_offset=cell_index(partner + 1, tail, sequence_length)]
        if following < FORBIDDEN:
            best = min(best, branch + following)
    return best


@always_inline
def multiloop_closable_cell(
    sequence: Pointer[Scalar[SymbolDType], _],
    branch_placed: Pointer[Scalar[EnergyDType], _],
    multiloop: Pointer[Scalar[EnergyDType], _],
    closable: Pointer[Scalar[EnergyDType], _],
    positions: Pointer[Scalar[PositionDType], _],
    bounds: Pointer[Scalar[PositionDType], _],
    sequence_length: Int,
    start: Int,
    window: Int,
) -> Int32:
    """One `closable` cell, at least two branches, which is what a closing pair may enclose.

    The only difference from `multiloop_cell` is the absent all-unpaired tail, and that absence is
    the whole mechanism by which Zuker's two-branch rule stays enforced.
    """
    var end = start + window - 1
    var best = FORBIDDEN
    if window > 1:
        var trimmed = closable[unsafe_offset=cell_index(start + 1, window - 1, sequence_length)]
        if trimmed < FORBIDDEN:
            best = min(best, trimmed + MULTILOOP_PER_UNPAIRED)
    var reach = reachable_partners(bounds, Int(sequence[unsafe_offset=start]), sequence_length, start, window)
    for index in range(reach.high - 1, reach.low - 1, -1):
        var partner = Int(positions[unsafe_offset=index])
        var branch = branch_placed[unsafe_offset=cell_index(start, partner - start + 1, sequence_length)]
        if branch >= FORBIDDEN:
            continue
        branch += MULTILOOP_PER_HELIX
        var following = multiloop[unsafe_offset=cell_index(partner + 1, end - partner, sequence_length)]
        if following < FORBIDDEN:
            best = min(best, branch + following)
    return best


@always_inline
def exterior_cell(
    sequence: Pointer[Scalar[SymbolDType], _],
    branch_placed: Pointer[Scalar[EnergyDType], _],
    exterior: Pointer[Scalar[EnergyDType], _],
    positions: Pointer[Scalar[PositionDType], _],
    bounds: Pointer[Scalar[PositionDType], _],
    sequence_length: Int,
    start: Int,
) -> Int32:
    """One `exterior` cell: the head unpaired and free, or the head opening the leftmost helix.

    Both reads preserve the end, so from `exterior[0]` the recursion never leaves the sequence's
    last position and every other cell of a square table is unreachable.
    """
    var best = exterior[unsafe_offset=start + 1]
    var reach = reachable_partners(
        bounds, Int(sequence[unsafe_offset=start]), sequence_length, start, sequence_length - start
    )
    for index in range(reach.high - 1, reach.low - 1, -1):
        var partner = Int(positions[unsafe_offset=index])
        var branch = branch_placed[unsafe_offset=cell_index(start, partner - start + 1, sequence_length)]
        if branch >= FORBIDDEN:
            continue
        best = min(best, branch + exterior[unsafe_offset=partner + 1])
    return best


def serial_fold_tables(
    sequence: ImmSpan[Scalar[SymbolDType], _], energy_model: ImmSpan[Scalar[EnergyDType], _]
) raises -> FoldTables:
    """Fills the square tables in increasing window, then `exterior` in one descending pass."""
    var sequence_length = len(sequence)
    var cells = (sequence_length + 1) * (sequence_length + 2)
    var paired = List[Scalar[EnergyDType]](length=cells, fill=FORBIDDEN)
    var multiloop = List[Scalar[EnergyDType]](length=cells, fill=FORBIDDEN)
    var closable = List[Scalar[EnergyDType]](length=cells, fill=FORBIDDEN)
    var exterior = List[Scalar[EnergyDType]](length=sequence_length + 2, fill=Scalar[EnergyDType](0))
    var branch_placed = List[Scalar[EnergyDType]](length=cells, fill=FORBIDDEN)
    var runs = partner_runs(sequence, energy_model, RNA_ALPHABET_SIZE)
    var sequence_pointer = sequence.unsafe_ptr()
    var energy_model_pointer = energy_model.unsafe_ptr()
    var paired_pointer = paired.unsafe_ptr()
    var multiloop_pointer = multiloop.unsafe_ptr()
    var closable_pointer = closable.unsafe_ptr()
    var exterior_pointer = exterior.unsafe_ptr()
    var branch_pointer = branch_placed.unsafe_ptr()
    var positions = runs.positions.unsafe_ptr()
    var bounds = runs.bounds.unsafe_ptr()

    for window in range(1, sequence_length + 1):
        for start in range(0, sequence_length - window + 1):
            var here = cell_index(start, window, sequence_length)
            paired[here] = paired_cell(
                sequence_pointer, energy_model_pointer, paired_pointer, closable_pointer, sequence_length, start, window
            )
            branch_placed[here] = branch_energy(
                sequence_pointer, energy_model_pointer, paired_pointer, sequence_length, start, start + window - 1
            )
            multiloop[here] = multiloop_cell(
                sequence_pointer,
                branch_pointer,
                multiloop_pointer,
                positions,
                bounds,
                sequence_length,
                start,
                window,
            )
            closable[here] = multiloop_closable_cell(
                sequence_pointer,
                branch_pointer,
                multiloop_pointer,
                closable_pointer,
                positions,
                bounds,
                sequence_length,
                start,
                window,
            )

    # `exterior` reads every helix placement its head can reach, so it follows the square tables
    # rather than interleaving with them, and one descending pass over the starts fills it.
    for start in range(sequence_length - 1, -1, -1):
        exterior[start] = exterior_cell(
            sequence_pointer, branch_pointer, exterior_pointer, positions, bounds, sequence_length, start
        )

    return FoldTables(paired^, multiloop^, closable^, exterior^)


# endregion Serial Reference

# region GPU Sweep

comptime INTERIOR_COMBINATIONS = (MAX_LOOP + 1) * (MAX_LOOP + 1)
"""
The interior-loop scan is a triangle of unpaired counts on each side; flattening it to a square and discarding the
corner keeps the thread mapping a division rather than a search.
"""


@always_inline
def block_min(
    warp_totals: Pointer[Scalar[EnergyDType], MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    best: Int32,
) -> Int32:
    """Block-wide minimum in two stages, left in every thread so the caller needs no second barrier.

    Each warp collapses to one value through registers, so the shared array is one entry per warp
    and the whole block crosses a single barrier rather than one per level of a tree.
    """
    var reduced = warp_min(best)
    if Int(thread_idx.x) % WARP_SIZE == 0:
        warp_totals[unsafe_offset=Int(thread_idx.x) // WARP_SIZE] = reduced
    barrier()
    var combined = warp_totals[unsafe_offset=0]
    for index in range(1, WARPS_PER_BLOCK):
        combined = min(combined, warp_totals[unsafe_offset=index])
    return combined


def paired_kernel(
    sequence: Pointer[Scalar[SymbolDType], MutAnyOrigin],
    energy_model: Pointer[Scalar[EnergyDType], MutAnyOrigin],
    paired: Pointer[Scalar[EnergyDType], MutAnyOrigin],
    closable: Pointer[Scalar[EnergyDType], MutAnyOrigin],
    branch_placed: Pointer[Scalar[EnergyDType], MutAnyOrigin],
    sequence_length_in: Int32,
    window_in: Int32,
):
    """One block per `paired` cell, threads splitting its interior-loop triangle.

    The multiloop closure is a single read now that a table holds spans of two branches or more,
    so the only scan left here is the bounded one.
    """
    var sequence_length = Int(sequence_length_in)
    var window = Int(window_in)
    var start = Int(block_idx.x)
    if start + window > sequence_length:
        return
    var end = start + window - 1
    var here = cell_index(start, window, sequence_length)
    var warp_totals = stack_allocation[WARPS_PER_BLOCK, Scalar[EnergyDType], address_space=AddressSpace.SHARED]()

    var pair = pair_of(energy_model, Int(sequence[unsafe_offset=start]), Int(sequence[unsafe_offset=end]))
    if not is_pair(pair) or window < MIN_PAIRED_WINDOW:
        if thread_idx.x == 0:
            paired[unsafe_offset=here] = FORBIDDEN
            branch_placed[unsafe_offset=here] = FORBIDDEN
        return

    var best = FORBIDDEN
    if thread_idx.x == 0:
        best = hairpin_energy(sequence, energy_model, start, end)
        var enclosed = closable[unsafe_offset=cell_index(start + 1, window - 2, sequence_length)]
        if enclosed < FORBIDDEN:
            best = min(best, enclosed + multiloop_closure(pair))

    var closing = closing_pair(sequence, energy_model, start, end)
    for combined in range(Int(thread_idx.x), INTERIOR_COMBINATIONS, THREADS_PER_BLOCK):
        var before = combined // (MAX_LOOP + 1)
        var after = combined % (MAX_LOOP + 1)
        if before + after > MAX_LOOP:
            continue
        var inner_start = start + 1 + before
        var inner_end = end - 1 - after
        var inner_window = inner_end - inner_start + 1
        if inner_window < MIN_PAIRED_WINDOW or inner_start >= inner_end:
            continue
        var nested = paired[unsafe_offset=cell_index(inner_start, inner_window, sequence_length)]
        if nested >= FORBIDDEN:
            continue
        var loop = interior_energy(sequence, energy_model, closing, start, end, inner_start, inner_end)
        if loop < FORBIDDEN:
            best = min(best, loop + nested)

    var answer = block_min(warp_totals, best)
    if thread_idx.x == 0:
        paired[unsafe_offset=here] = answer
        # Stored beside the cell it derives from, so the branching sweep reads a placement rather
        # than rebuilding one for every candidate it weighs.
        var placed = FORBIDDEN
        if answer < FORBIDDEN:
            placed = (
                answer + helix_end_penalty(pair) + dangle_energy(sequence, energy_model, start, end, sequence_length)
            )
        branch_placed[unsafe_offset=here] = placed


def branching_kernel(
    sequence: Pointer[Scalar[SymbolDType], MutAnyOrigin],
    branch_placed: Pointer[Scalar[EnergyDType], MutAnyOrigin],
    multiloop: Pointer[Scalar[EnergyDType], MutAnyOrigin],
    closable: Pointer[Scalar[EnergyDType], MutAnyOrigin],
    exterior: Pointer[Scalar[EnergyDType], MutAnyOrigin],
    positions: Pointer[Scalar[PositionDType], MutAnyOrigin],
    bounds: Pointer[Scalar[PositionDType], MutAnyOrigin],
    sequence_length_in: Int32,
    window_in: Int32,
):
    """One block per cell of the three tables a closing pair does not enclose.

    All three read `paired` at their own window and none reads another at the same window, so one
    launch fills them and `block_idx.y` names which. That is two launches per window rather than
    four, which is what binds at short spans.
    """
    var sequence_length = Int(sequence_length_in)
    var window = Int(window_in)
    var start = Int(block_idx.x)
    if start + window > sequence_length:
        return
    var end = start + window - 1
    var here = cell_index(start, window, sequence_length)
    var filling = unenclosed_table(Int(block_idx.y))
    var warp_totals = stack_allocation[WARPS_PER_BLOCK, Scalar[EnergyDType], address_space=AddressSpace.SHARED]()

    # `exterior` holds one live cell per window, at the start whose end is the sequence's last.
    if filling == TableName.EXTERIOR and start != sequence_length - window:
        return

    var best = FORBIDDEN
    if thread_idx.x == 0:
        if filling == TableName.EXTERIOR:
            best = exterior[unsafe_offset=start + 1]
        elif window > 1:
            var holding = multiloop if filling == TableName.MULTILOOP else closable
            var trimmed = holding[unsafe_offset=cell_index(start + 1, window - 1, sequence_length)]
            if trimmed < FORBIDDEN:
                best = trimmed + MULTILOOP_PER_UNPAIRED

    var reach = reachable_partners(bounds, Int(sequence[unsafe_offset=start]), sequence_length, start, window)
    for index in range(reach.low + Int(thread_idx.x), reach.high, THREADS_PER_BLOCK):
        var partner = Int(positions[unsafe_offset=index])
        var branch = branch_placed[unsafe_offset=cell_index(start, partner - start + 1, sequence_length)]
        if branch >= FORBIDDEN:
            continue
        var tail = end - partner
        if filling == TableName.EXTERIOR:
            best = min(best, branch + exterior[unsafe_offset=partner + 1])
            continue
        branch += MULTILOOP_PER_HELIX
        if filling == TableName.MULTILOOP:
            best = min(best, branch + MULTILOOP_PER_UNPAIRED * Int32(tail))
        var following = multiloop[unsafe_offset=cell_index(partner + 1, tail, sequence_length)]
        if following < FORBIDDEN:
            best = min(best, branch + following)

    var answer = block_min(warp_totals, best)
    if thread_idx.x == 0:
        if filling == TableName.MULTILOOP:
            multiloop[unsafe_offset=here] = answer
        elif filling == TableName.CLOSABLE:
            closable[unsafe_offset=here] = answer
        else:
            exterior[unsafe_offset=start] = answer


def device_fold_tables(
    ctx: DeviceContext, sequence: ImmSpan[Scalar[SymbolDType], _], energy_model: ImmSpan[Scalar[EnergyDType], _]
) raises -> FoldTables:
    """Sweeps the four tables on the device, two launches per window.

    `paired` reads `closable` at a strictly shorter window, and the other three read `paired` at
    their own, so the closing table goes first and the three unenclosed ones share a launch.
    """
    var sequence_length = len(sequence)
    var cells = (sequence_length + 1) * (sequence_length + 2)
    var runs = partner_runs(sequence, energy_model, RNA_ALPHABET_SIZE)

    var sequence_buffer = upload[SymbolDType](ctx, sequence)
    var energy_model_buffer = upload[EnergyDType](ctx, energy_model)
    var positions_buffer = upload[PositionDType](ctx, Span(runs.positions))
    var bounds_buffer = upload[PositionDType](ctx, Span(runs.bounds))
    var paired_buffer = ctx.enqueue_create_buffer[EnergyDType](cells)
    var multiloop_buffer = ctx.enqueue_create_buffer[EnergyDType](cells)
    var closable_buffer = ctx.enqueue_create_buffer[EnergyDType](cells)
    var exterior_buffer = zeroed[EnergyDType](ctx, sequence_length + 2)
    var branch_buffer = ctx.enqueue_create_buffer[EnergyDType](cells)
    ctx.enqueue_memset(paired_buffer, FORBIDDEN)
    ctx.enqueue_memset(multiloop_buffer, FORBIDDEN)
    ctx.enqueue_memset(closable_buffer, FORBIDDEN)
    ctx.enqueue_memset(branch_buffer, FORBIDDEN)

    for window in range(1, sequence_length + 1):
        ctx.enqueue_function[paired_kernel](
            sequence_buffer.unsafe_ptr(),
            energy_model_buffer.unsafe_ptr(),
            paired_buffer.unsafe_ptr(),
            closable_buffer.unsafe_ptr(),
            branch_buffer.unsafe_ptr(),
            Int32(sequence_length),
            Int32(window),
            grid_dim=sequence_length - window + 1,
            block_dim=THREADS_PER_BLOCK,
        )
        ctx.enqueue_function[branching_kernel](
            sequence_buffer.unsafe_ptr(),
            branch_buffer.unsafe_ptr(),
            multiloop_buffer.unsafe_ptr(),
            closable_buffer.unsafe_ptr(),
            exterior_buffer.unsafe_ptr(),
            positions_buffer.unsafe_ptr(),
            bounds_buffer.unsafe_ptr(),
            Int32(sequence_length),
            Int32(window),
            grid_dim=(sequence_length - window + 1, 3),
            block_dim=THREADS_PER_BLOCK,
        )
    ctx.synchronize()

    # The copy overwrites every byte, so filling these first is a memset of the whole footprint.
    var paired = List[Scalar[EnergyDType]](unsafe_uninit_length=cells)
    var multiloop = List[Scalar[EnergyDType]](unsafe_uninit_length=cells)
    var closable = List[Scalar[EnergyDType]](unsafe_uninit_length=cells)
    var exterior = List[Scalar[EnergyDType]](unsafe_uninit_length=sequence_length + 2)
    ctx.enqueue_copy(paired.unsafe_ptr(), paired_buffer)
    ctx.enqueue_copy(multiloop.unsafe_ptr(), multiloop_buffer)
    ctx.enqueue_copy(closable.unsafe_ptr(), closable_buffer)
    ctx.enqueue_copy(exterior.unsafe_ptr(), exterior_buffer)
    ctx.synchronize()
    return FoldTables(paired^, multiloop^, closable^, exterior^)


# endregion GPU Sweep

# region Traceback


@fieldwise_init
struct TableName(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Which of the four tables a pending traceback item belongs to."""

    var identifier: UInt8
    """Which table this names."""
    comptime PAIRED = Self(0)
    """The window's two ends form a pair."""
    comptime MULTILOOP = Self(1)
    """The window lies inside a multibranched loop, holding at least one branch."""
    comptime EXTERIOR = Self(2)
    """A window with nothing enclosing it."""
    comptime CLOSABLE = Self(3)
    """The same as `MULTILOOP` but holding at least two branches, which is what a pair may close."""


@always_inline
def unenclosed_table(layer: Int) -> TableName:
    """Which of the three tables no pair encloses `branching_kernel`'s grid layer fills."""
    if layer == 0:
        return TableName.MULTILOOP
    if layer == 1:
        return TableName.EXTERIOR
    return TableName.CLOSABLE


@fieldwise_init
struct PendingWindow(ImplicitlyCopyable, TrivialRegisterPassable):
    """A window still to be decomposed, and which table it was reached through."""

    var table: TableName
    """Which table this window was reached through, and so which cases apply to it."""
    var start: Int32
    """First position of the window."""
    var window: Int32
    """How many positions it covers."""


@always_inline
def winning_interior(
    sequence: Pointer[Scalar[SymbolDType], _],
    energy_model: Pointer[Scalar[EnergyDType], _],
    paired: Pointer[Scalar[EnergyDType], _],
    sequence_length: Int,
    start: Int,
    window: Int,
) -> PendingWindow:
    """The nested pair whose interior loop reproduces a `paired` cell, or a negative window for none."""
    var end = start + window - 1
    var stored = paired[unsafe_offset=cell_index(start, window, sequence_length)]
    var closing = closing_pair(sequence, energy_model, start, end)
    for inner_start in range(start + 1, end):
        if inner_start - start - 1 > MAX_LOOP:
            break
        for inner_end in range(interior_end_floor(start, end, inner_start), end):
            var inner_window = inner_end - inner_start + 1
            if inner_window < MIN_PAIRED_WINDOW:
                continue
            var nested = paired[unsafe_offset=cell_index(inner_start, inner_window, sequence_length)]
            if nested >= FORBIDDEN:
                continue
            if interior_energy(sequence, energy_model, closing, start, end, inner_start, inner_end) + nested == stored:
                return PendingWindow(TableName.PAIRED, Int32(inner_start), Int32(inner_window))
    return PendingWindow(TableName.PAIRED, 0, -1)


@fieldwise_init
struct BranchChoice(ImplicitlyCopyable, TrivialRegisterPassable):
    """Which partner reproduced a branching cell's stored score, and what follows it."""

    var partner: Int
    """Position the head paired with, or negative when no case reproduced the score."""
    var tail: Int
    """Window of the multiloop tail that follows, or zero when nothing branching follows."""


@always_inline
def winning_exterior(
    sequence: Pointer[Scalar[SymbolDType], _],
    energy_model: Pointer[Scalar[EnergyDType], _],
    paired: Pointer[Scalar[EnergyDType], _],
    exterior: Pointer[Scalar[EnergyDType], _],
    positions: Pointer[Scalar[PositionDType], _],
    reach: PartnerRange,
    sequence_length: Int,
    start: Int,
    window: Int,
) -> BranchChoice:
    """Which partner's placement reproduces an `exterior` cell, or a negative partner for none."""
    var end = start + window - 1
    var stored = exterior[unsafe_offset=start]
    for index in range(reach.high - 1, reach.low - 1, -1):
        var partner = Int(positions[unsafe_offset=index])
        var branch = branch_energy(sequence, energy_model, paired, sequence_length, start, partner)
        if branch >= FORBIDDEN:
            continue
        if branch + exterior[unsafe_offset=partner + 1] == stored:
            return BranchChoice(partner, end - partner)
    return BranchChoice(-1, 0)


@always_inline
def winning_branch(
    sequence: Pointer[Scalar[SymbolDType], _],
    energy_model: Pointer[Scalar[EnergyDType], _],
    paired: Pointer[Scalar[EnergyDType], _],
    multiloop: Pointer[Scalar[EnergyDType], _],
    positions: Pointer[Scalar[PositionDType], _],
    reach: PartnerRange,
    stored: Int32,
    table: TableName,
    sequence_length: Int,
    start: Int,
    window: Int,
) -> BranchChoice:
    """Which partner reproduces a `multiloop` or `closable` cell, or a negative partner for none."""
    var end = start + window - 1
    for index in range(reach.high - 1, reach.low - 1, -1):
        var partner = Int(positions[unsafe_offset=index])
        var branch = branch_energy(sequence, energy_model, paired, sequence_length, start, partner)
        if branch >= FORBIDDEN:
            continue
        branch += MULTILOOP_PER_HELIX
        var tail = end - partner
        if table == TableName.MULTILOOP and branch + MULTILOOP_PER_UNPAIRED * Int32(tail) == stored:
            return BranchChoice(partner, 0)
        var following = multiloop[unsafe_offset=cell_index(partner + 1, tail, sequence_length)]
        if following < FORBIDDEN and branch + following == stored:
            return BranchChoice(partner, tail)
    return BranchChoice(-1, 0)


def fold_traceback(
    sequence: ImmSpan[Scalar[SymbolDType], _],
    energy_model: ImmSpan[Scalar[EnergyDType], _],
    folded: FoldTables,
) raises AffineGapsError -> String:
    """Walks the tables into a dot-bracket structure.

    Nothing is recorded during the fill: the tables are resident anyway, so testing the cases in
    the order the fill tried them costs less than a parallel array of decisions. Partners are
    walked from the furthest back, so a tie resolves to the longest helix.
    """
    var sequence_length = len(sequence)
    var sequence_pointer = sequence.unsafe_ptr()
    var energy_model_pointer = energy_model.unsafe_ptr()
    var paired = folded.paired.unsafe_ptr()
    var multiloop = folded.multiloop.unsafe_ptr()
    var closable = folded.closable.unsafe_ptr()
    var exterior = folded.exterior.unsafe_ptr()
    var runs = partner_runs(sequence, energy_model, RNA_ALPHABET_SIZE)
    var positions = runs.positions.unsafe_ptr()
    var bounds = runs.bounds.unsafe_ptr()

    var structure = List[Byte](length=sequence_length, fill=UNPAIRED_BYTE)
    var work = List[PendingWindow]()
    work.append(PendingWindow(TableName.EXTERIOR, 0, Int32(sequence_length)))

    while len(work) > 0:
        var item = work.pop()
        var start = Int(item.start)
        var window = Int(item.window)
        if window <= 0:
            continue
        var end = start + window - 1
        var here = cell_index(start, window, sequence_length)
        var reach = reachable_partners(
            bounds, Int(sequence_pointer[unsafe_offset=start]), sequence_length, start, window
        )

        if item.table == TableName.EXTERIOR:
            var stored = exterior[unsafe_offset=start]
            if exterior[unsafe_offset=start + 1] == stored:
                work.append(PendingWindow(TableName.EXTERIOR, Int32(start + 1), Int32(window - 1)))
                continue
            var choice = winning_exterior(
                sequence_pointer,
                energy_model_pointer,
                paired,
                exterior,
                positions,
                reach,
                sequence_length,
                start,
                window,
            )
            if choice.partner < 0:
                raise AffineGapsError(ErrorKind.INCONSISTENT_TABLE, "Zuker exterior loop")
            work.append(PendingWindow(TableName.PAIRED, Int32(start), Int32(choice.partner - start + 1)))
            work.append(PendingWindow(TableName.EXTERIOR, Int32(choice.partner + 1), Int32(choice.tail)))
            continue

        if item.table == TableName.MULTILOOP or item.table == TableName.CLOSABLE:
            var holding = multiloop if item.table == TableName.MULTILOOP else closable
            var stored = holding[unsafe_offset=here]
            if window > 1:
                var trimmed = holding[unsafe_offset=cell_index(start + 1, window - 1, sequence_length)]
                if trimmed < FORBIDDEN and trimmed + MULTILOOP_PER_UNPAIRED == stored:
                    work.append(PendingWindow(item.table, Int32(start + 1), Int32(window - 1)))
                    continue
            var choice = winning_branch(
                sequence_pointer,
                energy_model_pointer,
                paired,
                multiloop,
                positions,
                reach,
                stored,
                item.table,
                sequence_length,
                start,
                window,
            )
            if choice.partner < 0:
                raise AffineGapsError(ErrorKind.INCONSISTENT_TABLE, "Zuker multiloop")
            work.append(PendingWindow(TableName.PAIRED, Int32(start), Int32(choice.partner - start + 1)))
            if choice.tail > 0:
                work.append(PendingWindow(TableName.MULTILOOP, Int32(choice.partner + 1), Int32(choice.tail)))
            continue

        var stored = paired[unsafe_offset=here]
        var pair = pair_of(energy_model_pointer, Int(sequence[start]), Int(sequence[end]))
        structure[start] = OPEN_BYTE
        structure[end] = CLOSE_BYTE
        if hairpin_energy(sequence_pointer, energy_model_pointer, start, end) == stored:
            continue
        var winner = winning_interior(sequence_pointer, energy_model_pointer, paired, sequence_length, start, window)
        if winner.window > 0:
            work.append(PendingWindow(TableName.PAIRED, winner.start, winner.window))
            continue
        var enclosed = closable[unsafe_offset=cell_index(start + 1, window - 2, sequence_length)]
        if enclosed >= FORBIDDEN or enclosed + multiloop_closure(pair) != stored:
            raise AffineGapsError(ErrorKind.INCONSISTENT_TABLE, "Zuker closing pair")
        work.append(PendingWindow(TableName.CLOSABLE, Int32(start + 1), Int32(window - 2)))

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
    var energy_model = packed_energy_model()
    var folded = serial_fold_tables(Span(sequence), Span(energy_model))
    var energy = folded.exterior[0]
    var structure = fold_traceback(Span(sequence), Span(energy_model), folded)
    return FoldResult(structure^, energy)


def device_fold(ctx: DeviceContext, sequence_text: String) raises -> FoldResult:
    """Folds one RNA sequence with the sweep on the device and the walk on the host.

    Traceback stays on the host because it is a serial walk over tables the device already filled,
    and they are only quadratic, so bringing them back is cheap.
    """
    var alphabet = String(DEFAULT_RNA_ALPHABET)
    var sequence = translate(sequence_text, alphabet)
    var energy_model = packed_energy_model()
    var folded = device_fold_tables(ctx, Span(sequence), Span(energy_model))
    var energy = folded.exterior[0]
    var structure = fold_traceback(Span(sequence), Span(energy_model), folded)
    return FoldResult(structure^, energy)


# endregion Interface
