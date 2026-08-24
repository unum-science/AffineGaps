"""
Zuker minimum free energy folding, the parity oracle for `folding.mojo`.

The recurrence is Zuker's, evaluated over the Turner nearest-neighbour model in `turner.py`. Four
tables are filled: `paired` for a subsequence whose ends pair, `multiloop` for one that sits inside
a multibranched loop, `closable` for one holding two branches or more, and `exterior` for a suffix
nothing encloses. The first three are filled together in increasing window, because each reads
`paired` at its own; `exterior` holds one live cell per window and follows in a descending pass.

Everything is indexed by `(start, length)` rather than by two endpoints, so a bifurcation reads
strictly smaller lengths and every cell of a window is independent, which is what the device sweep
needs. Memory is $O(n^2)$ and time is $O(n^3)$, the interior-loop term being bounded by capping a
loop at `MAX_LOOP` unpaired bases as every implementation of this recurrence does.

Energies are integer decikilocalories per mole throughout, so a fold is reproducible bit for bit.
"""

# pyright: reportArgumentType=false, reportReturnType=false

from enum import StrEnum

import numpy as np

from common import default_rna_alphabet, jit_if_available
from turner import (
    BULGE_INITIATION,
    DANGLE_AFTER,
    DANGLE_BEFORE,
    FORBIDDEN,
    HAIRPIN_INITIATION,
    INTERNAL_INITIATION,
    LOOP_LIMIT,
    MULTILOOP_OFFSET,
    MULTILOOP_PER_HELIX,
    MULTILOOP_PER_UNPAIRED,
    NINIO_CAP,
    NINIO_PER_ASYMMETRY,
    PAIR_INDEX,
    HEXALOOP_ENERGIES,
    HEXALOOP_KEYS,
    STACK,
    TERMINAL_AU,
    TERMINAL_MISMATCH_HAIRPIN,
    TERMINAL_MISMATCH_INTERNAL,
    TETRALOOP_ENERGIES,
    TETRALOOP_KEYS,
    TRILOOP_ENERGIES,
    TRILOOP_KEYS,
)

# Turner's model caps a bulge or internal loop at thirty unpaired bases, which is also what turns
# the interior-loop search from quartic into a constant-bounded scan.
MAX_LOOP = LOOP_LIMIT

# The backbone cannot reverse in fewer than three unpaired bases, so this floors every pair.
MIN_TURN = 3
# The turn, plus the partner past it: the shortest distance from a head to anything it can pair with.
MIN_CLOSING_REACH = MIN_TURN + 1
# The reach, plus the head: the shortest window `paired` can be finite on.
MIN_PAIRED_WINDOW = MIN_CLOSING_REACH + 1


# The two Watson-Crick pairs that are not charged a helix-end penalty.
PAIR_CG, PAIR_GC = 1, 2


NO_PAIR = -1
"""What a `PAIR_INDEX` lookup answers when two bases form none of the six pairs."""


@jit_if_available(nopython=True)
def _is_pair(index: int) -> bool:
    """Whether a `PAIR_INDEX` answer names a real pair."""
    return index != NO_PAIR


@jit_if_available(nopython=True)
def _helix_end_penalty(pair: int) -> int:
    """The helix-end penalty, charged to everything but a Watson-Crick CG or GC pair."""
    if pair in (PAIR_CG, PAIR_GC):
        return 0
    return TERMINAL_AU


@jit_if_available(nopython=True)
def _packed_key(sequence: np.ndarray, start: int, end: int) -> int:
    """The closing pair and loop bases packed base-four, most significant first."""
    key = 0
    for index in range(start, end + 1):
        key = key * 4 + sequence[index]
    return key


@jit_if_available(nopython=True)
def _special_hairpin(sequence: np.ndarray, start: int, end: int, size: int) -> int:
    """A tabulated hairpin energy for this exact loop, or `FORBIDDEN` when there is none.

    These replace initiation and terminal mismatch rather than adding to them, which is how the
    tables are defined and how `efn2` applies them.
    """
    if size == 3:
        keys, energies = TRILOOP_KEYS, TRILOOP_ENERGIES
    elif size == 4:
        keys, energies = TETRALOOP_KEYS, TETRALOOP_ENERGIES
    elif size == 6:
        keys, energies = HEXALOOP_KEYS, HEXALOOP_ENERGIES
    else:
        return FORBIDDEN
    key = _packed_key(sequence, start, end)
    for index in range(keys.shape[0]):
        if keys[index] == key:
            return energies[index]
    return FORBIDDEN


@jit_if_available(nopython=True)
def _dangle_energy(sequence: np.ndarray, start: int, end: int, length: int) -> int:
    """Both dangles on a helix placed in an exterior loop or a multiloop.

    The pair is read from outside the helix, and both neighbours are charged whenever they exist,
    which is the simple treatment that needs no extra states in the recurrence.
    """
    outward = PAIR_INDEX[sequence[end], sequence[start]]
    if not _is_pair(outward):
        return 0
    total = 0
    if start > 0:
        total += DANGLE_BEFORE[outward, sequence[start - 1]]
    if end < length - 1:
        total += DANGLE_AFTER[outward, sequence[end + 1]]
    return total


@jit_if_available(nopython=True)
def _hairpin_energy(sequence: np.ndarray, start: int, end: int) -> int:
    """A hairpin closed by `(start, end)`, with everything between it unpaired."""
    size = end - start - 1
    if size < MIN_TURN:
        return FORBIDDEN
    pair = PAIR_INDEX[sequence[start], sequence[end]]
    if not _is_pair(pair):
        return FORBIDDEN
    tabulated = _special_hairpin(sequence, start, end, size)
    if tabulated < FORBIDDEN:
        # Tabulated with the helix end factored out, as the mismatch tables are.
        return tabulated + _helix_end_penalty(pair)
    if size <= LOOP_LIMIT:
        initiation = HAIRPIN_INITIATION[size]
    else:
        # Polymer theory beyond the tabulated sizes, which is what the model prescribes.
        initiation = HAIRPIN_INITIATION[LOOP_LIMIT] + round(10.79 * np.log(size / LOOP_LIMIT))
    if size == MIN_TURN:
        return initiation + _helix_end_penalty(pair)
    # The mismatch table prices the stack across the loop; the helix end is charged separately.
    mismatch = TERMINAL_MISMATCH_HAIRPIN[pair, sequence[start + 1], sequence[end - 1]]
    return initiation + _helix_end_penalty(pair) + mismatch


@jit_if_available(nopython=True)
def _closing_pair(sequence: np.ndarray, start: int, end: int) -> tuple:
    """What a cell's closing pair contributes to every interior loop it can close.

    None of it depends on the nested pair, so a cell reads it once instead of once per candidate,
    and the interior scan is the hottest loop in the recurrence.
    """
    pair = PAIR_INDEX[sequence[start], sequence[end]]
    if not _is_pair(pair):
        return pair, 0, 0
    mismatch = TERMINAL_MISMATCH_INTERNAL[pair, sequence[start + 1], sequence[end - 1]]
    return pair, mismatch, _helix_end_penalty(pair)


@jit_if_available(nopython=True)
def _interior_energy(
    sequence: np.ndarray, closing: tuple, start: int, end: int, inner_start: int, inner_end: int
) -> int:
    """The loop between an outer pair and the pair nested directly inside it.

    Zero unpaired bases on both sides is a stack, zero on one side is a bulge, and anything else
    is an internal loop carrying Ninio's asymmetry correction.
    """
    outer, outer_mismatch, outer_penalty = closing
    inner = PAIR_INDEX[sequence[inner_start], sequence[inner_end]]
    if not _is_pair(outer) or not _is_pair(inner):
        return FORBIDDEN
    unpaired_before = inner_start - start - 1
    unpaired_after = end - inner_end - 1
    if unpaired_before + unpaired_after > MAX_LOOP:
        return FORBIDDEN
    if unpaired_before == 0 and unpaired_after == 0:
        return STACK[outer, inner]
    if unpaired_before == 0 or unpaired_after == 0:
        size = unpaired_before + unpaired_after
        if size == 1:
            # A single-base bulge keeps the helix stacked across it.
            return BULGE_INITIATION[size] + STACK[outer, inner]
        return BULGE_INITIATION[size] + outer_penalty + _helix_end_penalty(inner)
    size = unpaired_before + unpaired_after
    correction = min(abs(unpaired_before - unpaired_after) * NINIO_PER_ASYMMETRY, NINIO_CAP)
    # The loop sees the inner helix from outside, so that pair is read reversed, the same way
    # the dangle tables are indexed.
    reversed_inner = PAIR_INDEX[sequence[inner_end], sequence[inner_start]]
    inner_mismatch = TERMINAL_MISMATCH_INTERNAL[reversed_inner, sequence[inner_end + 1], sequence[inner_start - 1]]
    closure = outer_penalty + _helix_end_penalty(reversed_inner)
    return INTERNAL_INITIATION[size] + correction + closure + outer_mismatch + inner_mismatch


@jit_if_available(nopython=True)
def _interior_end_floor(start: int, end: int, inner_start: int) -> int:
    """The earliest inner end that keeps the loop within `MAX_LOOP` unpaired bases."""
    unpaired_before = inner_start - start - 1
    return max(end - 1 - (MAX_LOOP - unpaired_before), inner_start + MIN_CLOSING_REACH)


@jit_if_available(nopython=True)
def _winning_interior(sequence: np.ndarray, paired: np.ndarray, start: int, window: int) -> tuple:
    """The nested pair whose interior loop reproduces a `paired` cell, or a negative window for none."""
    end = start + window - 1
    stored = paired[start, window]
    closing = _closing_pair(sequence, start, end)
    for inner_start in range(start + 1, end):
        if inner_start - start - 1 > MAX_LOOP:
            break
        for inner_end in range(_interior_end_floor(start, end, inner_start), end):
            inner_window = inner_end - inner_start + 1
            if inner_window < MIN_PAIRED_WINDOW:
                continue
            nested = paired[inner_start, inner_window]
            if nested >= FORBIDDEN:
                continue
            if _interior_energy(sequence, closing, start, end, inner_start, inner_end) + nested == stored:
                return inner_start, inner_window
    return 0, -1


@jit_if_available(nopython=True)
def _partner_index(sequence: np.ndarray) -> tuple:
    """Where each letter's possible partners sit, as one ascending run per letter.

    `bounds[letter, threshold]` indexes that letter's run at its first entry from `threshold`
    onwards, so a window's candidates are a contiguous slice rather than a scan with a test in it.
    """
    length = sequence.shape[0]
    alphabet_size = PAIR_INDEX.shape[0]
    positions = np.zeros(alphabet_size * length, dtype=np.int64)
    bounds = np.zeros((alphabet_size, length + 1), dtype=np.int64)
    filled = 0
    for letter in range(alphabet_size):
        for position in range(length):
            bounds[letter, position] = filled
            if _is_pair(PAIR_INDEX[letter, sequence[position]]):
                positions[filled] = position
                filled += 1
        bounds[letter, length] = filled
    return positions, bounds


@jit_if_available(nopython=True)
def _branch_energy(sequence: np.ndarray, paired: np.ndarray, start: int, partner: int) -> int:
    """One helix placed as a branch: what it encloses, its helix-end penalty and its dangles.

    A pure function of where the helix starts and ends, never of what encloses it, which is why
    the sweep stores it once per window instead of recomputing it once per candidate.
    """
    closed = paired[start, partner - start + 1]
    if closed >= FORBIDDEN:
        return FORBIDDEN
    pair = PAIR_INDEX[sequence[start], sequence[partner]]
    return closed + _helix_end_penalty(pair) + _dangle_energy(sequence, start, partner, sequence.shape[0])


@jit_if_available(nopython=True)
def _reachable(bounds: np.ndarray, head: int, length: int, start: int, window: int) -> tuple:
    """The slice of `head`'s partner run this window can close a helix with.

    The `window < MIN_PAIRED_WINDOW` guard is load-bearing rather than an optimization: without it the
    threshold can run past the end of `bounds`.
    """
    if window < MIN_PAIRED_WINDOW:
        return 0, 0
    return bounds[head, start + MIN_CLOSING_REACH], bounds[head, start + window]


@jit_if_available(nopython=True)
def _multiloop_closure(pair: int) -> int:
    """What a pair pays to close a multiloop, over and above what it encloses.

    One transcription serves the sweep and the traceback, so a closure can never be tested against
    a price the fill did not use.
    """
    return MULTILOOP_OFFSET + MULTILOOP_PER_HELIX + _helix_end_penalty(pair)


@jit_if_available(nopython=True)
def _paired_cell(sequence: np.ndarray, paired: np.ndarray, closable: np.ndarray, start: int, window: int) -> int:
    """Every case of one `paired` cell: the hairpin, the interior loops, and the multiloop closure."""
    end = start + window - 1
    pair = PAIR_INDEX[sequence[start], sequence[end]]
    if not _is_pair(pair) or window < MIN_PAIRED_WINDOW:
        return FORBIDDEN

    best = _hairpin_energy(sequence, start, end)
    closing = _closing_pair(sequence, start, end)
    for inner_start in range(start + 1, end):
        if inner_start - start - 1 > MAX_LOOP:
            break
        for inner_end in range(_interior_end_floor(start, end, inner_start), end):
            inner_window = inner_end - inner_start + 1
            if inner_window < MIN_PAIRED_WINDOW:
                continue
            nested = paired[inner_start, inner_window]
            if nested >= FORBIDDEN:
                continue
            loop = _interior_energy(sequence, closing, start, end, inner_start, inner_end)
            if loop < FORBIDDEN and loop + nested < best:
                best = loop + nested

    # A multiloop closed by this pair, whose enclosed span must already hold two branches.
    enclosed = closable[start + 1, window - 2]
    if enclosed < FORBIDDEN:
        closed = enclosed + _multiloop_closure(pair)
        if closed < best:
            best = closed
    return best


@jit_if_available(nopython=True)
def _multiloop_cell(
    sequence: np.ndarray,
    branch_placed: np.ndarray,
    multiloop: np.ndarray,
    positions: np.ndarray,
    bounds: np.ndarray,
    start: int,
    window: int,
) -> int:
    """One `multiloop` cell, holding at least one branch: the head unpaired, or the head opening one."""
    end = start + window - 1
    best = FORBIDDEN
    if window > 1:
        trimmed = multiloop[start + 1, window - 1]
        if trimmed < FORBIDDEN and trimmed + MULTILOOP_PER_UNPAIRED < best:
            best = trimmed + MULTILOOP_PER_UNPAIRED
    low, high = _reachable(bounds, sequence[start], sequence.shape[0], start, window)
    for index in range(high - 1, low - 1, -1):
        partner = positions[index]
        branch = branch_placed[start, partner - start + 1]
        if branch >= FORBIDDEN:
            continue
        branch += MULTILOOP_PER_HELIX
        tail = end - partner
        # Nothing more pairs after this branch, which also covers a tail of no bases at all.
        if branch + MULTILOOP_PER_UNPAIRED * tail < best:
            best = branch + MULTILOOP_PER_UNPAIRED * tail
        following = multiloop[partner + 1, tail]
        if following < FORBIDDEN and branch + following < best:
            best = branch + following
    return best


@jit_if_available(nopython=True)
def _multiloop_closable_cell(
    sequence: np.ndarray,
    branch_placed: np.ndarray,
    multiloop: np.ndarray,
    closable: np.ndarray,
    positions: np.ndarray,
    bounds: np.ndarray,
    start: int,
    window: int,
) -> int:
    """One `multiloop_closable` cell: two or more branches, which is what a closing pair may enclose.

    The only difference from `_multiloop_cell` is the absent all-unpaired tail, and that absence is
    the entire mechanism by which Zuker's two-branch rule stays enforced.
    """
    end = start + window - 1
    best = FORBIDDEN
    if window > 1:
        trimmed = closable[start + 1, window - 1]
        if trimmed < FORBIDDEN and trimmed + MULTILOOP_PER_UNPAIRED < best:
            best = trimmed + MULTILOOP_PER_UNPAIRED
    low, high = _reachable(bounds, sequence[start], sequence.shape[0], start, window)
    for index in range(high - 1, low - 1, -1):
        partner = positions[index]
        branch = branch_placed[start, partner - start + 1]
        if branch >= FORBIDDEN:
            continue
        branch += MULTILOOP_PER_HELIX
        following = multiloop[partner + 1, end - partner]
        if following < FORBIDDEN and branch + following < best:
            best = branch + following
    return best


@jit_if_available(nopython=True)
def _exterior_cell(
    sequence: np.ndarray,
    branch_placed: np.ndarray,
    exterior: np.ndarray,
    positions: np.ndarray,
    bounds: np.ndarray,
    start: int,
) -> int:
    """One `exterior` cell: the head unpaired and free, or the head opening the leftmost helix.

    Indexed by its start alone. Both reads preserve the end, so from `exterior[0]` the recursion
    never leaves the sequence's last position and the other cells of a square table are unreachable.
    """
    length = sequence.shape[0]
    best = exterior[start + 1]
    low, high = _reachable(bounds, sequence[start], length, start, length - start)
    for index in range(high - 1, low - 1, -1):
        partner = positions[index]
        branch = branch_placed[start, partner - start + 1]
        if branch >= FORBIDDEN:
            continue
        candidate = branch + exterior[partner + 1]
        if candidate < best:
            best = candidate
    return best


@jit_if_available(nopython=True)
def _zuker_tables_recurrence(sequence: np.ndarray) -> tuple:
    """Fills the square tables in increasing window, then `exterior` in one descending pass."""
    length = sequence.shape[0]
    paired = np.full((length + 1, length + 2), FORBIDDEN, dtype=np.int32)
    multiloop = np.full((length + 1, length + 2), FORBIDDEN, dtype=np.int32)
    closable = np.full((length + 1, length + 2), FORBIDDEN, dtype=np.int32)
    exterior = np.zeros(length + 2, dtype=np.int32)
    branch_placed = np.full((length + 1, length + 2), FORBIDDEN, dtype=np.int32)
    positions, bounds = _partner_index(sequence)

    for window in range(1, length + 1):
        for start in range(0, length - window + 1):
            paired[start, window] = _paired_cell(sequence, paired, closable, start, window)
            branch_placed[start, window] = _branch_energy(sequence, paired, start, start + window - 1)
            multiloop[start, window] = _multiloop_cell(
                sequence, branch_placed, multiloop, positions, bounds, start, window
            )
            closable[start, window] = _multiloop_closable_cell(
                sequence, branch_placed, multiloop, closable, positions, bounds, start, window
            )

    # `exterior` reads every helix placement its head can reach, so it follows the square tables
    # rather than interleaving with them, and one pass over the starts fills it.
    for start in range(length - 1, -1, -1):
        exterior[start] = _exterior_cell(sequence, branch_placed, exterior, positions, bounds, start)

    return paired, multiloop, closable, exterior


class TableName(StrEnum):
    """Which of the four tables a pending traceback item belongs to."""

    PAIRED = "paired"
    """The window's two ends form a pair."""
    MULTILOOP = "multiloop"
    """The window lies inside a multibranched loop, holding at least one branch."""
    EXTERIOR = "exterior"
    """A window with nothing enclosing it."""
    CLOSABLE = "closable"
    """The same as `MULTILOOP` but holding two branches or more, which is what a pair may close."""


def _traceback(sequence: np.ndarray, tables: tuple) -> list:
    """Re-derives the decomposition of each cell, collecting the pairs it commits to.

    Nothing is recorded during the fill: the tables are resident anyway, so testing the cases in
    the same order the fill tried them costs less than a parallel array of decisions. Partners are
    walked from the furthest back, so a tie resolves to the longest helix.
    """
    paired, multiloop, closable, exterior = tables
    length = sequence.shape[0]
    base_pairs: list = []
    work = [(TableName.EXTERIOR, 0, length)]
    while work:
        table, start, window = work.pop()
        if window <= 0:
            continue
        end = start + window - 1

        if table is TableName.EXTERIOR:
            stored = exterior[start]
            if exterior[start + 1] == stored:
                work.append((TableName.EXTERIOR, start + 1, window - 1))
                continue
            for partner in range(end, start + MIN_CLOSING_REACH - 1, -1):
                branch = _branch_energy(sequence, paired, start, partner)
                if branch >= FORBIDDEN:
                    continue
                if branch + exterior[partner + 1] == stored:
                    work.append((TableName.PAIRED, start, partner - start + 1))
                    work.append((TableName.EXTERIOR, partner + 1, end - partner))
                    break
            else:
                raise ValueError("the table and the traceback disagree: Zuker exterior loop")
            continue

        if table in (TableName.MULTILOOP, TableName.CLOSABLE):
            holding = multiloop if table is TableName.MULTILOOP else closable
            stored = holding[start, window]
            if window > 1:
                trimmed = holding[start + 1, window - 1]
                if trimmed < FORBIDDEN and trimmed + MULTILOOP_PER_UNPAIRED == stored:
                    work.append((table, start + 1, window - 1))
                    continue
            for partner in range(end, start + MIN_CLOSING_REACH - 1, -1):
                branch = _branch_energy(sequence, paired, start, partner)
                if branch >= FORBIDDEN:
                    continue
                branch += MULTILOOP_PER_HELIX
                tail = end - partner
                if table is TableName.MULTILOOP and branch + MULTILOOP_PER_UNPAIRED * tail == stored:
                    work.append((TableName.PAIRED, start, partner - start + 1))
                    break
                following = multiloop[partner + 1, tail]
                if following < FORBIDDEN and branch + following == stored:
                    work.append((TableName.PAIRED, start, partner - start + 1))
                    work.append((TableName.MULTILOOP, partner + 1, tail))
                    break
            else:
                raise ValueError("the table and the traceback disagree: Zuker multiloop")
            continue

        if table is not TableName.PAIRED:
            raise ValueError(f"the table and the traceback disagree: unknown table {table!r}")

        stored = paired[start, window]
        base_pairs.append((start, end))
        if _hairpin_energy(sequence, start, end) == stored:
            continue
        inner_start, inner_window = _winning_interior(sequence, paired, start, window)
        if inner_window > 0:
            work.append((TableName.PAIRED, inner_start, inner_window))
            continue
        pair = PAIR_INDEX[sequence[start], sequence[end]]
        enclosed = closable[start + 1, window - 2]
        if enclosed < FORBIDDEN and enclosed + _multiloop_closure(pair) == stored:
            work.append((TableName.CLOSABLE, start + 1, window - 2))
            continue
        raise ValueError("the table and the traceback disagree: Zuker closing pair")
    return base_pairs


def zuker_fold(sequence: str, *, alphabet: str = default_rna_alphabet) -> tuple[str, float]:
    """Folds one RNA sequence, returning its dot-bracket structure and free energy."""
    codes = {letter: index for index, letter in enumerate(alphabet)}
    unknown = {letter for letter in set(sequence) if letter not in codes}
    if unknown:
        raise ValueError(f"Found characters outside the alphabet {alphabet!r}: {''.join(sorted(unknown))}")
    if not sequence:
        return "", 0.0

    encoded = np.array([codes[letter] for letter in sequence], dtype=np.int64)
    tables = _zuker_tables_recurrence(encoded)
    energy = int(tables[-1][0])
    structure = ["."] * len(sequence)
    for opening, closing in _traceback(encoded, tables):
        structure[opening] = "("
        structure[closing] = ")"
    return "".join(structure), energy / 10.0
