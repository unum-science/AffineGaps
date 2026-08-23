"""
Zuker minimum free energy folding, the parity oracle for `folding.mojo`.

The recurrence is Zuker's, evaluated over the Turner nearest-neighbour model in `turner.py`. Three
tables are filled in step: `paired` for a subsequence whose ends pair, `multiloop` for one that
sits inside a multibranched loop, and `external` for one that does not. `multiloop` and `external`
both read `paired` at the same span, so within one span the three are filled in that order.

Everything is indexed by `(start, length)` rather than by two endpoints, so a bifurcation reads
strictly smaller lengths and every cell of a span is independent, which is what the device sweep
needs. Memory is $O(n^2)$ and time is $O(n^3)$, the interior-loop term being bounded by capping a
loop at `MAX_LOOP` unpaired bases as every implementation of this recurrence does.

Energies are integer decikilocalories per mole throughout, so a fold is reproducible bit for bit.
"""

# pyright: reportArgumentType=false, reportReturnType=false

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

# A hairpin needs at least three unpaired bases to close.
MIN_HAIRPIN = 3

# The two Watson-Crick pairs that are not charged a helix-end penalty.
PAIR_CG, PAIR_GC = 1, 2


NO_PAIR = -1
"""What a `PAIR_INDEX` lookup answers when two bases form none of the six pairs."""


@jit_if_available(nopython=True)
def _pairs(index: int) -> bool:
    """Whether a `PAIR_INDEX` answer names a real pair."""
    return index != NO_PAIR


@jit_if_available(nopython=True)
def _terminal_penalty(pair: int) -> int:
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
    if not _pairs(outward):
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
    if size < MIN_HAIRPIN:
        return FORBIDDEN
    pair = PAIR_INDEX[sequence[start], sequence[end]]
    if not _pairs(pair):
        return FORBIDDEN
    tabulated = _special_hairpin(sequence, start, end, size)
    if tabulated < FORBIDDEN:
        return tabulated
    if size <= LOOP_LIMIT:
        initiation = HAIRPIN_INITIATION[size]
    else:
        # Polymer theory beyond the tabulated sizes, which is what the model prescribes.
        initiation = HAIRPIN_INITIATION[LOOP_LIMIT] + round(10.79 * np.log(size / LOOP_LIMIT))
    if size == MIN_HAIRPIN:
        return initiation + _terminal_penalty(pair)
    return initiation + TERMINAL_MISMATCH_HAIRPIN[pair, sequence[start + 1], sequence[end - 1]]


@jit_if_available(nopython=True)
def _interior_energy(sequence: np.ndarray, start: int, end: int, inner_start: int, inner_end: int) -> int:
    """The loop between an outer pair and the pair nested directly inside it.

    Zero unpaired bases on both sides is a stack, zero on one side is a bulge, and anything else
    is an internal loop carrying Ninio's asymmetry correction.
    """
    outer = PAIR_INDEX[sequence[start], sequence[end]]
    inner = PAIR_INDEX[sequence[inner_start], sequence[inner_end]]
    if not _pairs(outer) or not _pairs(inner):
        return FORBIDDEN
    left = inner_start - start - 1
    right = end - inner_end - 1
    if left + right > MAX_LOOP:
        return FORBIDDEN
    if left == 0 and right == 0:
        return STACK[outer, inner]
    if left == 0 or right == 0:
        size = left + right
        if size == 1:
            # A single-base bulge keeps the helix stacked across it.
            return BULGE_INITIATION[size] + STACK[outer, inner]
        return BULGE_INITIATION[size] + _terminal_penalty(outer) + _terminal_penalty(inner)
    size = left + right
    correction = min(abs(left - right) * NINIO_PER_ASYMMETRY, NINIO_CAP)
    # The loop sees the inner helix from outside, so that pair is read reversed, the same way
    # the dangle tables are indexed.
    reversed_inner = PAIR_INDEX[sequence[inner_end], sequence[inner_start]]
    outer_mismatch = TERMINAL_MISMATCH_INTERNAL[outer, sequence[start + 1], sequence[end - 1]]
    inner_mismatch = TERMINAL_MISMATCH_INTERNAL[reversed_inner, sequence[inner_end + 1], sequence[inner_start - 1]]
    return INTERNAL_INITIATION[size] + correction + outer_mismatch + inner_mismatch


@jit_if_available(nopython=True)
def _interior_end_floor(start: int, end: int, inner_start: int) -> int:
    """The earliest inner end that keeps the loop within `MAX_LOOP` unpaired bases."""
    left = inner_start - start - 1
    return max(end - 1 - (MAX_LOOP - left), inner_start + MIN_HAIRPIN + 1)


@jit_if_available(nopython=True)
def _winning_interior(sequence: np.ndarray, paired: np.ndarray, start: int, span: int) -> tuple:
    """The nested pair whose interior loop reproduces a `paired` cell, or a negative span for none."""
    end = start + span - 1
    stored = paired[start, span]
    for inner_start in range(start + 1, end):
        if inner_start - start - 1 > MAX_LOOP:
            break
        for inner_end in range(_interior_end_floor(start, end, inner_start), end):
            inner_span = inner_end - inner_start + 1
            if inner_span < MIN_HAIRPIN + 2:
                continue
            nested = paired[inner_start, inner_span]
            if nested >= FORBIDDEN:
                continue
            if _interior_energy(sequence, start, end, inner_start, inner_end) + nested == stored:
                return inner_start, inner_span
    return 0, -1


@jit_if_available(nopython=True)
def _paired_cell(sequence: np.ndarray, paired: np.ndarray, multiloop: np.ndarray, start: int, span: int) -> int:
    """Every case of one `paired` cell: the hairpin, the interior loops, and the multiloop closure."""
    end = start + span - 1
    pair = PAIR_INDEX[sequence[start], sequence[end]]
    if not _pairs(pair) or span < MIN_HAIRPIN + 2:
        return FORBIDDEN

    best = _hairpin_energy(sequence, start, end)
    for inner_start in range(start + 1, end):
        if inner_start - start - 1 > MAX_LOOP:
            break
        for inner_end in range(_interior_end_floor(start, end, inner_start), end):
            inner_span = inner_end - inner_start + 1
            if inner_span < MIN_HAIRPIN + 2:
                continue
            nested = paired[inner_start, inner_span]
            if nested >= FORBIDDEN:
                continue
            loop = _interior_energy(sequence, start, end, inner_start, inner_end)
            if loop < FORBIDDEN and loop + nested < best:
                best = loop + nested

    # A multiloop closed by this pair: two or more helices inside it.
    closure = MULTILOOP_OFFSET + MULTILOOP_PER_HELIX + _terminal_penalty(pair)
    for split in range(start + 2, end - 1):
        left = multiloop[start + 1, split - start - 1]
        right = multiloop[split, end - split]
        if left < FORBIDDEN and right < FORBIDDEN and left + right + closure < best:
            best = left + right + closure
    return best


@jit_if_available(nopython=True)
def _multiloop_seed(sequence: np.ndarray, paired: np.ndarray, multiloop: np.ndarray, start: int, span: int) -> int:
    """A `multiloop` cell before its split scan: this span closing a helix, or trimmed by a base."""
    end = start + span - 1
    best = FORBIDDEN
    closed = paired[start, span]
    if closed < FORBIDDEN:
        pair = PAIR_INDEX[sequence[start], sequence[end]]
        best = closed + MULTILOOP_PER_HELIX + _terminal_penalty(pair)
        best += _dangle_energy(sequence, start, end, sequence.shape[0])
    if span > 1:
        left_trimmed = multiloop[start + 1, span - 1]
        if left_trimmed < FORBIDDEN and left_trimmed + MULTILOOP_PER_UNPAIRED < best:
            best = left_trimmed + MULTILOOP_PER_UNPAIRED
        right_trimmed = multiloop[start, span - 1]
        if right_trimmed < FORBIDDEN and right_trimmed + MULTILOOP_PER_UNPAIRED < best:
            best = right_trimmed + MULTILOOP_PER_UNPAIRED
    return best


@jit_if_available(nopython=True)
def _external_seed(sequence: np.ndarray, paired: np.ndarray, external: np.ndarray, start: int, span: int) -> int:
    """An `external` cell before its split scan, where unpaired bases are free."""
    end = start + span - 1
    best = min(external[start + 1, span - 1], external[start, span - 1]) if span > 1 else 0
    closed = paired[start, span]
    if closed < FORBIDDEN:
        pair = PAIR_INDEX[sequence[start], sequence[end]]
        candidate = closed + _terminal_penalty(pair) + _dangle_energy(sequence, start, end, sequence.shape[0])
        if candidate < best:
            best = candidate
    return best


@jit_if_available(nopython=True)
def _zuker_kernel(sequence: np.ndarray) -> tuple:
    """Fills the three tables in increasing span, and within a span in dependency order."""
    length = sequence.shape[0]
    paired = np.full((length + 1, length + 2), FORBIDDEN, dtype=np.int32)
    multiloop = np.full((length + 1, length + 2), FORBIDDEN, dtype=np.int32)
    external = np.zeros((length + 1, length + 2), dtype=np.int32)

    for span in range(1, length + 1):
        for start in range(0, length - span + 1):
            paired[start, span] = _paired_cell(sequence, paired, multiloop, start, span)

            best = _multiloop_seed(sequence, paired, multiloop, start, span)
            for split in range(1, span):
                left = multiloop[start, split]
                right = multiloop[start + split, span - split]
                if left < FORBIDDEN and right < FORBIDDEN and left + right < best:
                    best = left + right
            multiloop[start, span] = best

            best = _external_seed(sequence, paired, external, start, span)
            for split in range(1, span):
                candidate = external[start, split] + external[start + split, span - split]
                if candidate < best:
                    best = candidate
            external[start, span] = best

    return paired, multiloop, external


def _traceback(sequence: np.ndarray, tables: tuple) -> list:
    """Re-derives the decomposition of each cell, collecting the pairs it commits to.

    Nothing is recorded during the fill: the tables are resident anyway, so testing the cases in
    the same order the fill tried them costs less than a parallel array of decisions.
    """
    paired, multiloop, external = tables
    pairs: list = []
    # Each item is a table name and the span it covers.
    work = [("external", 0, sequence.shape[0])]
    while work:
        table, start, span = work.pop()
        if span <= 0:
            continue
        end = start + span - 1
        pair = PAIR_INDEX[sequence[start], sequence[end]]

        if table == "external":
            stored = external[start, span]
            if span > 1 and external[start + 1, span - 1] == stored:
                work.append(("external", start + 1, span - 1))
                continue
            if span > 1 and external[start, span - 1] == stored:
                work.append(("external", start, span - 1))
                continue
            placed = (
                paired[start, span] + _terminal_penalty(pair) + _dangle_energy(sequence, start, end, sequence.shape[0])
            )
            if paired[start, span] < FORBIDDEN and placed == stored:
                work.append(("paired", start, span))
                continue
            for split in range(1, span):
                if external[start, split] + external[start + split, span - split] == stored:
                    work.append(("external", start, split))
                    work.append(("external", start + split, span - split))
                    break
            continue

        if table == "multiloop":
            stored = multiloop[start, span]
            placed = (
                paired[start, span]
                + MULTILOOP_PER_HELIX
                + _terminal_penalty(pair)
                + _dangle_energy(sequence, start, end, sequence.shape[0])
            )
            if paired[start, span] < FORBIDDEN and placed == stored:
                work.append(("paired", start, span))
                continue
            if span > 1 and multiloop[start + 1, span - 1] + MULTILOOP_PER_UNPAIRED == stored:
                work.append(("multiloop", start + 1, span - 1))
                continue
            if span > 1 and multiloop[start, span - 1] + MULTILOOP_PER_UNPAIRED == stored:
                work.append(("multiloop", start, span - 1))
                continue
            for split in range(1, span):
                if multiloop[start, split] + multiloop[start + split, span - split] == stored:
                    work.append(("multiloop", start, split))
                    work.append(("multiloop", start + split, span - split))
                    break
            continue

        stored = paired[start, span]
        pairs.append((start, end))
        if _hairpin_energy(sequence, start, end) == stored:
            continue
        inner_start, inner_span = _winning_interior(sequence, paired, start, span)
        if inner_span > 0:
            work.append(("paired", inner_start, inner_span))
            continue
        closure = MULTILOOP_OFFSET + MULTILOOP_PER_HELIX + _terminal_penalty(pair)
        for split in range(start + 2, end - 1):
            left = multiloop[start + 1, split - start - 1]
            right = multiloop[split, end - split]
            if left < FORBIDDEN and right < FORBIDDEN and left + right + closure == stored:
                work.append(("multiloop", start + 1, split - start - 1))
                work.append(("multiloop", split, end - split))
                break
    return pairs


def zuker_fold(sequence: str, *, alphabet: str = default_rna_alphabet) -> tuple[str, float]:
    """Folds one RNA sequence, returning its dot-bracket structure and free energy.

    The energy is kilocalories per mole, converted from the integer decikilocalories the
    recurrence works in.
    """
    codes = {letter: index for index, letter in enumerate(alphabet)}
    unknown = {letter for letter in sequence if letter not in codes}
    if unknown:
        raise ValueError(f"Found characters outside the alphabet {alphabet!r}: {''.join(sorted(unknown))}")
    if not sequence:
        return "", 0.0

    encoded = np.array([codes[letter] for letter in sequence], dtype=np.int64)
    tables = _zuker_kernel(encoded)
    energy = int(tables[2][0, len(sequence)])
    structure = ["."] * len(sequence)
    for opening, closing in _traceback(encoded, tables):
        structure[opening] = "("
        structure[closing] = ")"
    return "".join(structure), energy / 10.0
