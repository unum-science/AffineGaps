"""
Sankoff simultaneous alignment and folding, the parity oracle for `cofolding.mojo`.

Sankoff's 1985 recurrence aligns two sequences and folds them at the same time, so a base pair is
only credited when both sequences can form it. That makes the signal covariation rather than
thermodynamics, which is why no energy model appears here: the scoring is a substitution matrix
for the alignment and a pair table for the structure.

The table is indexed by `(start, length)` on each sequence rather than by four endpoints. Each
cell decides only what its two heads do, so the recurrence reads strictly smaller lengths in both
dimensions and every cell on the anti-diagonal `window_first + window_second` is independent,
which is what the device sweep needs.

The heads are gapped, aligned and unpaired, or aligned and paired with a later column, and that
last case is the whole scan: it names the partner of each head and covers both a pair closing the
window and a pair followed by more structure. Restricting the scan to partners the pair table
actually allows is what keeps it affordable, since only six of the sixteen letter pairs can close.
The traceback tries partners from the furthest back, so a tie resolves to the longest helix and a
stem comes out nested rather than chopped into neighbours. The fill's own order does not matter,
because it keeps only a maximum.

Cost is $O(n^6)$ in time and $O(n^2 m^2)$ in memory, and the memory is inherent: a pairing at one
layer reads every layer beneath it, so no Hirschberg-style band exists. Traceback is therefore
free, since the whole table is resident regardless.

The gap model is linear rather than affine. Affine gaps need the open state of both ends of both
windows, which multiplies a table that is already the binding constraint.
"""

# pyright: reportArgumentType=false, reportReturnType=false

from enum import StrEnum

import numpy as np

from common import default_rna_alphabet, jit_if_available

# Watson-Crick pairs score highest, the wobble pair lower, everything else cannot pair. These are
# the six chemically possible pairings over `default_rna_alphabet`, not measured energies.
# fmt: off
default_rna_pair_matrix = np.array(
    [
        [0, 0, 0, 2],   # A-U
        [0, 0, 3, 0],   # C-G
        [0, 3, 0, 1],   # G-C, G-U
        [2, 0, 1, 0],   # U-A, U-G
    ],
    dtype=np.int32,
)
# fmt: on

# Below any reachable score, with headroom for the penalties a reduction adds to the seed.
UNREACHABLE = np.int32(np.iinfo(np.int32).min // 4)

# A pair has to enclose enough bases to turn the backbone around, the same floor `folding.py`
# applies. Without it a head pairs with its own neighbour and the optimum is not a structure.
MIN_TURN = 3
# The turn, plus the partner past it: the shortest distance from a head to anything it can pair with.
MIN_CLOSING_REACH = MIN_TURN + 1


class SankoffCase(StrEnum):
    """Which decomposition produced a cell's score, in the order the recurrence tries them.

    The traceback walks that same order and takes the first case that reproduces the stored score,
    so a tie resolves identically in the fill and in the walk.
    """

    ALIGNED = "aligned"
    """Both heads consumed and aligned to each other, leaving that column unpaired."""
    GAP_IN_SECOND = "gap_in_second"
    """The first head consumed against a gap."""
    GAP_IN_FIRST = "gap_in_first"
    """The second head consumed against a gap."""
    PAIRED = "paired"
    """Both heads consumed, aligned and paired with a later column, which is where covariation is credited."""


def partner_index(encoded: np.ndarray, pair_scores: np.ndarray, alphabet_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Where each letter's possible partners sit in a sequence, as one ascending run per letter.

    `bounds[letter, threshold]` indexes that letter's run at its first entry from `threshold`
    onwards, so a window turns into a contiguous slice instead of a scan with a test inside it.
    """
    length = encoded.shape[0]
    runs = [np.flatnonzero(pair_scores[letter, encoded] > 0) for letter in range(alphabet_size)]
    positions = np.concatenate(runs).astype(np.int64) if length else np.zeros(0, dtype=np.int64)
    bounds = np.zeros((alphabet_size, length + 1), dtype=np.int64)
    offset = 0
    for letter, run in enumerate(runs):
        bounds[letter] = offset + np.searchsorted(run, np.arange(length + 1))
        offset += run.shape[0]
    return positions, bounds


@jit_if_available(nopython=True)
def _paired_heads(
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    start_first: int,
    start_second: int,
) -> tuple:
    """What a cell's two heads contribute to every candidate they can open.

    None of it depends on the partners, so a cell reads it once instead of once per candidate, and
    the partner scan is the hottest loop in the recurrence.
    """
    head_first = encoded_first[start_first]
    head_second = encoded_second[start_second]
    return head_first, head_second, substitution_matrix[head_first, head_second]


@jit_if_available(nopython=True)
def _paired_candidate(
    table: np.ndarray,
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    pair_scores: np.ndarray,
    heads: tuple,
    start_first: int,
    window_first: int,
    start_second: int,
    window_second: int,
    reach_first: int,
    reach_second: int,
) -> int:
    """What one pairing scores: its two closing columns, what they enclose, and what follows it."""
    head_first, head_second, head_substitution = heads
    partner_first = encoded_first[start_first + reach_first]
    partner_second = encoded_second[start_second + reach_second]
    inside = table[start_first + 1, reach_first - 1, start_second + 1, reach_second - 1]
    after = table[
        start_first + reach_first + 1,
        window_first - reach_first - 1,
        start_second + reach_second + 1,
        window_second - reach_second - 1,
    ]
    return (
        inside
        + after
        + pair_scores[head_first, partner_first]
        + pair_scores[head_second, partner_second]
        + head_substitution
        + substitution_matrix[partner_first, partner_second]
    )


@jit_if_available(nopython=True)
def _paired_best(
    table: np.ndarray,
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    pair_scores: np.ndarray,
    positions_first: np.ndarray,
    bounds_first: np.ndarray,
    positions_second: np.ndarray,
    bounds_second: np.ndarray,
    start_first: int,
    window_first: int,
    start_second: int,
    window_second: int,
) -> int:
    """The best pairing the two heads can open, over every partner both windows can reach."""
    best = UNREACHABLE
    heads = _paired_heads(encoded_first, encoded_second, substitution_matrix, start_first, start_second)
    head_first, head_second, _ = heads
    # Clamped to the window's own end, so a window too short to hold a hairpin yields no partners
    # rather than reading past the bounds table.
    floor_first = min(MIN_CLOSING_REACH, window_first)
    floor_second = min(MIN_CLOSING_REACH, window_second)
    low_first = bounds_first[head_first, start_first + floor_first]
    high_first = bounds_first[head_first, start_first + window_first]
    low_second = bounds_second[head_second, start_second + floor_second]
    high_second = bounds_second[head_second, start_second + window_second]
    for index_first in range(high_first - 1, low_first - 1, -1):
        reach_first = positions_first[index_first] - start_first
        for index_second in range(high_second - 1, low_second - 1, -1):
            candidate = _paired_candidate(
                table,
                encoded_first,
                encoded_second,
                substitution_matrix,
                pair_scores,
                heads,
                start_first,
                window_first,
                start_second,
                window_second,
                reach_first,
                positions_second[index_second] - start_second,
            )
            if candidate > best:
                best = candidate
    return int(best)


@jit_if_available(nopython=True)
def _cofold_cell(
    table: np.ndarray,
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    pair_scores: np.ndarray,
    positions_first: np.ndarray,
    bounds_first: np.ndarray,
    positions_second: np.ndarray,
    bounds_second: np.ndarray,
    gap: int,
    start_first: int,
    window_first: int,
    start_second: int,
    window_second: int,
) -> int:
    """Every case of one Sankoff cell, including the pairing scan.

    An empty window on either side can only be gapped through, which is what makes the base cases
    a pair of early returns rather than a branch wrapped around the whole body.
    """
    if window_first == 0:
        return window_second * gap
    if window_second == 0:
        return window_first * gap

    head_substitution = substitution_matrix[encoded_first[start_first], encoded_second[start_second]]
    best = table[start_first + 1, window_first - 1, start_second + 1, window_second - 1] + head_substitution
    candidate = table[start_first + 1, window_first - 1, start_second, window_second] + gap
    if candidate > best:
        best = candidate
    candidate = table[start_first, window_first, start_second + 1, window_second - 1] + gap
    if candidate > best:
        best = candidate

    candidate = _paired_best(
        table,
        encoded_first,
        encoded_second,
        substitution_matrix,
        pair_scores,
        positions_first,
        bounds_first,
        positions_second,
        bounds_second,
        start_first,
        window_first,
        start_second,
        window_second,
    )
    if candidate > best:
        best = candidate
    return best


@jit_if_available(nopython=True)
def _sankoff_table_recurrence(
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    pair_scores: np.ndarray,
    positions_first: np.ndarray,
    bounds_first: np.ndarray,
    positions_second: np.ndarray,
    bounds_second: np.ndarray,
    gap: int,
) -> np.ndarray:
    """Fills the whole Sankoff table, one anti-diagonal of lengths at a time.

    Every cell holds the best score for aligning-and-folding one window of each sequence. The
    caller reads the corner spanning both sequences in full.
    """
    rows = encoded_first.shape[0]
    columns = encoded_second.shape[0]
    table = np.zeros((rows + 1, rows + 1, columns + 1, columns + 1), dtype=np.int32)

    for window_first in range(rows + 1):
        for window_second in range(columns + 1):
            if window_first == 0 and window_second == 0:
                continue
            for start_first in range(rows - window_first + 1):
                for start_second in range(columns - window_second + 1):
                    table[start_first, window_first, start_second, window_second] = _cofold_cell(
                        table,
                        encoded_first,
                        encoded_second,
                        substitution_matrix,
                        pair_scores,
                        positions_first,
                        bounds_first,
                        positions_second,
                        bounds_second,
                        gap,
                        start_first,
                        window_first,
                        start_second,
                        window_second,
                    )
    return table


def _winning_case(
    table: np.ndarray,
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    pair_scores: np.ndarray,
    gap: int,
    start_first: int,
    window_first: int,
    start_second: int,
    window_second: int,
) -> tuple[SankoffCase, int, int]:
    """Which case produced a cell's stored score, and the two reaches when it opened a pair.

    Re-derived rather than recorded, because the table is resident anyway and a parallel array of
    decisions would cost as much again. The walk is one path rather than the whole table, so it
    tests each span for a possible pair instead of carrying the sweep's partner index.
    """
    stored = table[start_first, window_first, start_second, window_second]
    heads = _paired_heads(encoded_first, encoded_second, substitution_matrix, start_first, start_second)
    head_first, head_second, _ = heads

    if (
        table[start_first + 1, window_first - 1, start_second + 1, window_second - 1]
        + substitution_matrix[head_first, head_second]
        == stored
    ):
        return SankoffCase.ALIGNED, 0, 0
    if table[start_first + 1, window_first - 1, start_second, window_second] + gap == stored:
        return SankoffCase.GAP_IN_SECOND, 0, 0
    if table[start_first, window_first, start_second + 1, window_second - 1] + gap == stored:
        return SankoffCase.GAP_IN_FIRST, 0, 0

    for reach_first in range(window_first - 1, MIN_TURN, -1):
        if pair_scores[head_first, encoded_first[start_first + reach_first]] <= 0:
            continue
        for reach_second in range(window_second - 1, MIN_TURN, -1):
            if pair_scores[head_second, encoded_second[start_second + reach_second]] <= 0:
                continue
            if (
                _paired_candidate(
                    table,
                    encoded_first,
                    encoded_second,
                    substitution_matrix,
                    pair_scores,
                    heads,
                    start_first,
                    window_first,
                    start_second,
                    window_second,
                    reach_first,
                    reach_second,
                )
                == stored
            ):
                return SankoffCase.PAIRED, reach_first, reach_second
    raise ValueError("the table and the traceback disagree: no case reproduces the stored score")


def _sankoff_traceback(
    table: np.ndarray,
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    pair_scores: np.ndarray,
    gap: int,
    alphabet: str,
) -> tuple[str, str, str]:
    """Walks the table into two gapped sequences and the consensus structure over their columns."""

    def emit(start_first: int, window_first: int, start_second: int, window_second: int):
        if window_first == 0 and window_second == 0:
            return "", "", ""
        if window_first == 0:
            span = range(start_second, start_second + window_second)
            return (
                "-" * window_second,
                "".join(alphabet[encoded_second[position]] for position in span),
                "." * window_second,
            )
        if window_second == 0:
            span = range(start_first, start_first + window_first)
            return (
                "".join(alphabet[encoded_first[position]] for position in span),
                "-" * window_first,
                "." * window_first,
            )

        case, reach_first, reach_second = _winning_case(
            table,
            encoded_first,
            encoded_second,
            substitution_matrix,
            pair_scores,
            gap,
            start_first,
            window_first,
            start_second,
            window_second,
        )
        first_head, second_head = alphabet[encoded_first[start_first]], alphabet[encoded_second[start_second]]

        if case is SankoffCase.ALIGNED:
            left, right, shape = emit(start_first + 1, window_first - 1, start_second + 1, window_second - 1)
            return first_head + left, second_head + right, "." + shape
        if case is SankoffCase.GAP_IN_SECOND:
            left, right, shape = emit(start_first + 1, window_first - 1, start_second, window_second)
            return first_head + left, "-" + right, "." + shape
        if case is SankoffCase.GAP_IN_FIRST:
            left, right, shape = emit(start_first, window_first, start_second + 1, window_second - 1)
            return "-" + left, second_head + right, "." + shape

        inside_first, inside_second, inside_shape = emit(
            start_first + 1, reach_first - 1, start_second + 1, reach_second - 1
        )
        after_first, after_second, after_shape = emit(
            start_first + reach_first + 1,
            window_first - reach_first - 1,
            start_second + reach_second + 1,
            window_second - reach_second - 1,
        )
        first_partner = alphabet[encoded_first[start_first + reach_first]]
        second_partner = alphabet[encoded_second[start_second + reach_second]]
        return (
            first_head + inside_first + first_partner + after_first,
            second_head + inside_second + second_partner + after_second,
            "(" + inside_shape + ")" + after_shape,
        )

    return emit(0, encoded_first.shape[0], 0, encoded_second.shape[0])


def sankoff_cofold(
    first: str,
    second: str,
    *,
    alphabet: str = default_rna_alphabet,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
    pair_scores: np.ndarray | None = None,
) -> tuple[str, str, str, int]:
    """Aligns and folds two sequences at once, returning both gapped strings and their structure.

    The structure is dot-bracket over the alignment columns, so one string describes the pairing
    both sequences agree on. The score is the optimum of the recurrence, not a free energy.
    """
    codes = {letter: index for index, letter in enumerate(alphabet)}
    unknown = {letter for letter in set(first) | set(second) if letter not in codes}
    if unknown:
        raise ValueError(f"Found characters outside the alphabet {alphabet!r}: {''.join(sorted(unknown))}")

    encoded_first = np.array([codes[letter] for letter in first], dtype=np.int64)
    encoded_second = np.array([codes[letter] for letter in second], dtype=np.int64)
    size = len(alphabet)
    substitution_matrix = np.full((size, size), mismatch, dtype=np.int32)
    np.fill_diagonal(substitution_matrix, match)
    pairs = default_rna_pair_matrix if pair_scores is None else pair_scores
    positions_first, bounds_first = partner_index(encoded_first, pairs, size)
    positions_second, bounds_second = partner_index(encoded_second, pairs, size)

    table = _sankoff_table_recurrence(
        encoded_first,
        encoded_second,
        substitution_matrix,
        pairs,
        positions_first,
        bounds_first,
        positions_second,
        bounds_second,
        gap,
    )
    score = int(table[0, len(first), 0, len(second)])
    if not first or not second:
        gapped_first = first + "-" * len(second)
        gapped_second = "-" * len(first) + second
        return gapped_first, gapped_second, "." * (len(first) + len(second)), score
    aligned_first, aligned_second, structure = _sankoff_traceback(
        table, encoded_first, encoded_second, substitution_matrix, pairs, gap, alphabet
    )
    return aligned_first, aligned_second, structure, score
