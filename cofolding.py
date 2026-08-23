"""
Sankoff simultaneous alignment and folding, the parity oracle for `cofolding.mojo`.

Sankoff's 1985 recurrence aligns two sequences and folds them at the same time, so a base pair is
only credited when both sequences can form it. That makes the signal covariation rather than
thermodynamics, which is why no energy model appears here: the scoring is a substitution matrix
for the alignment and a pair table for the structure.

The table is indexed by `(start, length)` on each sequence rather than by four endpoints. The
bifurcation then reads strictly smaller lengths in both dimensions, so every cell on the
anti-diagonal `length_first + length_second` is independent, which is what the device sweep needs.

Cost is `O(n^6)` in time and `O(n^2 m^2)` in memory, and the memory is inherent: a bifurcation at
one layer reads every layer beneath it, so no Hirschberg-style band exists. Traceback is therefore
free, since the whole table is resident regardless.

The gap model is linear rather than affine. Affine gaps need the open state of both ends of both
windows, which multiplies a table that is already the binding constraint.
"""

# pyright: reportArgumentType=false, reportReturnType=false

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

# The recurrence's cases, in the order the kernel tries them. The traceback walks the same order
# and takes the first that reproduces the stored score, so ties resolve identically in both.
CASE_HEAD_ALIGNED = 0
CASE_TAIL_ALIGNED = 1
CASE_HEAD_GAP_IN_SECOND = 2
CASE_HEAD_GAP_IN_FIRST = 3
CASE_TAIL_GAP_IN_SECOND = 4
CASE_TAIL_GAP_IN_FIRST = 5
CASE_PAIRED = 6
CASE_BIFURCATION = 7


@jit_if_available(nopython=True)
def _bifurcation_best(
    table: np.ndarray,
    start_first: int,
    length_first: int,
    start_second: int,
    length_second: int,
) -> int:
    """The best split of both windows into two adjacent halves, or nothing when neither can split."""
    best = UNREACHABLE
    for cut_first in range(1, length_first):
        for cut_second in range(1, length_second):
            candidate = table[start_first, cut_first, start_second, cut_second]
            candidate += table[
                start_first + cut_first,
                length_first - cut_first,
                start_second + cut_second,
                length_second - cut_second,
            ]
            if candidate > best:
                best = candidate
    return best


@jit_if_available(nopython=True)
def _sankoff_cell(
    table: np.ndarray,
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    pair_matrix: np.ndarray,
    gap: int,
    start_first: int,
    length_first: int,
    start_second: int,
    length_second: int,
) -> int:
    """Every case of one Sankoff cell, including the bifurcation scan.

    An empty window on either side can only be gapped through, which is what makes the base cases
    a pair of early returns rather than a branch wrapped around the whole body.
    """
    if length_first == 0:
        return length_second * gap
    if length_second == 0:
        return length_first * gap

    head_first = encoded_first[start_first]
    head_second = encoded_second[start_second]
    tail_first = encoded_first[start_first + length_first - 1]
    tail_second = encoded_second[start_second + length_second - 1]
    head_substitution = substitution_matrix[head_first, head_second]
    tail_substitution = substitution_matrix[tail_first, tail_second]

    best = table[start_first + 1, length_first - 1, start_second + 1, length_second - 1] + head_substitution
    candidate = table[start_first, length_first - 1, start_second, length_second - 1] + tail_substitution
    if candidate > best:
        best = candidate
    candidate = table[start_first + 1, length_first - 1, start_second, length_second] + gap
    if candidate > best:
        best = candidate
    candidate = table[start_first, length_first, start_second + 1, length_second - 1] + gap
    if candidate > best:
        best = candidate
    candidate = table[start_first, length_first - 1, start_second, length_second] + gap
    if candidate > best:
        best = candidate
    candidate = table[start_first, length_first, start_second, length_second - 1] + gap
    if candidate > best:
        best = candidate

    # Both windows close a pair at once, which is where covariation is credited.
    if length_first >= 2 and length_second >= 2:
        closing_first = pair_matrix[head_first, tail_first]
        closing_second = pair_matrix[head_second, tail_second]
        if closing_first > 0 and closing_second > 0:
            candidate = table[start_first + 1, length_first - 2, start_second + 1, length_second - 2]
            candidate += closing_first + closing_second + head_substitution + tail_substitution
            if candidate > best:
                best = candidate
        candidate = _bifurcation_best(table, start_first, length_first, start_second, length_second)
        if candidate > best:
            best = candidate
    return best


@jit_if_available(nopython=True)
def _sankoff_kernel(
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    pair_matrix: np.ndarray,
    gap: int,
) -> np.ndarray:
    """Fills the whole Sankoff table, one anti-diagonal of lengths at a time.

    Every cell holds the best score for aligning-and-folding one window of each sequence. The
    caller reads the corner spanning both sequences in full.
    """
    rows = encoded_first.shape[0]
    columns = encoded_second.shape[0]
    table = np.zeros((rows + 1, rows + 1, columns + 1, columns + 1), dtype=np.int32)

    for length_first in range(rows + 1):
        for length_second in range(columns + 1):
            if length_first == 0 and length_second == 0:
                continue
            for start_first in range(rows - length_first + 1):
                for start_second in range(columns - length_second + 1):
                    table[start_first, length_first, start_second, length_second] = _sankoff_cell(
                        table,
                        encoded_first,
                        encoded_second,
                        substitution_matrix,
                        pair_matrix,
                        gap,
                        start_first,
                        length_first,
                        start_second,
                        length_second,
                    )
    return table


def _winning_case(
    table: np.ndarray,
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    pair_matrix: np.ndarray,
    gap: int,
    start_first: int,
    length_first: int,
    start_second: int,
    length_second: int,
) -> tuple[int, int, int]:
    """Which case produced a cell's stored score, and the split point when it bifurcated.

    Re-derived rather than recorded, because the table is resident anyway and a parallel array of
    decisions would cost as much again.
    """
    stored = table[start_first, length_first, start_second, length_second]
    head_first, head_second = encoded_first[start_first], encoded_second[start_second]
    tail_first = encoded_first[start_first + length_first - 1]
    tail_second = encoded_second[start_second + length_second - 1]

    if (
        table[start_first + 1, length_first - 1, start_second + 1, length_second - 1]
        + substitution_matrix[head_first, head_second]
        == stored
    ):
        return CASE_HEAD_ALIGNED, 0, 0
    if (
        table[start_first, length_first - 1, start_second, length_second - 1]
        + substitution_matrix[tail_first, tail_second]
        == stored
    ):
        return CASE_TAIL_ALIGNED, 0, 0
    if table[start_first + 1, length_first - 1, start_second, length_second] + gap == stored:
        return CASE_HEAD_GAP_IN_SECOND, 0, 0
    if table[start_first, length_first, start_second + 1, length_second - 1] + gap == stored:
        return CASE_HEAD_GAP_IN_FIRST, 0, 0
    if table[start_first, length_first - 1, start_second, length_second] + gap == stored:
        return CASE_TAIL_GAP_IN_SECOND, 0, 0
    if table[start_first, length_first, start_second, length_second - 1] + gap == stored:
        return CASE_TAIL_GAP_IN_FIRST, 0, 0

    if length_first >= 2 and length_second >= 2:
        closing_first = pair_matrix[head_first, tail_first]
        closing_second = pair_matrix[head_second, tail_second]
        if closing_first > 0 and closing_second > 0:
            inner = table[start_first + 1, length_first - 2, start_second + 1, length_second - 2]
            if (
                inner
                + closing_first
                + closing_second
                + substitution_matrix[head_first, head_second]
                + substitution_matrix[tail_first, tail_second]
                == stored
            ):
                return CASE_PAIRED, 0, 0
        for cut_first in range(1, length_first):
            for cut_second in range(1, length_second):
                left = table[start_first, cut_first, start_second, cut_second]
                right = table[
                    start_first + cut_first,
                    length_first - cut_first,
                    start_second + cut_second,
                    length_second - cut_second,
                ]
                if left + right == stored:
                    return CASE_BIFURCATION, cut_first, cut_second
    raise AssertionError("No case reproduces the stored score; the table and the traceback disagree")


def _sankoff_traceback(
    table: np.ndarray,
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    pair_matrix: np.ndarray,
    gap: int,
    alphabet: str,
) -> tuple[str, str, str]:
    """Walks the table into two gapped sequences and the consensus structure over their columns."""

    def emit(start_first: int, length_first: int, start_second: int, length_second: int):
        if length_first == 0 and length_second == 0:
            return "", "", ""
        if length_first == 0:
            span = range(start_second, start_second + length_second)
            return "-" * length_second, "".join(alphabet[encoded_second[i]] for i in span), "." * length_second
        if length_second == 0:
            span = range(start_first, start_first + length_first)
            return "".join(alphabet[encoded_first[i]] for i in span), "-" * length_first, "." * length_first

        case, cut_first, cut_second = _winning_case(
            table,
            encoded_first,
            encoded_second,
            substitution_matrix,
            pair_matrix,
            gap,
            start_first,
            length_first,
            start_second,
            length_second,
        )
        first_head, second_head = alphabet[encoded_first[start_first]], alphabet[encoded_second[start_second]]
        first_tail = alphabet[encoded_first[start_first + length_first - 1]]
        second_tail = alphabet[encoded_second[start_second + length_second - 1]]

        if case == CASE_HEAD_ALIGNED:
            left, right, shape = emit(start_first + 1, length_first - 1, start_second + 1, length_second - 1)
            return first_head + left, second_head + right, "." + shape
        if case == CASE_TAIL_ALIGNED:
            left, right, shape = emit(start_first, length_first - 1, start_second, length_second - 1)
            return left + first_tail, right + second_tail, shape + "."
        if case == CASE_HEAD_GAP_IN_SECOND:
            left, right, shape = emit(start_first + 1, length_first - 1, start_second, length_second)
            return first_head + left, "-" + right, "." + shape
        if case == CASE_HEAD_GAP_IN_FIRST:
            left, right, shape = emit(start_first, length_first, start_second + 1, length_second - 1)
            return "-" + left, second_head + right, "." + shape
        if case == CASE_TAIL_GAP_IN_SECOND:
            left, right, shape = emit(start_first, length_first - 1, start_second, length_second)
            return left + first_tail, right + "-", shape + "."
        if case == CASE_TAIL_GAP_IN_FIRST:
            left, right, shape = emit(start_first, length_first, start_second, length_second - 1)
            return left + "-", right + second_tail, shape + "."
        if case == CASE_PAIRED:
            left, right, shape = emit(start_first + 1, length_first - 2, start_second + 1, length_second - 2)
            return (
                first_head + left + first_tail,
                second_head + right + second_tail,
                "(" + shape + ")",
            )

        left_first, left_second, left_shape = emit(start_first, cut_first, start_second, cut_second)
        right_first, right_second, right_shape = emit(
            start_first + cut_first,
            length_first - cut_first,
            start_second + cut_second,
            length_second - cut_second,
        )
        return left_first + right_first, left_second + right_second, left_shape + right_shape

    return emit(0, encoded_first.shape[0], 0, encoded_second.shape[0])


def sankoff_cofold(
    first: str,
    second: str,
    *,
    alphabet: str = default_rna_alphabet,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
    pair_matrix: np.ndarray | None = None,
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
    pairs = default_rna_pair_matrix if pair_matrix is None else pair_matrix

    table = _sankoff_kernel(encoded_first, encoded_second, substitution_matrix, pairs, gap)
    score = int(table[0, len(first), 0, len(second)])
    if not first or not second:
        gapped_first = first + "-" * len(second)
        gapped_second = "-" * len(first) + second
        return gapped_first, gapped_second, "." * (len(first) + len(second)), score
    aligned_first, aligned_second, structure = _sankoff_traceback(
        table, encoded_first, encoded_second, substitution_matrix, pairs, gap, alphabet
    )
    return aligned_first, aligned_second, structure, score
