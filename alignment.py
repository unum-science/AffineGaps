"""
Gotoh affine-gap reference kernels, the parity oracle for `alignment.mojo`.

Every recurrence here is the definition the compiled kernels are checked against, so it favours
being obviously correct over being fast.
"""

from collections.abc import Callable
from enum import IntEnum, StrEnum
import numpy as np

from common import (
    AffineGapCosts,
    Background,
    SubstitutionCosts,
    TabulatedSubstitutionCosts,
    default_proteins_alphabet,
    jit_if_available,
)


class Mode(StrEnum):
    """Which of the two alignment problems the recurrence solves.

    Mirrors `AlignmentMode` in `alignment.mojo`, so the two files name the choice the same way.
    """

    GLOBAL = "global"
    LOCAL = "local"


class Layer(IntEnum):
    """Which of Gotoh's three layers a score came from, and which one a traceback is inside.

    Mirrors `Layer` in `alignment.mojo`. A walk never needs to know whether an aligning step
    matched or substituted, so the two collapse into one layer and only the emitted letters
    differ; those come from the sequences.
    """

    ALIGNING = 0
    """The score came from aligning two symbols."""
    DELETING = 1
    """The score came from a run of gaps in the second sequence."""
    INSERTING = 2
    """The score came from a run of gaps in the first sequence."""


# By default, we use BLOSUM62 with affine gap penalties
# fmt: off
default_proteins_matrix = (
    np.array(
        [
            4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4,
            -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4,
            -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4,
            -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4,
            0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4,
            -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4,
            -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,
            0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4,
            -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4,
            -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4,
            -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4,
            -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4,
            -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4,
            -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4,
            -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4,
            1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4,
            0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4,
            -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4,
            -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4,
            0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4,
            -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4,
            -1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,
            0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4,
            -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1,
        ],
        dtype=np.int8,
    ).reshape(24, 24)
    * 5
)
# fmt: on


def _reconstruct_alignment(
    changes: np.ndarray,
    scores: np.ndarray,
    deletes: np.ndarray,
    inserts: np.ndarray,
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    opening: int,
    extend: int,
    code_to_char: Callable,
    should_continue: Callable,
    mode: Mode = Mode.GLOBAL,
) -> tuple[str, str]:
    """Walks the three layers back, so a gap run is never charged twice.

    Reading only `changes` conflates the best move at a cell with whether a gap run is still opening,
    which can split one run in two and pay a second opening penalty. Consulting `deletes` and
    `inserts` keeps the walk in the layer it entered until that layer says the run began.
    """

    first_gapped, second_gapped = "", ""
    i, j = len(encoded_first), len(encoded_second)
    state = Layer.ALIGNING

    # Backtrack to recover the alignment
    while should_continue(i, j):
        if state is Layer.DELETING:
            first_gapped += code_to_char(encoded_first[i - 1])
            second_gapped += "-"
            extends = deletes[i - 1, j] + extend > scores[i - 1, j] + opening
            i -= 1
            state = Layer.DELETING if extends else Layer.ALIGNING
        elif state is Layer.INSERTING:
            first_gapped += "-"
            second_gapped += code_to_char(encoded_second[j - 1])
            extends = inserts[i, j - 1] + extend > scores[i, j - 1] + opening
            j -= 1
            state = Layer.INSERTING if extends else Layer.ALIGNING
        elif changes[i, j] == Layer.DELETING:
            state = Layer.DELETING
        elif changes[i, j] == Layer.INSERTING:
            state = Layer.INSERTING
        else:  # An aligning step, whether the two symbols matched or not
            first_gapped += code_to_char(encoded_first[i - 1])
            second_gapped += code_to_char(encoded_second[j - 1])
            i -= 1
            j -= 1

    # A global path must reach the origin, so whatever is left is genuinely aligned against gaps.
    # A local path stops wherever the score falls to zero, and everything before that is outside
    # the alignment entirely.
    if mode is Mode.LOCAL:
        return first_gapped[::-1], second_gapped[::-1]

    # Add remaining characters from `encoded_first` (with gaps in `encoded_second`)
    while i > 0:
        first_gapped += code_to_char(encoded_first[i - 1])
        second_gapped += "-"
        i -= 1

    # Add remaining characters from `encoded_second` (with gaps in `encoded_first`)
    while j > 0:
        first_gapped += "-"
        second_gapped += code_to_char(encoded_second[j - 1])
        j -= 1

    return first_gapped[::-1], second_gapped[::-1]


def _validate_gotoh_arguments(
    substitution: SubstitutionCosts | None = None,
    gaps: AffineGapCosts | None = None,
) -> tuple[str, np.ndarray, int, int]:
    """Resolves the two cost records into the alphabet, table and penalties a kernel wants.

    There is nothing left to validate. The pairing rules the flat parameters needed checking for
    are carried by the types: a uniform cost cannot omit half of itself, and it cannot also be a
    table.
    """
    gaps = gaps or AffineGapCosts()
    if substitution is None:
        return default_proteins_alphabet, default_proteins_matrix, gaps.open, gaps.extend
    if isinstance(substitution, TabulatedSubstitutionCosts):
        return substitution.alphabet, substitution.matrix, gaps.open, gaps.extend

    width = len(default_proteins_alphabet)
    matrix = np.full((width, width), substitution.mismatch)
    matrix[np.diag_indices(width)] = substitution.match
    return default_proteins_alphabet, matrix, gaps.open, gaps.extend


@jit_if_available(nopython=True)
def _levenshtein_alignment_recurrence(
    encoded_first: np.ndarray, encoded_second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns two sequences using Levenshtein's algorithm.
    The returned distance is the minimum number of single-character edits,
    including insertions, deletions, and substitutions, required to change one
    sequence into the other.

    The kernel has quadratic complexity in space and time, as it stores the
    entire scoring matrix and the operations for each cell, to allow the
    reconstruction of the alignment. Should be called through `levenshtein_alignment`.
    """
    first_length = len(encoded_first)
    second_length = len(encoded_second)

    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    scores = np.empty((first_length + 1, second_length + 1), dtype=np.int32)
    changes = np.empty((first_length + 1, second_length + 1), dtype=np.uint8)

    # Initialize the scoring matrix
    scores[0, 0] = 0
    for i in range(1, first_length + 1):
        scores[i, 0] = i
        changes[i, 0] = Layer.DELETING
    for j in range(1, second_length + 1):
        scores[0, j] = j
        changes[0, j] = Layer.INSERTING

    # Fill the scoring matrix and track operations
    for i in range(1, first_length + 1):
        for j in range(1, second_length + 1):

            substitution = int(encoded_first[i - 1] != encoded_second[j - 1])

            delete = scores[i - 1, j] + 1
            insert = scores[i, j - 1] + 1
            replace = scores[i - 1, j - 1] + substitution
            score = min(replace, delete, insert)
            scores[i, j] = score

            # Determine the minimum cost operation
            if score == replace:
                changes[i, j] = Layer.ALIGNING
            elif score == delete:
                changes[i, j] = Layer.DELETING
            else:
                changes[i, j] = Layer.INSERTING

    return scores, changes


def levenshtein_alignment(first: str, second: str) -> tuple[str, str, int]:
    """
    Aligns two sequences using Levenshtein's algorithm.
    The returned distance is the minimum number of single-character edits,
    including insertions, deletions, and substitutions, required to change one
    sequence into the other.
    """
    encoded_first = np.array([ord(c) for c in first], dtype=np.uint32)
    encoded_second = np.array([ord(c) for c in second], dtype=np.uint32)
    scores, changes = _levenshtein_alignment_recurrence(encoded_first, encoded_second)
    first_gapped, second_gapped = _reconstruct_alignment(
        changes, scores, scores, scores, encoded_first, encoded_second, 1, 1, chr, lambda i, j: i > 0 and j > 0
    )
    return first_gapped, second_gapped, int(scores[-1, -1])


@jit_if_available(nopython=True)
def _needleman_wunsch_gotoh_recurrence(
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    opening: int,
    extend: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aligns two sequences using Gotoh's affine gap penalty extensions for the
    Needleman-Wunsch global alignment algorithm.

    The kernel has quadratic complexity in space and time, as it stores the
    entire scoring matrix and the operations for each cell, to allow the
    reconstruction of the alignment. Allocates four equivalent-size matrices
    to store the scores, running cost of gaps in the first sequence, running
    cost of gaps in the second sequence, and the operations for each cell.
    Should be called through `needleman_wunsch_gotoh`.

    Returns the scores, the operation taken at each cell, and the two gap-run matrices, which the
    traceback needs together because a score alone cannot say whether a run was already open.
    """
    first_length = len(encoded_first)
    second_length = len(encoded_second)

    # Initialize the scoring matrix, following the suggestions in the paper.
    # There:
    #
    #   v ~ is gap opening penalty (always non-negative in paper, opposite for us)
    #   u ~ is gap extension penalty (always non-positive in paper, opposite for us)
    #   w(k) = u * k + v
    #
    #   D(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n
    #   P(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n,
    #             that end with a deletion of at least one residue from A, such that A(m)
    #             is aligned with a gap symbol
    #   Q(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n,
    #             that end with an insertion of at least one residue from B, such that B(n)
    #             is aligned with a gap symbol
    #
    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    scores = np.empty((first_length + 1, second_length + 1), dtype=np.int32)
    deletes = np.empty((first_length + 1, second_length + 1), dtype=np.int32)
    inserts = np.empty((first_length + 1, second_length + 1), dtype=np.int32)
    changes = np.empty((first_length + 1, second_length + 1), dtype=np.uint8)

    # Initialize the scoring matrix, following the suggestions in the paper,
    # so that the values in header (left or top) "gaps" are always smaller than those
    # in the "scores", and they are not considered as starting points in each iteration.
    scores[0, 0] = 0
    for j in range(1, second_length + 1):
        scores[0, j] = opening + (j - 1) * extend
        deletes[0, j] = scores[0, j] + opening + extend
        changes[0, j] = Layer.INSERTING

    # Fill the scoring matrix
    for i in range(1, first_length + 1):
        scores[i, 0] = opening + (i - 1) * extend
        inserts[i, 0] = scores[i, 0] + opening + extend
        changes[i, 0] = Layer.DELETING

        for j in range(1, second_length + 1):
            substitution = substitution_matrix[encoded_first[i - 1], encoded_second[j - 1]]
            delete = max(
                scores[i - 1, j] + opening,
                deletes[i - 1, j] + extend,
            )
            insert = max(
                scores[i, j - 1] + opening,
                inserts[i, j - 1] + extend,
            )
            replace = scores[i - 1, j - 1] + substitution
            score = max(replace, delete, insert)
            scores[i, j] = score
            deletes[i, j] = delete
            inserts[i, j] = insert

            # Track changes
            if score == replace:
                changes[i, j] = Layer.ALIGNING
            elif score == delete:
                changes[i, j] = Layer.DELETING
            else:
                changes[i, j] = Layer.INSERTING

    return scores, changes, deletes, inserts


@jit_if_available(nopython=True)
def _needleman_wunsch_gotoh_score_recurrence(
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    opening: int,
    extend: int,
) -> int:
    """
    Measures the alignment score of two sequences using Gotoh's affine gap penalty extensions for the
    Needleman-Wunsch global alignment algorithm. Uses less memory than the alignment function.

    The kernel has quadratic complexity in time and linear in space, as it stores
    only two rows of each matrix. Allocates four equivalent-size matrices
    to store the scores, running cost of gaps in the first sequence, running
    cost of gaps in the second sequence, and the operations for each cell.
    Should be called through `needleman_wunsch_gotoh_score`.
    """

    first_length = len(encoded_first)
    second_length = len(encoded_second)

    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    old_scores = np.empty(second_length + 1, dtype=np.int32)
    new_scores = np.empty(second_length + 1, dtype=np.int32)
    old_deletes = np.empty(second_length + 1, dtype=np.int32)
    new_deletes = np.empty(second_length + 1, dtype=np.int32)
    old_inserts = np.empty(second_length + 1, dtype=np.int32)
    new_inserts = np.empty(second_length + 1, dtype=np.int32)

    # Initialize the scoring matrix, following the suggestions in the paper,
    # so that the values in header (left or top) "gaps" are always smaller than those
    # in the "scores", and they are not considered as starting points in each iteration.
    old_scores[0] = 0
    for j in range(1, second_length + 1):
        old_scores[j] = opening + (j - 1) * extend
        old_deletes[j] = old_scores[j] + opening + extend

    for i in range(1, first_length + 1):
        new_scores[0] = opening + (i - 1) * extend
        new_inserts[0] = new_scores[0] + opening + extend

        for j in range(1, second_length + 1):
            substitution = substitution_matrix[encoded_first[i - 1], encoded_second[j - 1]]
            delete = max(old_scores[j] + opening, old_deletes[j] + extend)
            insert = max(new_scores[j - 1] + opening, new_inserts[j - 1] + extend)
            replace = old_scores[j - 1] + substitution
            score = max(replace, delete, insert)
            new_scores[j] = score
            new_deletes[j] = delete
            new_inserts[j] = insert

        # Swap rows
        old_scores, new_scores = new_scores, old_scores
        old_deletes, new_deletes = new_deletes, old_deletes
        old_inserts, new_inserts = new_inserts, old_inserts

    return old_scores[-1]


@jit_if_available(nopython=True)
def _smith_waterman_gotoh_recurrence(
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    opening: int,
    extend: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Aligns two sequences using Gotoh's affine gap penalty extensions for the
    Smith-Waterman local alignment algorithm.

    The kernel has quadratic complexity in space and time, as it stores the
    entire scoring matrix and the operations for each cell, to allow the
    reconstruction of the alignment. Allocates four equivalent-size matrices
    to store the scores, running cost of gaps in the first sequence, running
    cost of gaps in the second sequence, and the operations for each cell.
    Should be called through `smith_waterman_gotoh`.
    """
    first_length = len(encoded_first)
    second_length = len(encoded_second)

    # Initialize the scoring matrix, following the suggestions in the paper.
    # There:
    #
    #   v ~ is gap opening penalty (always non-negative in paper, opposite for us)
    #   u ~ is gap extension penalty (always non-positive in paper, opposite for us)
    #   w(k) = u * k + v
    #
    #   D(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n
    #   P(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n,
    #             that end with a deletion of at least one residue from A, such that A(m)
    #             is aligned with a gap symbol
    #   Q(m, n) ~ is the score of the optimal alignment of the prefixes of length m and n,
    #             that end with an insertion of at least one residue from B, such that B(n)
    #             is aligned with a gap symbol
    #
    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    scores = np.empty((first_length + 1, second_length + 1), dtype=np.int32)
    deletes = np.empty((first_length + 1, second_length + 1), dtype=np.int32)
    inserts = np.empty((first_length + 1, second_length + 1), dtype=np.int32)
    changes = np.empty((first_length + 1, second_length + 1), dtype=np.uint8)

    # Initialize the scoring matrix, following the suggestions in the paper,
    # so that the values in header (left or top) "gaps" are always smaller than those
    # in the "scores", and they are not considered as starting points in each iteration.
    scores[0, :] = 0
    deletes[0, :] = opening + extend
    changes[0, :] = Layer.INSERTING

    # Unlike Needleman-Wunsch, we also track the position of the maximum score.
    max_score = 0
    best_place = (0, 0)

    # Fill the scoring matrix
    for i in range(1, first_length + 1):
        scores[i, 0] = 0
        inserts[i, 0] = opening + extend
        changes[i, 0] = Layer.DELETING

        for j in range(1, second_length + 1):
            substitution = substitution_matrix[encoded_first[i - 1], encoded_second[j - 1]]
            delete = max(
                scores[i - 1, j] + opening,
                deletes[i - 1, j] + extend,
            )
            insert = max(
                scores[i, j - 1] + opening,
                inserts[i, j - 1] + extend,
            )
            replace = scores[i - 1, j - 1] + substitution
            score = max(replace, delete, insert, 0)
            scores[i, j] = score
            deletes[i, j] = delete
            inserts[i, j] = insert

            # Track changes
            if score == replace:
                changes[i, j] = Layer.ALIGNING
            elif score == delete:
                changes[i, j] = Layer.DELETING
            else:
                changes[i, j] = Layer.INSERTING

            # Update max score and position
            if score > max_score:
                max_score = score
                best_place = (i, j)

    return scores, changes, deletes, inserts, best_place


@jit_if_available(nopython=True)
def _smith_waterman_gotoh_score_recurrence(
    encoded_first: np.ndarray,
    encoded_second: np.ndarray,
    substitution_matrix: np.ndarray,
    opening: int,
    extend: int,
) -> int:
    """
    Computes the Smith-Waterman alignment score using Gotoh's affine gap penalty extensions.
    Uses only two rows per matrix to reduce memory usage.
    """
    first_length = len(encoded_first)
    second_length = len(encoded_second)

    # Let's use `np.empty` instead of `np.zeros` to avoid the initialization step.
    old_scores = np.empty(second_length + 1, dtype=np.int32)
    new_scores = np.empty(second_length + 1, dtype=np.int32)
    old_deletes = np.empty(second_length + 1, dtype=np.int32)
    new_deletes = np.empty(second_length + 1, dtype=np.int32)
    old_inserts = np.empty(second_length + 1, dtype=np.int32)
    new_inserts = np.empty(second_length + 1, dtype=np.int32)

    # Initialize the scoring matrix, following the suggestions in the paper,
    # so that the values in header (left or top) "gaps" are always smaller than those
    # in the "scores", and they are not considered as starting points in each iteration.
    old_scores[0] = 0
    for j in range(1, second_length + 1):
        old_scores[j] = 0
        old_deletes[j] = opening + extend

    max_score = 0

    for i in range(1, first_length + 1):
        new_scores[0] = 0
        new_inserts[0] = opening + extend

        for j in range(1, second_length + 1):
            substitution = substitution_matrix[encoded_first[i - 1], encoded_second[j - 1]]
            delete = max(old_scores[j] + opening, old_deletes[j] + extend)
            insert = max(new_scores[j - 1] + opening, new_inserts[j - 1] + extend)
            replace = old_scores[j - 1] + substitution
            score = max(replace, delete, insert, 0)
            new_scores[j] = score
            new_deletes[j] = delete
            new_inserts[j] = insert

            if score > max_score:
                max_score = score

        # Swap rows
        old_scores, new_scores = new_scores, old_scores
        old_deletes, new_deletes = new_deletes, old_deletes
        old_inserts, new_inserts = new_inserts, old_inserts

    return max_score


def colorize_alignment(
    first_gapped: str, second_gapped: str, background: Background = Background.DARK
) -> tuple[str, str]:
    """Paints matches, mismatches and gaps, picking the gap colour the background can show."""
    from colorama import Fore, Style

    match_color = Fore.GREEN
    mismatch_color = Fore.RED
    gap_color = Fore.WHITE if Background(background) is Background.DARK else Fore.BLACK

    colored_first = ""
    colored_second = ""

    for a, b in zip(first_gapped, second_gapped, strict=True):
        if a == b and a != "-":
            colored_first += match_color + a + Style.RESET_ALL
            colored_second += match_color + b + Style.RESET_ALL
        elif a == "-" or b == "-":
            colored_first += gap_color + a + Style.RESET_ALL
            colored_second += gap_color + b + Style.RESET_ALL
        else:
            colored_first += mismatch_color + a + Style.RESET_ALL
            colored_second += mismatch_color + b + Style.RESET_ALL

    return colored_first, colored_second
