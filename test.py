#!/usr/bin/env python3
"""
Test suite for affine gap alignment.

Every property that belongs to the algorithm is asserted against each backend, so the NumPy
reference and the compiled kernels are held to one standard rather than compared after the fact.
A test opts into that axis by naming a `backend` argument; `conftest.py` supplies the values and
skips the ones the machine cannot serve.

The suite leans on three kinds of oracle:

- __Self-consistency__, where the traceback's score must equal the score-only kernel's.
- __Cross-implementation__, where every backend must return what the reference returns.
- __External__, where BioPython and brute-force enumeration answer the same question independently.

The third kind matters most: the first two would agree with each other while both being wrong.
"""

# pyright: reportArgumentType=false, reportAssignmentType=false, reportIndexIssue=false
# pyright: reportReturnType=false

import json
import os
import pathlib
import re
import subprocess
import sys
from itertools import combinations, product
from random import choice, randint, seed as random_seed
from typing import TypedDict

import numpy as np
import pytest
from Bio import Align
from Bio.Align import substitution_matrices

import affinegaps
import cofolding
import folding
import turner
from affinegaps import (
    AffineGapCosts,
    Backend,
    Device,
    TabulatedSubstitutionCosts,
    UniformSubstitutionCosts,
    available,
    default_proteins_alphabet,
    levenshtein_alignment,
    needleman_wunsch_gotoh_alignment,
    needleman_wunsch_gotoh_alignments,
    needleman_wunsch_gotoh_score,
    sankoff_cofold,
    smith_waterman_gotoh_alignment,
    smith_waterman_gotoh_alignments,
    smith_waterman_gotoh_score,
    zuker_fold,
)

randomized_repetitions_count: int = int(os.environ.get("AFFINEGAPS_REPETITIONS", "10"))
"""How many times a randomised test runs. Override with `AFFINEGAPS_REPETITIONS`."""

_seed_base: int | None = int(s) if (s := os.environ.get("AFFINEGAPS_SEED")) is not None else None
"""Base seed, if the caller wants a run reproduced. Unset means fresh data every run."""


@pytest.fixture(autouse=True)
def seed_rng(__pytest_repeat_step_number: int) -> int:
    """Seeds the generator before every test, giving each repeat step its own derived seed.

    Only when `AFFINEGAPS_SEED` is set, so the default run keeps exploring new inputs and a
    failure can still be reproduced exactly by naming the seed it ran with.
    """
    step = __pytest_repeat_step_number or 0
    seed = (_seed_base or 0) + step
    if _seed_base is not None:
        random_seed(seed)
    return seed


MODES = ["global", "local"]

# The substitution table is BLOSUM62 scaled by five, so anything compared against it must be
# scaled to match or the comparison is vacuous.
BLOSUM_SCALE = 5


def costs(match: int, mismatch: int, opening: int, extend: int) -> dict:
    """The two cost records as keyword arguments, so a test can splat one scoring into any call."""
    return {
        "substitution": UniformSubstitutionCosts(match=match, mismatch=mismatch),
        "gaps": AffineGapCosts(open=opening, extend=extend),
    }


# One representative scoring per regime rather than a grid: a cheap gap, an expensive one, and
# unit costs. The grids they replace were overlapping draws from the same family.
SCORINGS = [
    pytest.param(costs(5, -4, -20, -1), id="expensive-gap"),
    pytest.param(costs(2, -1, -2, -1), id="cheap-gap"),
    pytest.param(costs(0, -1, -1, -1), id="unit-cost"),
    pytest.param(costs(1, -1, -5, 0), id="free-extension"),
]
SCORING: dict = SCORINGS[0].values[0]


def aligner_for(mode: str):
    """The alignment entry point for a mode."""
    return needleman_wunsch_gotoh_alignment if mode == "global" else smith_waterman_gotoh_alignment


def batch_aligner_for(mode: str):
    """The batched alignment entry point for a mode."""
    return needleman_wunsch_gotoh_alignments if mode == "global" else smith_waterman_gotoh_alignments


def scorer_for(mode: str):
    """The score-only entry point for a mode."""
    return needleman_wunsch_gotoh_score if mode == "global" else smith_waterman_gotoh_score


def random_pair(shortest: int = 5, longest: int = 25, alphabet: str = default_proteins_alphabet):
    """Two independent random sequences over the default alphabet."""
    return (
        "".join(choice(alphabet) for _ in range(randint(shortest, longest))),
        "".join(choice(alphabet) for _ in range(randint(shortest, longest))),
    )


def rescore(first: str, second: str, scoring: dict | None = None) -> int:
    """Scores a gapped pair under the affine rule the recurrence claims to optimize.

    Independent of the dynamic programming, so it catches a traceback that returns a path the
    score never took.
    """
    chosen = SCORING if scoring is None else scoring
    total, in_first, in_second = 0, False, False
    for left, right in zip(first, second, strict=True):
        if left == "-" and right == "-":
            continue
        if left == "-":
            total += chosen["gaps"].extend if in_first else chosen["gaps"].open
            in_first, in_second = True, False
        elif right == "-":
            total += chosen["gaps"].extend if in_second else chosen["gaps"].open
            in_first, in_second = False, True
        else:
            total += chosen["substitution"].match if left == right else chosen["substitution"].mismatch
            in_first = in_second = False
    return total


# region Backend Axis

# Keyed by both axes. Naming a key just "gpu" would be ambiguous the moment a second backend grows
# a device path, and Numba already has one in `numba.cuda`.
ALL_BACKENDS = {
    "python-cpu": ("python", "cpu"),
    "numba-cpu": ("numba", "cpu"),
    "mojo-cpu": ("mojo", "cpu"),
    "mojo-gpu": ("mojo", "gpu"),
}


def requested_backends() -> list:
    """Which backends to exercise, narrowed by `AFFINEGAPS_BACKENDS` when it is set.

    An environment variable rather than a command-line option, because `pytest_addoption` is only
    honoured from a `conftest.py` and this suite is one file. Narrowing is rarely needed anyway:
    a backend the machine cannot serve is skipped by the probe below without being asked.
    """
    names = [n.strip() for n in os.environ.get("AFFINEGAPS_BACKENDS", "").split(",") if n.strip()]
    if not names:
        return list(ALL_BACKENDS)
    unknown = set(names) - set(ALL_BACKENDS)
    if unknown:
        raise pytest.UsageError(f"Unknown backend(s): {', '.join(sorted(unknown))}")
    return names


def pytest_generate_tests(metafunc):
    """Supplies the backend axis to any test that names it."""
    for argument, choices in (
        ("backend", requested_backends()),
        ("compiled_backend", [n for n in requested_backends() if not n.startswith("python")]),
    ):
        if argument in metafunc.fixturenames:
            metafunc.parametrize(argument, choices, indirect=True)


class Placement(TypedDict):
    """The two keywords that select where a call runs."""

    backend: Backend
    device: Device


def _scoring_keywords(name: str) -> Placement:
    """The keywords selecting one backend, skipping when this machine cannot serve it."""
    backend, device = ALL_BACKENDS[name]
    if not available(backend, device):
        pytest.skip(f"the {name} backend does not run here; see the README for how to build it")
    return {"backend": backend, "device": device}


@pytest.fixture
def backend(request):
    """Scoring keywords for one backend, across every implementation."""
    return _scoring_keywords(request.param)


@pytest.fixture
def compiled_backend(request):
    """The same, restricted to the compiled backends."""
    return _scoring_keywords(request.param)


# endregion Backend Axis


# region Algorithm Properties


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scoring", SCORINGS)
def test_score_matches_alignment(backend, mode: str, scoring: dict):
    """The traceback's score must equal what the score-only kernel computes.

    The two are separate entry points in every compiled backend, so this is the invariant a kernel
    can break on its own without any cross-backend comparison noticing.
    """
    first, second = random_pair()
    assert aligner_for(mode)(first, second, **scoring, **backend)[2] == scorer_for(mode)(
        first, second, **scoring, **backend
    )


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scoring", SCORINGS)
def test_alignment_achieves_its_score(backend, mode: str, scoring: dict):
    """Every returned path must be well formed and must realize the score reported beside it.

    With affine gaps neither is automatic. A walk that reads only the winning operation at each
    cell can leave a gap run and re-enter it, paying a second opening penalty the score never did;
    and a local walk that flushes what it did not trace returns flanking sequence that was never
    part of the alignment.
    """
    first, second = random_pair()
    gapped_first, gapped_second, score = aligner_for(mode)(first, second, **scoring, **backend)
    assert len(gapped_first) == len(gapped_second)
    assert gapped_first.replace("-", "") in first
    assert gapped_second.replace("-", "") in second
    assert rescore(gapped_first, gapped_second, scoring) == score


@pytest.mark.repeat(randomized_repetitions_count)
def test_symmetry(backend):
    """Swapping the arguments must not change the score."""
    first, second = random_pair()
    assert needleman_wunsch_gotoh_score(first, second, **SCORING, **backend) == (
        needleman_wunsch_gotoh_score(second, first, **SCORING, **backend)
    )


@pytest.mark.repeat(randomized_repetitions_count)
def test_against_levenshtein(backend):
    """At unit costs the Gotoh recurrence must reduce to edit distance.

    A second algorithm answering the same question, which pins the recurrence where a cross-backend
    comparison cannot: both backends could agree and both be wrong.
    """
    first, second = random_pair(3, 15)
    distance = levenshtein_alignment(first, second)[2]
    unit = costs(0, -1, -1, -1)
    assert -needleman_wunsch_gotoh_score(first, second, **unit, **backend) == distance


@pytest.mark.repeat(randomized_repetitions_count)
def test_gap_expansions(backend):
    """A gap that costs nothing to extend must cost the same however wide it is.

    The filler is `W`, which the default alphabet carries so the test stays on every backend, and
    the scoring makes opening a gap cheaper than a mismatch so the filler is always gapped rather
    than mismatched. Without that precondition the recurrence may prefer to mismatch the filler and
    the width legitimately changes the score — which is a property of the scoring, not a bug.
    """
    free_extension = costs(5, -10, -1, 0)
    first, second = random_pair(5, 15, alphabet="ACGT")
    cut = len(second) // 2

    widths = {}
    for width in range(1, 6):
        widened = second[:cut] + "W" * width + second[cut:]
        widths[width] = needleman_wunsch_gotoh_score(first, widened, **free_extension, **backend)
    assert len(set(widths.values())) == 1, f"a free-extension gap changed price with its width: {widths}"


@pytest.mark.parametrize("mode", MODES)
def test_scores_match_brute_force_enumeration(backend, mode: str):
    """Checks the recurrence against enumerating every alignment, using no dynamic programming.

    The strongest oracle in the suite, and the only one that can catch a recurrence which is
    self-consistently wrong across every backend at once.
    """

    def best_global(first: str, second: str) -> int:
        rows, columns = len(first), len(second)
        if not rows and not columns:
            return 0
        best = None
        for length in range(max(rows, columns), rows + columns + 1):
            for first_gaps in combinations(range(length), length - rows):
                for second_gaps in combinations(range(length), length - columns):
                    if set(first_gaps) & set(second_gaps):
                        continue
                    top, bottom = [], []
                    taken_first = taken_second = 0
                    for position in range(length):
                        if position in first_gaps:
                            top.append("-")
                        else:
                            top.append(first[taken_first])
                            taken_first += 1
                        if position in second_gaps:
                            bottom.append("-")
                        else:
                            bottom.append(second[taken_second])
                            taken_second += 1
                    candidate = rescore("".join(top), "".join(bottom))
                    if best is None or candidate > best:
                        best = candidate
        return best if best is not None else 0

    def brute_optimal(first: str, second: str) -> int:
        if mode == "global":
            return best_global(first, second)
        best = 0
        for start_first in range(len(first) + 1):
            for stop_first in range(start_first, len(first) + 1):
                for start_second in range(len(second) + 1):
                    for stop_second in range(start_second, len(second) + 1):
                        piece_first, piece_second = first[start_first:stop_first], second[start_second:stop_second]
                        if piece_first and piece_second:
                            best = max(best, best_global(piece_first, piece_second))
        return best

    scorer = scorer_for(mode)
    for length in range(4):
        for first in ["".join(t) for t in product("AR", repeat=length)] or [""]:
            for second_length in range(4):
                for second in ["".join(t) for t in product("AR", repeat=second_length)] or [""]:
                    if len(first) + len(second) <= 5:
                        assert scorer(first, second, **SCORING, **backend) == brute_optimal(first, second)


# endregion Algorithm Properties

# region External Oracles


def biopython_aligner(mode: str, scoring: dict):
    """A BioPython aligner scaled to match our table, so the comparison can actually fail.

    Our default matrix is BLOSUM62 multiplied by five. Handing BioPython the unscaled matrix while
    giving both the same gap penalties makes our score larger by construction, which is how the
    comparison this replaces could never fail.
    """
    aligner = Align.PairwiseAligner(mode=mode)
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62") * BLOSUM_SCALE
    aligner.open_gap_score = scoring["gaps"].open
    aligner.extend_gap_score = scoring["gaps"].extend
    return aligner


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "pair",
    [
        ("GIVEQCCTSICSLYQLENYCN", "HSQGTFTSDYSKYLDSRAEQDFV"),
        ("MSTAVLENPGLGRKLSDFGQETSYIEDNC", "MSTAVLENPGLGRKLSDFGQETSYIEDNS"),
        ("ACGTACGTACGT", "ACGTCGTACGTA"),
        ("W", "W"),
        ("WWWWW", "W"),
        ("ARNDCQEGHILKMFPSTWYV", "VYWTSPFMKLIHGEQCDNRA"),
    ],
)
def test_against_biopython(backend, mode: str, pair: tuple):
    """Our score must equal BioPython's on the same matrix and the same penalties.

    Scaled to match, this is an equality rather than an inequality, so a change that inflates our
    scores now fails here instead of passing silently.
    """
    gaps = {"gaps": AffineGapCosts(open=-20, extend=-1)}
    first, second = pair
    expected = biopython_aligner(mode, gaps).score(first, second)
    assert scorer_for(mode)(first, second, **gaps, **backend) == expected


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("mode", MODES)
def test_against_biopython_fuzzy(backend, mode: str):
    """The same equality on random proteins rather than curated pairs."""
    gaps = {"gaps": AffineGapCosts(open=-20, extend=-1)}
    first, second = random_pair(10, 40)
    expected = biopython_aligner(mode, gaps).score(first, second)
    assert scorer_for(mode)(first, second, **gaps, **backend) == expected


# endregion External Oracles

# region Compiled Backends


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scoring", SCORINGS)
def test_matches_reference(compiled_backend, mode: str, scoring: dict):
    """A compiled backend must return exactly what the reference returns, strings included."""
    first, second = random_pair()
    assert aligner_for(mode)(first, second, **scoring, **compiled_backend) == aligner_for(mode)(
        first, second, **scoring, backend="python"
    )


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scoring", SCORINGS)
def test_linear_space_matches_stored(compiled_backend, mode: str, scoring: dict, monkeypatch):
    """The linear-space traceback must reach the same answer as a stored decision matrix.

    Both are compiled paths, so the budget constant is the seam that forces each; there is no
    public knob and the two are otherwise indistinguishable from outside.

    The pair is long enough to split several times. A short one bottoms out in a single leaf and
    is solved by the same code either way, so it never exercises the join at all.

    Once the recursion really splits, the two need not return the same string: where several
    alignments tie, which one a divide-and-conquer join lands on is not the one a single backward
    walk lands on. What both must agree on is the score, and each must return a path that earns it.
    """
    first, second = random_pair(shortest=140, longest=260)
    monkeypatch.setattr(affinegaps, "_STORED_MATRIX_BUDGET", 10**12)
    stored = aligner_for(mode)(first, second, **scoring, **compiled_backend)
    monkeypatch.setattr(affinegaps, "_STORED_MATRIX_BUDGET", 0)
    linear = aligner_for(mode)(first, second, **scoring, **compiled_backend)
    assert linear[2] == stored[2] == scorer_for(mode)(first, second, **scoring, backend="python")
    for gapped_first, gapped_second, score in (stored, linear):
        assert len(gapped_first) == len(gapped_second)
        assert gapped_first.replace("-", "") in first
        assert gapped_second.replace("-", "") in second
        assert rescore(gapped_first, gapped_second, scoring) == score


@pytest.mark.parametrize("mode", MODES)
def test_linear_space_beats_the_stored_limit(compiled_backend, mode: str, monkeypatch):
    """Linear space must carry a pair far past what a stored matrix could hold.

    A global path spans both sequences; a local one spans a substring of each, which is the only
    difference between the two modes here.
    """
    monkeypatch.setattr(affinegaps, "_STORED_MATRIX_BUDGET", 0)
    alphabet = default_proteins_alphabet
    first = "".join(choice(alphabet) for _ in range(3000))
    second = "".join(choice(alphabet) for _ in range(3000))
    gapped_first, gapped_second, score = aligner_for(mode)(first, second, **SCORING, **compiled_backend)
    assert gapped_first.replace("-", "") in first
    assert gapped_second.replace("-", "") in second
    assert rescore(gapped_first, gapped_second) == score


@pytest.mark.parametrize("mode", MODES)
def test_batch_matches_single_pair(mode: str):
    """The batched entry points must reproduce the reference pair by pair."""
    pairs = [random_pair(5, 40) for _ in range(24)]
    firsts, seconds = [a for a, _ in pairs], [b for _, b in pairs]
    batched = batch_aligner_for(mode)
    produced = batched(firsts, seconds, **SCORING)
    expected = [aligner_for(mode)(a, b, **SCORING, backend="python") for a, b in pairs]
    assert produced == expected


@pytest.mark.parametrize("mode", MODES)
def test_batch_survives_a_pair_the_batch_kernel_cannot_take(compiled_backend, mode: str):
    """A pair the stored kernel refuses must take the linear path alone, not sink the batch.

    The device kernel indexes its carry by the first sequence, so a tall pair fails it even when
    the matrix is small; and a pair over the stored budget fails a different bound. Either one
    used to divert every other pair in the batch onto the slow per-pair route.
    """
    tall = ("A" * 40_000, "ACGT" * 4)
    ordinary = random_pair(20, 60)
    firsts = [tall[0], ordinary[0]]
    seconds = [tall[1], ordinary[1]]
    batched = batch_aligner_for(mode)
    produced = batched(firsts, seconds, **SCORING, **compiled_backend)
    for first, second, (left, right, score) in zip(firsts, seconds, produced, strict=True):
        # A global path spans both sequences; a local one spans only the core it found.
        core_first, core_second = left.replace("-", ""), right.replace("-", "")
        assert core_first == first if mode == "global" else core_first in first
        assert core_second == second if mode == "global" else core_second in second
        assert rescore(left, right) == score


def test_rejects_what_it_cannot_do():
    """The dispatcher must refuse rather than quietly doing something else."""
    with pytest.raises(ValueError):
        needleman_wunsch_gotoh_score("AR", "RA", backend="python", device="gpu")
    with pytest.raises(ValueError):
        needleman_wunsch_gotoh_score("AR", "RA", backend="CPU")  # type: ignore[arg-type]
    if available("mojo", "cpu"):
        with pytest.raises(NotImplementedError):
            needleman_wunsch_gotoh_score(
                "AR",
                "RA",
                substitution=TabulatedSubstitutionCosts("AR", np.zeros((2, 2), dtype=np.int8)),
                backend="mojo",
            )


def test_costs_cannot_express_a_contradiction():
    """The pairing rules that once needed runtime checks are carried by the types."""
    with pytest.raises(TypeError):
        UniformSubstitutionCosts(match=5)  # type: ignore[call-arg]  # cannot omit half of itself
    with pytest.raises(ValueError):
        AffineGapCosts(open=-1, extend=-20)  # a gap that costs less to open than to extend


# endregion Compiled Backends


# region Cofolding

COFOLD_ALPHABET = "ACGU"
COFOLD_MATCH, COFOLD_MISMATCH, COFOLD_GAP = 2, -1, -2


def random_rna(length: int) -> str:
    return "".join(choice(COFOLD_ALPHABET) for _ in range(length))


def random_rna_pair() -> tuple:
    """A pair of RNA sequences short enough for a recurrence that is quartic in memory."""
    return random_rna(randint(1, 12)), random_rna(randint(1, 12))


def structure_partners(structure: str) -> list:
    """The partner index of every position, or minus one where it is unpaired."""
    partners = [-1] * len(structure)
    stack: list = []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            assert stack, "a closing bracket with nothing open"
            opening = stack.pop()
            partners[opening], partners[index] = index, opening
        else:
            assert symbol == ".", f"unexpected structure symbol {symbol!r}"
    assert not stack, "an opening bracket that never closes"
    return partners


def structure_pairs(structure: str) -> list:
    """The paired column indices of a dot-bracket string, or a failure if it is unbalanced."""
    return [(opening, closing) for opening, closing in enumerate(structure_partners(structure)) if closing > opening]


def rescore_cofold(gapped_first: str, gapped_second: str, structure: str) -> int:
    """Scores an emitted alignment independently of the recurrence that produced it."""
    pair_matrix = cofolding.default_rna_pair_matrix
    total = 0
    for left, right in zip(gapped_first, gapped_second, strict=True):
        if left == "-" or right == "-":
            total += COFOLD_GAP
        else:
            total += COFOLD_MATCH if left == right else COFOLD_MISMATCH
    for opening, closing in structure_pairs(structure):
        for sequence in (gapped_first, gapped_second):
            left = COFOLD_ALPHABET.index(sequence[opening])
            right = COFOLD_ALPHABET.index(sequence[closing])
            total += int(pair_matrix[left, right])
    return total


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_recovers_both_sequences(backend):
    """Dropping the gaps from either row must give that row's input back, unchanged."""
    first, second = random_rna_pair()
    gapped_first, gapped_second, structure, _ = sankoff_cofold(first, second, **backend)
    assert len(gapped_first) == len(gapped_second) == len(structure)
    assert gapped_first.replace("-", "") == first
    assert gapped_second.replace("-", "") == second


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_pairs_are_possible_in_both_sequences(backend):
    """A base pair is only credited where both sequences can actually form it."""
    pair_matrix = cofolding.default_rna_pair_matrix
    first, second = random_rna_pair()
    gapped_first, gapped_second, structure, _ = sankoff_cofold(first, second, **backend)
    for opening, closing in structure_pairs(structure):
        for sequence in (gapped_first, gapped_second):
            assert sequence[opening] != "-" and sequence[closing] != "-", "a pair closed onto a gap"
            left = COFOLD_ALPHABET.index(sequence[opening])
            right = COFOLD_ALPHABET.index(sequence[closing])
            assert pair_matrix[left, right] > 0, f"{sequence[opening]}-{sequence[closing]} cannot pair"


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_alignment_achieves_its_score(backend):
    """The reported score must be the score of the alignment and structure actually emitted."""
    first, second = random_rna_pair()
    gapped_first, gapped_second, structure, score = sankoff_cofold(first, second, **backend)
    assert rescore_cofold(gapped_first, gapped_second, structure) == score


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_is_symmetric(backend):
    """Swapping the two sequences cannot change the optimum, because the recurrence is symmetric."""
    first, second = random_rna_pair()
    forward = sankoff_cofold(first, second, **backend)[3]
    reversed_order = sankoff_cofold(second, first, **backend)[3]
    assert forward == reversed_order


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_matches_the_reference(compiled_backend):
    """Every compiled backend must reproduce the Python oracle exactly, structure included."""
    first, second = random_rna_pair()
    expected = sankoff_cofold(first, second, backend="python", device="cpu")
    assert sankoff_cofold(first, second, **compiled_backend) == expected


def test_cofold_folds_a_hairpin(backend):
    """A stem with a loop in the middle must be found, in both sequences, against itself."""
    hairpin = "GGGGCAAAAGCCCC"
    gapped_first, gapped_second, structure, score = sankoff_cofold(hairpin, hairpin, **backend)
    assert gapped_first == gapped_second == hairpin
    assert structure.count("(") >= 3, f"expected a stem, found {structure}"
    assert rescore_cofold(gapped_first, gapped_second, structure) == score


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_credits_a_compensatory_mutation(backend):
    """A pair preserved by covariation must still be found, which is the point of the recurrence.

    The stem is GC in one sequence and AU in the other at the same columns, so the substitution
    term penalises every stem position while the pairing term rewards it.
    """
    paired_by_covariation = sankoff_cofold("GGGGAAAACCCC", "AAAAGGGGUUUU", **backend)
    structure = paired_by_covariation[2]
    assert structure.count("(") >= 2, f"covariation was not credited: {structure}"
    assert rescore_cofold(*paired_by_covariation[:3]) == paired_by_covariation[3]


# endregion Cofolding


# region Folding


def rescore_fold(sequence: str, structure: str) -> int:
    """Sums the model's terms over a structure's loop decomposition, in decikilocalories.

    Independent of the recurrence that produced the structure, so it catches a table read at the
    wrong offset or a traceback that took a case the fill did not. Helices placed in the exterior
    loop or a multiloop carry their dangles, as they do in the fill.
    """
    codes = np.array([folding.default_rna_alphabet.index(letter) for letter in sequence], dtype=np.int64)
    partners = structure_partners(structure)
    total = 0

    def children(low: int, high: int) -> list:
        found, index = [], low
        while index <= high:
            if partners[index] > index:
                found.append((index, partners[index]))
                index = partners[index] + 1
            else:
                index += 1
        return found

    def walk(opening: int, closing: int) -> None:
        nonlocal total
        nested = children(opening + 1, closing - 1)
        if not nested:
            total += int(folding._hairpin_energy(codes, opening, closing))
            return
        if len(nested) == 1:
            inner_open, inner_close = nested[0]
            total += int(folding._interior_energy(codes, opening, closing, inner_open, inner_close))
            walk(inner_open, inner_close)
            return
        pair = turner.PAIR_INDEX[codes[opening], codes[closing]]
        unpaired = (closing - opening - 1) - sum(b - a + 1 for a, b in nested)
        total += (
            turner.MULTILOOP_OFFSET
            + turner.MULTILOOP_PER_HELIX * (len(nested) + 1)
            + turner.MULTILOOP_PER_UNPAIRED * unpaired
            + int(folding._terminal_penalty(pair))
        )
        for inner_open, inner_close in nested:
            total += int(folding._terminal_penalty(turner.PAIR_INDEX[codes[inner_open], codes[inner_close]]))
            total += int(folding._dangle_energy(codes, inner_open, inner_close, len(sequence)))
            walk(inner_open, inner_close)

    for opening, closing in children(0, len(sequence) - 1):
        total += int(folding._terminal_penalty(turner.PAIR_INDEX[codes[opening], codes[closing]]))
        total += int(folding._dangle_energy(codes, opening, closing, len(sequence)))
        walk(opening, closing)
    return total


def random_foldable_rna() -> str:
    """One RNA sequence long enough to form a structure worth checking."""
    return random_rna(randint(8, 45))


@pytest.mark.repeat(randomized_repetitions_count)
def test_fold_structures_are_well_formed(backend):
    """Brackets balance, every pair is chemically possible, and hairpins are not too tight."""
    sequence = random_foldable_rna()
    structure, _ = zuker_fold(sequence, **backend)
    assert len(structure) == len(sequence)
    partners = structure_partners(structure)
    for opening, closing in enumerate(partners):
        if closing <= opening:
            continue
        left = folding.default_rna_alphabet.index(sequence[opening])
        right = folding.default_rna_alphabet.index(sequence[closing])
        assert turner.PAIR_INDEX[left, right] >= 0, f"{sequence[opening]}-{sequence[closing]} cannot pair"
        assert closing - opening - 1 >= folding.MIN_HAIRPIN, "a hairpin closed too tightly"


@pytest.mark.repeat(randomized_repetitions_count)
def test_fold_energy_matches_its_structure(backend):
    """The reported energy must be the model's energy for the structure actually emitted."""
    sequence = random_foldable_rna()
    structure, energy = zuker_fold(sequence, **backend)
    assert rescore_fold(sequence, structure) == round(energy * 10)


@pytest.mark.repeat(randomized_repetitions_count)
def test_fold_matches_the_reference(compiled_backend):
    """Every compiled backend must reproduce the Python oracle exactly, structure included."""
    sequence = random_foldable_rna()
    expected = zuker_fold(sequence, backend="python", device="cpu")
    assert zuker_fold(sequence, **compiled_backend) == expected


def test_fold_reproduces_published_energies(backend):
    """Two hairpins whose energies were checked against RNAstructure's `efn2`, term by term."""
    assert zuker_fold("GGGGCAAAAGCCCC", **backend) == ("(((((....)))))", -9.2)
    assert zuker_fold("AGGGGCAAAAGCCCCU", **backend) == ("((((((....))))))", -10.8)


def test_fold_finds_no_structure_without_pairs(backend):
    """A sequence that cannot pair with itself folds flat and costs nothing."""
    structure, energy = zuker_fold("AAAAAAAAAAAA", **backend)
    assert structure == "." * 12
    assert energy == 0.0


# endregion Folding


# region Command Line

NATIVE_BINARY = pathlib.Path(__file__).parent / "build" / "affinegaps"
"""The compiled command line, which mirrors the Python one verb for verb."""

VERB_ARGUMENTS = (
    ["align", "GIVEQCCTSICSLYQLENYCN", "HSQGTFTSDYSKYLDSRAEQDFV"],
    ["align", "GIVEQ", "HSQGT", "--local"],
    ["align", "GIVEQ", "HSQGT", "--open", "-5", "--extend", "-2"],
    ["fold", "GGGGCUUCGGCCCC"],
    ["cofold", "GGGGCAAAAGCCCC", "GGGGCUUUUGCCCC"],
    ["cofold", "GGGGCAAAAGCCCC", "GGGGCUUUUGCCCC", "--gap", "-3"],
    ["align", "GIVEQ", "HSQGT", "--gpu-id", "0", "--threads", "2"],
    ["fold", "GGGGCUUCGGCCCC", "--gpu-id", "0"],
)
"""Argument vectors both binaries must answer identically."""


def run_python_cli(arguments: list) -> subprocess.CompletedProcess:
    """The Python entry point as a subprocess, so exit codes and streams are observable."""
    return subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).parent / "affinegaps.py"), *arguments],
        capture_output=True,
        text=True,
    )


def advertised_flags(usage: str) -> set:
    """Every long flag a usage string names, which the matching parser must accept."""
    return set(re.findall(r"--[a-z][a-z-]*", usage))


def test_every_advertised_flag_is_accepted():
    """The native usage text and its parser must agree, which is the bug that prompted this design.

    The old text offered `--gpu` while the parser rejected it, so the help was the half that lied.
    """
    source = (pathlib.Path(__file__).parent / "cli.mojo").read_text()
    for verb, constant in (("align", "USAGE_ALIGN"), ("fold", "USAGE_FOLD"), ("cofold", "USAGE_COFOLD")):
        block = re.search(rf'comptime {constant} = """(.*?)"""', source, re.S)
        assert block, f"{constant} is missing from cli.mojo"
        # `align` scores over the protein alphabet by default; the folding verbs are RNA-only.
        sequences = {
            "align": ["GIVEQCCTSICSLY", "HSQGTFTSDYSKYL"],
            "fold": ["GGGGCAAAAGCCCC"],
            "cofold": ["GGGGCAAAAGCCCC", "GGGGCUUUUGCCCC"],
        }[verb]
        for flag in sorted(advertised_flags(block.group(1))):
            if flag == "--help":
                continue
            valued = {"--device": "cpu", "--format": "human", "--color": "never", "--gpu-id": "0", "--threads": "1"}
            extra = [flag] if flag in ("--local", "--verbose") else [flag, valued.get(flag, "-1")]
            # A uniform score is half a record, so the two halves are only ever given together.
            if flag in ("--match", "--mismatch"):
                extra = ["--match", "2", "--mismatch", "-1"]
            outcome = run_python_cli([verb, *sequences, *extra])
            assert outcome.returncode == 0, f"{verb} rejected its own advertised {flag}: {outcome.stderr}"


@pytest.mark.parametrize("threads", (1, 2, 8))
def test_thread_width_does_not_move_the_answer(threads: int):
    """A width is a placement knob, so the traceback it parallelizes must land on the same alignment."""
    first, second = "GIVEQCCTSICSLYQLENYCN" * 8, "HSQGTFTSDYSKYLDSRAEQDFV" * 8
    outcome = run_python_cli(
        ["align", first, second, "--threads", str(threads), "--format", "json", "--backend", "mojo", "--device", "gpu"]
    )
    assert outcome.returncode == 0, outcome.stderr
    payload = json.loads(outcome.stdout)
    assert payload["threads"] == threads
    assert payload["score"] == json.loads(run_python_cli(["align", first, second, "--format", "json"]).stdout)["score"]


@pytest.mark.parametrize("verb", ("align", "fold", "cofold"))
def test_an_absent_accelerator_is_refused(verb: str):
    """Naming a device the machine does not have is the library refusing a request, not a usage error.

    Four thousand and ninety six is past any node's accelerator count, so this holds on one GPU and on eight.
    """
    sequences = {
        "align": ["GIVEQ", "HSQGT"],
        "fold": ["GGGGCUUCGGCCCC"],
        "cofold": ["GGGGCAAAAGCCCC", "GGGGCUUUUGCCCC"],
    }[verb]
    outcome = run_python_cli([verb, *sequences, "--backend", "mojo", "--device", "gpu", "--gpu-id", "4096"])
    assert outcome.returncode == 1, outcome.stdout


@pytest.mark.parametrize("flag", ("--gpu-id", "--threads"))
def test_a_negative_count_is_a_usage_error(flag: str):
    """Neither an accelerator index nor a thread width has a meaning below zero."""
    assert run_python_cli(["align", "GIVEQ", "HSQGT", flag, "-1"]).returncode == 2


@pytest.mark.parametrize("arguments", VERB_ARGUMENTS, ids=lambda a: a[0] + "-" + str(len(a)))
def test_both_binaries_agree(arguments: list):
    """Two parsers can drift, so the guard is that they answer identically, not that we were careful."""
    if not NATIVE_BINARY.exists():
        pytest.skip("the native binary is not built; run `pixi run build-cli`")
    # A vector that names its own placement keeps it; the rest are pinned to the host, which both
    # binaries reach without an accelerator.
    pinned = [] if "--gpu-id" in arguments or "--device" in arguments else ["--device", "cpu"]
    native = subprocess.run([str(NATIVE_BINARY), *arguments, "--format", "json"], capture_output=True, text=True)
    hosted = run_python_cli([*arguments, "--format", "json", "--backend", "mojo", *pinned])
    assert native.returncode == 0 and hosted.returncode == 0, native.stderr + hosted.stderr
    assert json.loads(native.stdout) == json.loads(hosted.stdout)


@pytest.mark.parametrize("arguments", VERB_ARGUMENTS, ids=lambda a: a[0] + "-" + str(len(a)))
def test_json_carries_the_placement(arguments: list):
    """Every payload says which recurrence ran and where, so a log of runs is self-describing."""
    outcome = run_python_cli([*arguments, "--format", "json"])
    assert outcome.returncode == 0, outcome.stderr
    payload = json.loads(outcome.stdout)
    assert payload["operation"] == arguments[0]
    assert payload["backend"] in ("python", "numba", "mojo")
    assert payload["device"] in ("cpu", "gpu")


def test_verbose_leaves_stdout_parseable():
    """Placement and throughput go to stderr, so a piped payload survives `2>/dev/null`."""
    outcome = run_python_cli(["fold", "GGGGCAAAAGCCCC", "--format", "json", "--verbose"])
    assert json.loads(outcome.stdout)["structure"] == "(((((....)))))"
    assert "throughput" in outcome.stderr


def test_color_stays_out_of_pipes():
    """Colour is decided by whether stdout is a terminal, not by whether colorama imports."""
    piped = run_python_cli(["align", "GIVEQ", "HSQGT"])
    forced = run_python_cli(["align", "GIVEQ", "HSQGT", "--color", "always"])
    assert "\033" not in piped.stdout
    assert "\033" in forced.stdout


@pytest.mark.parametrize(
    "arguments,expected",
    [
        ([], 2),
        (["nonsense"], 2),
        (["align", "GIVEQ", "HSQGT", "--bogus"], 2),
        (["align", "GIVEQ", "HSQGT", "--match", "5"], 2),
        (["fold", "GGGGXAAAAGCCCC"], 1),
        (["align", "GIVEQ", "HSQGT"], 0),
    ],
    ids=["no-verb", "unknown-verb", "unknown-flag", "half-scoring", "bad-character", "success"],
)
def test_exit_codes(arguments: list, expected: int):
    """Two is a usage error the parser caught; one is a request the library refused."""
    assert run_python_cli(arguments).returncode == expected


def test_verbs_reproduce_the_validated_values():
    """The numbers checked against `efn2` and BioPython must survive any change to the surface."""
    align = json.loads(run_python_cli(["align", *VERB_ARGUMENTS[0][1:], "--format", "json"]).stdout)
    fold = json.loads(run_python_cli(["fold", "GGGGCUUCGGCCCC", "--format", "json"]).stdout)
    cofold = json.loads(run_python_cli(["cofold", *VERB_ARGUMENTS[4][1:], "--format", "json"]).stdout)
    assert align["score"] == 22
    assert fold["energy_kcal_per_mol"] == -9.6
    assert cofold["score"] == 46


# endregion Command Line
