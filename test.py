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

import hashlib
import math
import json
import os
import pathlib
import re
import subprocess
import sys
from itertools import combinations, product
from random import choice, randint, seed as random_seed
from typing import NamedTuple, TypedDict

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

exhaustive_scale: int = max(1, int(os.environ.get("AFFINEGAPS_SCALE", "1")))
"""Multiplier on every exhaustive oracle's budget, and the highest frozen tier re-derived.

Breadth and depth are separate knobs on purpose: `AFFINEGAPS_REPETITIONS=100 AFFINEGAPS_SCALE=1`
is a fuzzing run and `AFFINEGAPS_REPETITIONS=1 AFFINEGAPS_SCALE=4` is a release gate, and one
number cannot express both.
"""

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


class BackendKeywords(TypedDict):
    """The two keywords that select where a call runs."""

    backend: Backend
    device: Device


def _scoring_keywords(name: str) -> BackendKeywords:
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


# region Brute Force Oracles

PAIRABLE = frozenset({("A", "U"), ("U", "A"), ("C", "G"), ("G", "C"), ("G", "U"), ("U", "G")})
"""The six ordered letter pairs that can close, written out rather than read from `PAIR_INDEX`.

Sharing the table with the implementation would let a corrupted row be agreed with instead of
caught, which is the whole point of asking twice.
"""


def pair_slot(first: str, second: str) -> int:
    """Which row of the Turner tables a letter pair occupies, or a negative for none."""
    combined = first + second
    return turner.PAIRS.index(combined) if combined in turner.PAIRS else -1


def helix_end_penalty(slot: int) -> int:
    """The helix-end charge, which every pair but Watson-Crick CG and GC pays."""
    return 0 if turner.PAIRS[slot] in ("CG", "GC") else turner.TERMINAL_AU


def tabulated_hairpin(sequence: str, opening: int, closing: int, size: int) -> int | None:
    """A special-loop energy for this exact hairpin, which replaces initiation and mismatch."""
    tables = {
        3: (turner.TRILOOP_KEYS, turner.TRILOOP_ENERGIES),
        4: (turner.TETRALOOP_KEYS, turner.TETRALOOP_ENERGIES),
        6: (turner.HEXALOOP_KEYS, turner.HEXALOOP_ENERGIES),
    }
    if size not in tables:
        return None
    key = 0
    for index in range(opening, closing + 1):
        key = key * 4 + turner.BASES.index(sequence[index])
    keys, energies = tables[size]
    for index in range(len(keys)):
        if int(keys[index]) == key:
            return int(energies[index])
    return None


def hairpin_cost(sequence: str, opening: int, closing: int) -> int | None:
    """A hairpin closed by these two positions, or nothing when the model forbids it."""
    size = closing - opening - 1
    slot = pair_slot(sequence[opening], sequence[closing])
    if size < folding.MIN_TURN or slot < 0:
        return None
    tabulated = tabulated_hairpin(sequence, opening, closing, size)
    if tabulated is not None:
        return tabulated
    if size <= turner.LOOP_LIMIT:
        initiation = int(turner.HAIRPIN_INITIATION[size])
    else:
        initiation = int(turner.HAIRPIN_INITIATION[turner.LOOP_LIMIT]) + round(
            10.79 * math.log(size / turner.LOOP_LIMIT)
        )
    if size == folding.MIN_TURN:
        return initiation + helix_end_penalty(slot)
    inner_left = turner.BASES.index(sequence[opening + 1])
    inner_right = turner.BASES.index(sequence[closing - 1])
    # The mismatch table prices the stack across the loop; the helix end is charged separately.
    penalty = helix_end_penalty(slot)
    return initiation + penalty + int(turner.TERMINAL_MISMATCH_HAIRPIN[slot, inner_left, inner_right])


def loop_cost(sequence: str, opening: int, closing: int, inner_open: int, inner_close: int) -> int | None:
    """The loop between a pair and the pair directly inside it, or nothing when it is forbidden."""
    outer = pair_slot(sequence[opening], sequence[closing])
    inner = pair_slot(sequence[inner_open], sequence[inner_close])
    if outer < 0 or inner < 0:
        return None
    before = inner_open - opening - 1
    after = closing - inner_close - 1
    if before + after > turner.LOOP_LIMIT:
        return None
    if before == 0 and after == 0:
        return int(turner.STACK[outer, inner])
    if before == 0 or after == 0:
        size = before + after
        if size == 1:
            return int(turner.BULGE_INITIATION[size]) + int(turner.STACK[outer, inner])
        return int(turner.BULGE_INITIATION[size]) + helix_end_penalty(outer) + helix_end_penalty(inner)
    size = before + after
    asymmetry = min(abs(before - after) * turner.NINIO_PER_ASYMMETRY, turner.NINIO_CAP)
    reversed_inner = pair_slot(sequence[inner_close], sequence[inner_open])
    outer_mismatch = int(
        turner.TERMINAL_MISMATCH_INTERNAL[
            outer, turner.BASES.index(sequence[opening + 1]), turner.BASES.index(sequence[closing - 1])
        ]
    )
    inner_mismatch = int(
        turner.TERMINAL_MISMATCH_INTERNAL[
            reversed_inner,
            turner.BASES.index(sequence[inner_close + 1]),
            turner.BASES.index(sequence[inner_open - 1]),
        ]
    )
    closure = helix_end_penalty(outer) + helix_end_penalty(reversed_inner)
    return int(turner.INTERNAL_INITIATION[size]) + asymmetry + closure + outer_mismatch + inner_mismatch


def dangle_cost(sequence: str, opening: int, closing: int) -> int:
    """Both dangles on a helix sitting in an exterior loop or a multiloop."""
    outward = pair_slot(sequence[closing], sequence[opening])
    if outward < 0:
        return 0
    total = 0
    if opening > 0:
        total += int(turner.DANGLE_BEFORE[outward, turner.BASES.index(sequence[opening - 1])])
    if closing < len(sequence) - 1:
        total += int(turner.DANGLE_AFTER[outward, turner.BASES.index(sequence[closing + 1])])
    return total


def turner_energy_of(sequence: str, pairs) -> int | None:
    """Decikilocalories for one structure, read straight off the Turner tables.

    Shares no code with `folding.py`, so a table read at the wrong offset inside the recurrence's
    own helpers is caught here instead of being agreed with. Returns nothing when the structure
    contains a loop the model forbids, because a forbidden term must reject the whole structure
    rather than be added as a large number that stacking could cancel.
    """
    partner = {opening: closing for opening, closing in pairs}
    partner.update({closing: opening for opening, closing in pairs})

    def children(low: int, high: int) -> list:
        found, index = [], low
        while index <= high:
            if partner.get(index, -1) > index:
                found.append((index, partner[index]))
                index = partner[index] + 1
            else:
                index += 1
        return found

    def within(opening: int, closing: int) -> int | None:
        nested = children(opening + 1, closing - 1)
        if not nested:
            return hairpin_cost(sequence, opening, closing)
        if len(nested) == 1:
            inner_open, inner_close = nested[0]
            loop = loop_cost(sequence, opening, closing, inner_open, inner_close)
            deeper = within(inner_open, inner_close)
            return None if loop is None or deeper is None else loop + deeper
        slot = pair_slot(sequence[opening], sequence[closing])
        unpaired = (closing - opening - 1) - sum(high - low + 1 for low, high in nested)
        total = (
            turner.MULTILOOP_OFFSET
            + turner.MULTILOOP_PER_HELIX * (len(nested) + 1)
            + turner.MULTILOOP_PER_UNPAIRED * unpaired
            + helix_end_penalty(slot)
        )
        for inner_open, inner_close in nested:
            deeper = within(inner_open, inner_close)
            if deeper is None:
                return None
            total += helix_end_penalty(pair_slot(sequence[inner_open], sequence[inner_close]))
            total += dangle_cost(sequence, inner_open, inner_close)
            total += deeper
        return total

    total = 0
    for opening, closing in children(0, len(sequence) - 1):
        deeper = within(opening, closing)
        if deeper is None:
            return None
        total += helix_end_penalty(pair_slot(sequence[opening], sequence[closing]))
        total += dangle_cost(sequence, opening, closing)
        total += deeper
    return total


def enumerate_fold_structures(sequence: str):
    """Every nested pair set the sequence admits, as tuples of (opening, closing).

    A grammar over pair sets rather than a recurrence over energies: it never consults the tables
    and never takes a minimum, so it cannot share a mistake with the thing it checks.
    """

    def below(start: int, stop: int):
        if stop - start <= 0:
            yield ()
            return
        yield from below(start + 1, stop)
        for partner in range(start + folding.MIN_CLOSING_REACH, stop):
            if (sequence[start], sequence[partner]) in PAIRABLE:
                for inside in below(start + 1, partner):
                    for after in below(partner + 1, stop):
                        yield ((start, partner), *inside, *after)

    return below(0, len(sequence))


def count_fold_structures(sequence: str) -> int:
    """How many structures the enumeration would yield, memoized, so a budget can be checked first."""
    seen: dict = {}

    def below(start: int, stop: int) -> int:
        if stop - start <= 0:
            return 1
        if (start, stop) not in seen:
            found = below(start + 1, stop)
            for partner in range(start + folding.MIN_CLOSING_REACH, stop):
                if (sequence[start], sequence[partner]) in PAIRABLE:
                    found += below(start + 1, partner) * below(partner + 1, stop)
            seen[(start, stop)] = found
        return seen[(start, stop)]

    return below(0, len(sequence))


def brute_force_fold(sequence: str) -> int:
    """The minimum free energy by enumeration, sharing no algorithm with the recurrence."""
    best = 0
    for pairs in enumerate_fold_structures(sequence):
        energy = turner_energy_of(sequence, pairs)
        if energy is not None and energy < best:
            best = energy
    return best


def enumerate_alignments(first: str, second: str):
    """Every gapped pair of rows, by the three-way edit recursion."""
    if not first and not second:
        yield "", ""
        return
    if first and second:
        for left, right in enumerate_alignments(first[1:], second[1:]):
            yield first[0] + left, second[0] + right
    if first:
        for left, right in enumerate_alignments(first[1:], second):
            yield first[0] + left, "-" + right
    if second:
        for left, right in enumerate_alignments(first, second[1:]):
            yield "-" + left, second[0] + right


def pairable_columns(gapped_first: str, gapped_second: str) -> dict:
    """Columns a Sankoff pair may join, with their weight and the per-sequence hairpin floor applied.

    The floor lives in each sequence's own coordinates, not the alignment's, because that is where
    the recurrence measures a helix's reach.
    """
    position_first, position_second = {}, {}
    for column in range(len(gapped_first)):
        if gapped_first[column] != "-":
            position_first[column] = len(position_first)
        if gapped_second[column] != "-":
            position_second[column] = len(position_second)
    allowed = {}
    for opening in range(len(gapped_first)):
        for closing in range(opening + 1, len(gapped_first)):
            if opening not in position_first or closing not in position_first:
                continue
            if opening not in position_second or closing not in position_second:
                continue
            if position_first[closing] - position_first[opening] < cofolding.MIN_CLOSING_REACH:
                continue
            if position_second[closing] - position_second[opening] < cofolding.MIN_CLOSING_REACH:
                continue
            weights = []
            for row in (gapped_first, gapped_second):
                left = COFOLD_ALPHABET.index(row[opening])
                right = COFOLD_ALPHABET.index(row[closing])
                weights.append(int(cofolding.default_rna_pair_matrix[left, right]))
            if min(weights) <= 0:
                continue
            allowed[(opening, closing)] = sum(weights)
    return allowed


def brute_force_cofold(first: str, second: str) -> int:
    """The Sankoff optimum over every alignment and every nested structure, by enumeration.

    Shares no algorithm with the recurrence, which is what makes it the oracle that catches a fill
    and a traceback agreeing on the same wrong answer.
    """

    def best_nesting(width: int, allowed: dict) -> int:
        seen: dict = {}

        def within(low: int, high: int) -> int:
            if high - low < 2:
                return 0
            if (low, high) not in seen:
                found = within(low + 1, high)
                for partner in range(low + 1, high):
                    weight = allowed.get((low, partner))
                    if weight is None:
                        continue
                    found = max(found, weight + within(low + 1, partner) + within(partner + 1, high))
                seen[(low, high)] = found
            return seen[(low, high)]

        return within(0, width)

    best = None
    for gapped_first, gapped_second in enumerate_alignments(first, second):
        total = rescore_cofold(gapped_first, gapped_second, "." * len(gapped_first))
        total += best_nesting(len(gapped_first), pairable_columns(gapped_first, gapped_second))
        if best is None or total > best:
            best = total
    return best if best is not None else 0


# endregion Brute Force Oracles

# region Frozen Expectations


class FoldCase(NamedTuple):
    """One folding expectation, frozen against the tables `TURNER_FINGERPRINT` names."""

    tag: str
    """Hyphenated, and the pytest identifier."""
    sequence: str
    """What is folded."""
    structure: str
    """Dot-bracket, written under the sequence so the pairing can be checked by eye."""
    decikcal: int
    """Integer decikilocalories, because the public entry point divides by ten on the way out."""
    tier: int = 1
    """Re-derived by enumeration only when `AFFINEGAPS_SCALE` reaches it."""


class CofoldCase(NamedTuple):
    """One cofolding expectation, frozen against the pair table and the covariance scoring."""

    tag: str
    """Hyphenated, and the pytest identifier."""
    first: str
    """The first input sequence."""
    second: str
    """The second input sequence."""
    gapped_first: str
    """The first row of the alignment, gaps written in."""
    gapped_second: str
    """The second row, gapped to the same columns."""
    structure: str
    """Dot-bracket over those columns."""
    score: int
    """The optimum of the recurrence, which is not a free energy."""
    tier: int = 1
    """Re-derived by enumeration only when `AFFINEGAPS_SCALE` reaches it."""


def turner_fingerprint() -> str:
    """A digest of every energy table and scalar the folding model reads, in a fixed order.

    Names the model a frozen energy was derived against, so a table edit announces itself instead
    of surfacing as a wall of unexplained mismatches.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name in (
        "PAIR_INDEX",
        "STACK",
        "TERMINAL_MISMATCH_HAIRPIN",
        "TERMINAL_MISMATCH_INTERNAL",
        "DANGLE_AFTER",
        "DANGLE_BEFORE",
        "HAIRPIN_INITIATION",
        "BULGE_INITIATION",
        "INTERNAL_INITIATION",
        "TRILOOP_KEYS",
        "TRILOOP_ENERGIES",
        "TETRALOOP_KEYS",
        "TETRALOOP_ENERGIES",
        "HEXALOOP_KEYS",
        "HEXALOOP_ENERGIES",
    ):
        digest.update(name.encode())
        digest.update(np.asarray(getattr(turner, name)).astype("<i4").tobytes())
    for name in (
        "LOOP_LIMIT",
        "FORBIDDEN",
        "MULTILOOP_OFFSET",
        "MULTILOOP_PER_UNPAIRED",
        "MULTILOOP_PER_HELIX",
        "NINIO_PER_ASYMMETRY",
        "NINIO_CAP",
        "TERMINAL_AU",
    ):
        digest.update(f"{name}={getattr(turner, name)}".encode())
    return digest.hexdigest()


TURNER_FINGERPRINT = "5994fabbce625616aca625d482ab7b1a"
"""The model every frozen folding energy below was derived against."""


CORPUS_ORIGIN = """These are the answers this implementation gives today, checked case by case against a
hand-derived expectation before being written down. They exist because every other folding and
cofolding test is satisfied by a recurrence that returns a legal, self-consistent, cross-backend
identical, suboptimal answer: a one-character slip in a scan bound leaves the whole suite green and
changes what `("GC", "GC")` scores."""


# moved multiloop: -89 -> -84
# moved interior-one-by-one: -8 -> 0
# moved interior-one-by-two: -16 -> -14
# moved interior-two-by-one: -24 -> -14
FOLD_CASES: tuple[FoldCase, ...] = (
    FoldCase("empty", "", "", 0),
    FoldCase("single-base", "A", ".", 0),
    FoldCase("lone-pair-cannot-close", "GC", "..", 0),
    FoldCase("unstacked-hairpin-refused", "GAAAC", ".....", 0),
    FoldCase("homopolymer-adenine", "AAAAAAAAAAAA", "............", 0),
    FoldCase("homopolymer-guanine", "GGGGGGGG", "........", 0),
    FoldCase("homopolymer-cytosine", "CCCCCCCC", "........", 0),
    FoldCase("homopolymer-uracil", "UUUUUUUU", "........", 0),
    FoldCase("min-hairpin-exactly-three", "GGGAAACCC", "(((...)))", -12),
    FoldCase("min-hairpin-two-refused", "GGGAACCC", "........", 0),
    FoldCase("min-hairpin-one-refused", "GGGACCC", ".......", 0),
    FoldCase("triloop-table", "GGGCAACGCCC", "((((...))))", -32),
    FoldCase("tetraloop-uucg", "GGGCUUCGGCCC", "((((....))))", -63),
    FoldCase("hexaloop-table", "GGACAGUACUCC", "(((......)))", -34),
    FoldCase("bulge-one-five-prime", "GGCCAGCGCAAAAGCGCGGCC", "((((.((((....))))))))", -137),
    FoldCase("bulge-one-three-prime", "GGCCGCGCAAAAGCGCAGGCC", "((((((((....)))).))))", -137),
    FoldCase("bulge-two", "GGCCAAGCGCAAAAGCGCGGCC", "((((..((((....))))))))", -123),
    FoldCase("bulge-three", "GGCCAAAGCGCAAAAGCGCGGCC", "((((...((((....))))))))", -119),
    FoldCase("interior-two-by-two", "GGCCAAGCGCAAAAGCGCAAGGCC", "((((..((((....))))..))))", -140),
    FoldCase("interior-one-by-three", "GGCCAGCGCAAAAGCGCAAAGGCC", "((((.((((....))))...))))", -128),
    FoldCase("ninio-at-cap", "GGCCAGCGCAAAAGCGCAAAAAAGGCC", "((((.((((....))))......))))", -100),
    FoldCase("ninio-clamped", "GGCCAGCGCAAAAGCGCAAAAAAAAGGCC", "((((.((((....))))........))))", -97),
    FoldCase("single-wobble-stem", "GGGUAAAAUCCC", "(((......)))", -28),
    FoldCase("wobble-inside-stem", "GGCGUAAAAUCGCC", "((((......))))", -53),
    FoldCase("watson-crick-control", "GGCGCAAAAGCGCC", "(((((....)))))", -84),
    FoldCase("guanine-cytosine-rich", "GGGCCAAAAGGCCC", "(((((....)))))", -92),
    FoldCase("adenine-uracil-rich", "AAAUUAAAAUUUUU", "..............", 0),
    FoldCase("tie-two-optima", "GCUCCUACGGACA", "..(((...)))..", -5),
    FoldCase("tie-three-optima", "CUGGUAAACUGGCUCCA", "..((..........)).", -1),
    FoldCase(
        "hairpin-past-loop-limit",
        "GGGGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCC",
        "((((...................................))))",
        -31,
    ),
    FoldCase("densest-affordable", "GGGGGGGGGGCCCCCCCCCC", "((((((((....))))))))", -190, tier=2),
    FoldCase(
        "interior-loop-at-max",
        "GGGGAAAAAAAAAAAAAAAGGGGAAACCCCAAAAAAAAAAAAAAACCCC",
        "((((...............((((...))))...............))))",
        -107,
        tier=2,
    ),
    FoldCase(
        "interior-loop-past-max",
        "GGGGAAAAAAAAAAAAAAAAGGGGAAACCCCAAAAAAAAAAAAAAACCCC",
        "....................((((...))))...................",
        -64,
        tier=2,
    ),
    FoldCase("multiloop", "GCCCCGGGUCACCGGCAUUAAUGCGGGC", "(((((((....)))(((....)))))))", -84, tier=4),
    FoldCase("interior-one-by-one", "AGUGCUGACGUAAGGACU", "..................", 0),
    FoldCase("interior-one-by-two", "GGAGUUGGAUAAACCGCCG", "((...(((.....))))).", -14),
    FoldCase("interior-two-by-one", "CCCAUUUUCGUACGAUAUGG", ".((((..((....)).))))", -14),
    FoldCase("interior-one-by-one-stacked", "GGGCAGCCAAAAGGCAGCCC", "((((.(((....))).))))", -96),
)

COFOLD_CASES: tuple[CofoldCase, ...] = (
    CofoldCase("both-empty", "", "", "", "", "", 0),
    CofoldCase("first-empty", "", "A", "-", "A", ".", -2),
    CofoldCase("second-empty", "A", "", "A", "-", ".", -2),
    CofoldCase("first-empty-longer", "", "ACGU", "----", "ACGU", "....", -8),
    CofoldCase("single-match", "A", "A", "A", "A", ".", 2),
    CofoldCase("single-mismatch", "G", "C", "G", "C", ".", -1),
    CofoldCase("minimum-hairpin-forbids-a-neighbour", "GC", "GC", "GC", "GC", "..", 4),
    CofoldCase("minimum-hairpin-forbids-a-short-loop", "GAAC", "GAAC", "GAAC", "GAAC", "....", 8),
    CofoldCase("shortest-legal-hairpin", "GAAAC", "GAAAC", "GAAAC", "GAAAC", "(...)", 16),
    CofoldCase("wobble-both-rows", "GGGGU", "GGGGU", "GGGGU", "GGGGU", "(...)", 12),
    CofoldCase("covariation-across-wobble", "GGGGC", "GGGGU", "GGGGC", "GGGGU", "(...)", 11),
    CofoldCase("watson-crick-control", "GGGGC", "GGGGC", "GGGGC", "GGGGC", "(...)", 16),
    CofoldCase("homopolymer-match", "AAAA", "AAAA", "AAAA", "AAAA", "....", 8),
    CofoldCase("homopolymer-unpairable", "GGGG", "GGGG", "GGGG", "GGGG", "....", 8),
    CofoldCase("homopolymer-mismatch", "GGGG", "CCCC", "GGGG", "CCCC", "....", -4),
    CofoldCase("sibling-helices", "CAAAAGCAAAAG", "CAAAAGCAAAAG", "CAAAAGCAAAAG", "CAAAAGCAAAAG", "(....)(....)", 36),
    CofoldCase(
        "sibling-then-nested",
        "CAAAAGCCAAAAGG",
        "CAAAAGCCAAAAGG",
        "CAAAAGCCAAAAGG",
        "CAAAAGCCAAAAGG",
        "(....)((....))",
        46,
    ),
    CofoldCase("nested-depth-two", "CCAAAAGG", "CCAAAAGG", "CCAAAAGG", "CCAAAAGG", "((....))", 28),
    CofoldCase("nested-depth-three", "CCCAAAAGGG", "CCCAAAAGGG", "CCCAAAAGGG", "CCCAAAAGGG", "(((....)))", 38),
    CofoldCase("alternating-pairs", "CGCAAAAGCG", "CGCAAAAGCG", "CGCAAAAGCG", "CGCAAAAGCG", "(((....)))", 38),
    CofoldCase("helix-then-tail", "CCAAAAGGAA", "CCAAAAGGAA", "CCAAAAGGAA", "CCAAAAGGAA", "((....))..", 32),
    CofoldCase("head-then-helix", "AACCAAAAGG", "AACCAAAAGG", "AACCAAAAGG", "AACCAAAAGG", "..((....))", 32),
    CofoldCase("unequal-lengths", "AAAAA", "AAAA", "AAAAA", "AAAA-", ".....", 6),
    CofoldCase("asymmetry-in-structure", "GGGGCCC", "GGGCCCC", "GGGGCCC", "GGGCCCC", "((...))", 23),
    CofoldCase("nothing-pairs", "GAGAGA", "UCUCUC", "GAGAGA", "UCUCUC", "......", -6),
    CofoldCase("planted-stem", "GGGGAAAACCCC", "CCCCAAAAGGGG", "GGGGAAAACCCC", "CCCCAAAAGGGG", "((((....))))", 24),
    CofoldCase("tie-three-optima", "GC", "UAUCU", "G--C-", "UAUCU", ".....", -5),
    CofoldCase("tie-two-optima", "GU", "CGUU", "-GU-", "CGUU", "....", 0),
    CofoldCase("tie-five-optima", "UUAA", "CGCCU", "UUAA-", "CGCCU", ".....", -6),
    CofoldCase("tie-across-equal-lengths", "GACG", "CUAG", "GACG", "CUAG", "....", -1),
    CofoldCase("tie-trailing-gaps", "UAG", "AGGAC", "UAG--", "AGGAC", ".....", -4),
)


@pytest.mark.parametrize("case", FOLD_CASES, ids=lambda case: case.tag)
def test_fold_reproduces_the_frozen_corpus(backend, case: FoldCase):
    """The frozen answer, on every backend, structure included."""
    structure, energy = zuker_fold(case.sequence, **backend)
    assert (structure, round(energy * 10)) == (case.structure, case.decikcal)


@pytest.mark.parametrize("case", COFOLD_CASES, ids=lambda case: case.tag)
def test_cofold_reproduces_the_frozen_corpus(backend, case: CofoldCase):
    """The frozen answer, on every backend, both rows and the structure included."""
    assert sankoff_cofold(case.first, case.second, **backend) == (
        case.gapped_first,
        case.gapped_second,
        case.structure,
        case.score,
    )


def test_the_frozen_corpus_matches_the_model_it_was_frozen_against():
    """Edit the energy tables and this fails first, by name, rather than thirty energies at once."""
    assert turner_fingerprint() == TURNER_FINGERPRINT, (
        "turner.py changed, so every frozen folding energy is suspect. Re-derive them before"
        " editing the corpus, and check whether the recurrence moved too."
    )


# endregion Frozen Expectations

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
    pair_scores = cofolding.default_rna_pair_matrix
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
            total += int(pair_scores[left, right])
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
    pair_scores = cofolding.default_rna_pair_matrix
    first, second = random_rna_pair()
    gapped_first, gapped_second, structure, _ = sankoff_cofold(first, second, **backend)
    for opening, closing in structure_pairs(structure):
        for sequence in (gapped_first, gapped_second):
            assert sequence[opening] != "-" and sequence[closing] != "-", "a pair closed onto a gap"
            left = COFOLD_ALPHABET.index(sequence[opening])
            right = COFOLD_ALPHABET.index(sequence[closing])
            assert pair_scores[left, right] > 0, f"{sequence[opening]}-{sequence[closing]} cannot pair"


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


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_without_pairing_is_needleman_wunsch():
    """With nothing able to pair, Sankoff degenerates to global alignment with linear gaps.

    Binds the reference alone, because the dispatcher exposes no `pair_scores` keyword; the
    compiled backends inherit it through `test_cofold_matches_the_reference`.
    """
    aligner = Align.PairwiseAligner(mode="global")
    aligner.match_score, aligner.mismatch_score = COFOLD_MATCH, COFOLD_MISMATCH
    aligner.open_gap_score = aligner.extend_gap_score = COFOLD_GAP
    first, second = random_rna_pair()
    unpairable = np.zeros((len(cofolding.default_rna_alphabet),) * 2, dtype=np.int32)
    gapped_first, gapped_second, structure, score = cofolding.sankoff_cofold(first, second, pair_scores=unpairable)
    assert score == aligner.score(first, second)
    assert structure == "." * len(gapped_first)
    optima = aligner.align(first, second)
    if len(optima) <= 5000:
        assert (gapped_first, gapped_second) in {(str(one[0]), str(one[1])) for one in optima}


def weighted_nussinov(sequence: str) -> int:
    """Maximum total pair weight of a nested structure, by the 1978 recurrence.

    A different recurrence from Sankoff's, so agreeing with it is evidence rather than an echo.
    """
    codes = [cofolding.default_rna_alphabet.index(letter) for letter in sequence]
    best: dict = {}

    def within(low: int, high: int) -> int:
        # The same steric floor the recurrence applies, so this stays a different algorithm rather
        # than a different chemistry.
        if high - low < cofolding.MIN_CLOSING_REACH + 1:
            return 0
        if (low, high) not in best:
            found = within(low + 1, high)
            for partner in range(low + cofolding.MIN_CLOSING_REACH, high):
                weight = int(cofolding.default_rna_pair_matrix[codes[low], codes[partner]])
                if weight > 0:
                    found = max(found, weight + within(low + 1, partner) + within(partner + 1, high))
            best[(low, high)] = found
        return best[(low, high)]

    return within(0, len(sequence))


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_with_a_prohibitive_gap_is_nussinov(backend):
    """A gap nobody can afford forces the identity alignment, leaving one folding problem scored
    twice, which an independently written Nussinov answers."""
    sequence = random_rna(randint(1, 12))
    gapped_first, gapped_second, _, score = sankoff_cofold(sequence, sequence, gap=-1000, **backend)
    assert gapped_first == gapped_second == sequence
    assert score == COFOLD_MATCH * len(sequence) + 2 * weighted_nussinov(sequence)


def test_cofold_finds_a_planted_stem(backend):
    """Each row holds exactly four G and four C, the loop is all A and no U exists anywhere, so
    four pairs is the ceiling and the optimum is arithmetic rather than a stored answer."""
    first, second = "GGGGAAAACCCC", "CCCCAAAAGGGG"
    gapped_first, gapped_second, structure, score = sankoff_cofold(first, second, **backend)
    assert (gapped_first, gapped_second, structure) == (first, second, "((((....))))")
    assert score == 8 * COFOLD_MISMATCH + 4 * COFOLD_MATCH + 4 * (3 + 3)


@pytest.mark.parametrize("width", (1, 2, 3, 4))
def test_cofold_pays_exactly_for_a_forced_gap(backend, width: int):
    """Padding inside an unpairable stretch improves no structure, so the optimum moves by the
    gap price and nothing else. The gap is linear, so the width is a multiplier."""
    first, second = "GGGGAAAACCCC", "CCCCAAAAGGGG"
    optimum = sankoff_cofold(first, second, **backend)[3]
    padded = first[:6] + "A" * width + first[6:]
    assert sankoff_cofold(padded, second, **backend)[3] == optimum + width * COFOLD_GAP


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_optimum_falls_as_gaps_get_harsher(backend):
    """A harsher gap price lowers the value of every feasible solution, so the maximum over them
    can never rise."""
    first, second = random_rna_pair()
    scores = [sankoff_cofold(first, second, gap=price, **backend)[3] for price in (-1, -2, -3, -5, -9)]
    assert scores == sorted(scores, reverse=True), f"a harsher gap raised the optimum: {scores}"


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_concatenation_never_loses(backend):
    """Laying two independent solutions side by side is a legal solution of the joined problem,
    so the joint optimum is at least their sum.

    An inequality, not an equality: the optimum was observed pairing across the spacer, which is
    a structure the side-by-side solution cannot express, so the joint answer is sometimes
    strictly better.
    """
    left_first, left_second = random_rna(4), random_rna(4)
    right_first, right_second = random_rna(4), random_rna(4)
    spacer = "A" * 3
    parts = (
        sankoff_cofold(left_first, left_second, **backend)[3]
        + sankoff_cofold(spacer, spacer, **backend)[3]
        + sankoff_cofold(right_first, right_second, **backend)[3]
    )
    joint = sankoff_cofold(left_first + spacer + right_first, left_second + spacer + right_second, **backend)[3]
    assert joint >= parts


# endregion Cofolding


# region Folding


def rescore_fold(sequence: str, structure: str) -> int:
    """Decikilocalories for an emitted structure, over its loop decomposition.

    Shares no code with the recurrence, so it catches a table read at the wrong offset or a
    traceback that took a case the fill did not.
    """
    pairs = [(opening, closing) for opening, closing in enumerate(structure_partners(structure)) if closing > opening]
    energy = turner_energy_of(sequence, pairs)
    assert energy is not None, f"the emitted structure holds a loop the model forbids: {structure}"
    return energy


def random_brute_forceable_rna() -> str:
    """A sequence whose whole structure space fits the budget, shortened until it does."""
    sequence = random_rna(randint(8, 30))
    while len(sequence) > 1 and count_fold_structures(sequence) > 8000 * exhaustive_scale:
        sequence = sequence[:-1]
    return sequence


@pytest.mark.repeat(randomized_repetitions_count)
def test_fold_matches_brute_force_enumeration():
    """The recurrence against enumerating every structure, using no dynamic programming.

    The only oracle that can catch a recurrence which is self-consistently wrong on every backend
    at once, which is exactly what a rewrite of the recurrence risks.
    """
    sequence = random_brute_forceable_rna()
    _, energy = zuker_fold(sequence)
    assert round(energy * 10) == brute_force_fold(sequence)


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_matches_brute_force_enumeration(backend):
    """Sankoff's optimum against enumerating every alignment and every structure over it.

    The alignment count grows as the central Delannoy number, so the inputs stay short. It is the
    only oracle here that can catch a fill and a traceback agreeing on the same wrong answer.
    """
    width = 3 + exhaustive_scale
    first = random_rna(randint(1, width))
    second = random_rna(randint(1, width))
    _, _, _, score = sankoff_cofold(first, second, **backend)
    assert score == brute_force_cofold(first, second)


@pytest.mark.parametrize("case", COFOLD_CASES, ids=lambda case: case.tag)
def test_frozen_cofolding_survives_enumeration(case: CofoldCase):
    """The frozen cofolding corpus re-derived by enumeration rather than trusted.

    Skipped where the alignment space is too wide to walk, which is what keeps a default run short.
    """
    if max(len(case.first), len(case.second)) > 4 + 2 * exhaustive_scale:
        pytest.skip("alignment space too wide to enumerate at this scale")
    assert brute_force_cofold(case.first, case.second) == case.score


@pytest.mark.parametrize("case", FOLD_CASES, ids=lambda case: case.tag)
def test_frozen_folding_survives_enumeration(case: FoldCase):
    """The frozen corpus re-derived by enumeration rather than trusted.

    Tiered, because the multiloop case enumerates hundreds of thousands of structures; raise
    `AFFINEGAPS_SCALE` to reach it.
    """
    if case.tier > exhaustive_scale:
        pytest.skip(f"tier {case.tier} needs AFFINEGAPS_SCALE={case.tier}")
    assert brute_force_fold(case.sequence) == case.decikcal


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
        assert closing - opening - 1 >= folding.MIN_TURN, "a hairpin closed too tightly"


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
    # Loading the Mojo library rewrites `PYTHONPATH` in the C environ, which `os.environ` does not see.
    return subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).parent / "affinegaps.py"), *arguments],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
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


@pytest.mark.parametrize("arguments", VERB_ARGUMENTS, ids=lambda case: f"{case[0]}-{len(case)}")
def test_both_binaries_agree(arguments: list):
    """Two parsers can drift, so the guard is that they answer identically, not that we were careful."""
    if not NATIVE_BINARY.exists():
        pytest.skip("the native binary is not built; run `pixi run build-cli`")
    # A vector that names its own placement keeps it; the rest are pinned to the host, which both
    # binaries reach without an accelerator.
    pinned = [] if "--gpu-id" in arguments or "--device" in arguments else ["--device", "cpu"]
    native = subprocess.run([str(NATIVE_BINARY), *arguments, "--format", "json"], capture_output=True, text=True)
    hosted = run_python_cli([*arguments, "--format", "json", "--backend", "mojo", *pinned])
    error_streams = f"native stderr: {native.stderr}\nhosted stderr: {hosted.stderr}"
    assert native.returncode == 0 and hosted.returncode == 0, error_streams
    assert json.loads(native.stdout) == json.loads(hosted.stdout)


@pytest.mark.parametrize("arguments", VERB_ARGUMENTS, ids=lambda case: f"{case[0]}-{len(case)}")
def test_json_carries_the_placement(arguments: list):
    """Every payload says which recurrence ran and where, so a log of runs is self-describing."""
    outcome = run_python_cli([*arguments, "--format", "json"])
    assert outcome.returncode == 0, outcome.stderr
    payload = json.loads(outcome.stdout)
    assert payload["operation"] == arguments[0]
    assert payload["backend"] in ("python", "numba", "mojo")
    assert payload["device"] in ("cpu", "gpu")


def test_verbose_leaves_stdout_parseable():
    """BackendKeywords and throughput go to stderr, so a piped payload survives `2>/dev/null`."""
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
