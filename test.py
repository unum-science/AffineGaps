#!/usr/bin/env python3
"""
Test suite for affine gap alignment, folding and cofolding.

Every property that belongs to an algorithm is asserted against each backend, so the NumPy
reference and the compiled kernels are held to one standard rather than compared after the fact.
A test opts into that axis by naming a `backend` argument; `pytest_generate_tests` supplies the
values and the fixture skips the ones the machine cannot serve.

The suite leans on three kinds of oracle:

- __Self-consistency__, where the traceback's score must equal the score-only kernel's.
- __Cross-implementation__, where every backend must return what the reference returns.
- __External__, where BioPython, ViennaRNA and brute-force enumeration answer independently.

The third kind matters most: the first two would agree with each other while both being wrong.
"""

# pyright: reportArgumentType=false, reportAssignmentType=false, reportIndexIssue=false
# pyright: reportReturnType=false

import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from functools import cache, lru_cache
from importlib.util import find_spec
from itertools import chain, product
from random import Random, choice, randint, seed as random_seed
from typing import NamedTuple

import numpy as np
import pytest
from Bio import Align
from Bio.Align import substitution_matrices

try:
    import RNA
except ImportError:
    RNA = None

import affinegaps
import cofolding
import folding
import turner
from affinegaps import (
    AffineGapCosts,
    Backend,
    Background,
    Device,
    Placement,
    TabulatedSubstitutionCosts,
    UniformSubstitutionCosts,
    available,
    colorize_alignment,
    default_proteins_alphabet,
    default_proteins_costs,
    default_proteins_matrix,
    default_proteins_scale,
    gpu_specs,
    levenshtein_alignment,
    needleman_wunsch_gotoh_alignment,
    needleman_wunsch_gotoh_alignments,
    needleman_wunsch_gotoh_score,
    needleman_wunsch_gotoh_scores,
    sankoff_cofold,
    smith_waterman_gotoh_alignment,
    smith_waterman_gotoh_alignments,
    smith_waterman_gotoh_score,
    smith_waterman_gotoh_scores,
    zuker_fold,
)

# region Harness

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent
"""Where the sources, `cli.mojo` and the `build/` directory live."""

NATIVE_BINARY = REPOSITORY_ROOT / "build" / "affinegaps"
"""The compiled command-line tool, which only some tests need."""

# Keyed by both axes. Naming a key just "gpu" would be ambiguous the moment a second backend grows
# a device path, and Numba already has one in `numba.cuda`.
ALL_BACKENDS = {
    "python-cpu": (Backend.PYTHON, Device.CPU),
    "numba-cpu": (Backend.NUMBA, Device.CPU),
    "mojo-cpu": (Backend.MOJO, Device.CPU),
    "mojo-gpu": (Backend.MOJO, Device.GPU),
}
"""Every placement the suite knows how to exercise, as members rather than as their spellings."""

# `skipif` is resolved by pytest itself, so these need no hook and no `conftest.py`.
needs_gpu = pytest.mark.skipif(
    not available(Backend.MOJO, Device.GPU), reason="this machine has no accelerator the compiled backend can reach"
)
"""Declares a test that cannot run without a reachable accelerator."""
needs_native_binary = pytest.mark.skipif(
    not NATIVE_BINARY.exists(), reason="the native binary is not built; run `pixi run build-cli`"
)
"""Declares a test that shells out to `build/affinegaps` rather than to the Python entry point."""
needs_viennarna = pytest.mark.skipif(RNA is None, reason="ViennaRNA is not installed; it ships in the `test` group")
"""Declares a test that cannot run without the external folding oracle."""
needs_mojo_cpu = pytest.mark.skipif(
    not available(Backend.MOJO, Device.CPU), reason="the compiled backend is not built here"
)
"""Declares a test that asks the compiled backend to refuse something."""
needs_colorama = pytest.mark.skipif(
    find_spec("colorama") is None, reason="colorama is not installed; it ships in the `color` extra"
)
"""Declares a test that paints an alignment rather than only scoring one."""

randomized_repetitions_count: int = int(os.environ.get("AFFINEGAPS_REPETITIONS", "10"))
"""How many times a randomised test runs. Override with `AFFINEGAPS_REPETITIONS`."""

exhaustive_scale: int = max(1, int(os.environ.get("AFFINEGAPS_SCALE", "1")))
"""Multiplier on every exhaustive oracle's budget.

Depth, where `AFFINEGAPS_REPETITIONS` is breadth. `CONTRIBUTING.md` argues why one number
cannot express both.
"""

SESSION_SEED: int = int(os.environ.get("AFFINEGAPS_SEED", "0"))
"""Base seed for every draw. `AFFINEGAPS_SEED=$RANDOM` explores elsewhere and names how to return."""


def requested_backends() -> list:
    """Which placements to exercise, narrowed by `AFFINEGAPS_BACKENDS` when it is set."""
    names = [name.strip() for name in os.environ.get("AFFINEGAPS_BACKENDS", "").split(",") if name.strip()]
    if not names:
        return list(ALL_BACKENDS)
    unknown = set(names) - set(ALL_BACKENDS)
    if unknown:
        raise pytest.UsageError(f"Unknown backend(s): {', '.join(sorted(unknown))}")
    return names


def pytest_generate_tests(metafunc):
    """Supplies the backend axis to any test that names it.

    One of the few hooks a test module may own, which is why this file needs no `conftest.py`.
    """
    chosen = requested_backends()
    for argument, choices in (
        ("backend", chosen),
        ("compiled_backend", [name for name in chosen if not name.startswith("python")]),
    ):
        if argument in metafunc.fixturenames:
            metafunc.parametrize(argument, choices, indirect=True)


def _placement_keywords(name: str) -> dict:
    """The keywords selecting one placement, skipping when this machine cannot serve it."""
    backend, device = ALL_BACKENDS[name]
    if not available(backend, device):
        pytest.skip(f"the {name} backend does not run here; see the README for how to build it")
    return {"backend": backend, "device": device}


@pytest.fixture
def backend(request):
    """Placement keywords for one backend, across every implementation."""
    return _placement_keywords(request.param)


@pytest.fixture
def compiled_backend(request):
    """The same, restricted to the backends that compile."""
    return _placement_keywords(request.param)


@pytest.fixture(autouse=True)
def seed_rng(__pytest_repeat_step_number: int) -> int:
    """Seeds the generator per repeat step, so every backend at one step is handed one input.

    Sharing the draw is what lets the reference be memoized, and what turns a cross-backend
    disagreement from something found by luck into something found by construction.
    """
    seed = SESSION_SEED + (__pytest_repeat_step_number or 0)
    random_seed(seed)
    return seed


# endregion Harness

# region Corpora

RNA_ALPHABET = "ACGU"
"""The four bases every folding and cofolding draw is made of."""


def random_rna(length: int) -> str:
    """One RNA sequence of exactly this length."""
    return "".join(choice(RNA_ALPHABET) for _ in range(length))


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


# endregion Corpora

# region Library Surface


def taller_than_the_band() -> int:
    """A first-sequence length no block here can carry, so only the tiled sweep can serve it.

    Asked of the card rather than written down, because a wider accelerator would otherwise leave
    these tests quietly measuring the banded path they exist to bypass.
    """
    specs = gpu_specs()
    if specs is None:
        return 40_000
    return (specs.shared_memory_per_multiprocessor - specs.reserved_memory_per_block) // 4 + 1


def test_default_table_and_alphabet_agree():
    """The shipped pair must satisfy the record the package offers for carrying a table.

    Both are public, and the documented way to start from the default and tweak it is to hand them
    to `TabulatedSubstitutionCosts` — which checks the shape precisely because NumBa indexes with
    bounds checking off.
    """
    assert default_proteins_matrix.shape == (len(default_proteins_alphabet), len(default_proteins_alphabet))
    rebuilt = TabulatedSubstitutionCosts(default_proteins_alphabet, default_proteins_matrix)
    assert rebuilt.alphabet == default_proteins_costs.alphabet
    assert np.array_equal(rebuilt.matrix, default_proteins_costs.matrix)


def test_compiled_table_matches_the_reference():
    """The two default tables are written out separately, so only a direct comparison ties them.

    Every other cross-check goes through BioPython, which is optional, so without this a drift in
    either table survives wherever that oracle is absent.
    """
    module = affinegaps._mojo_backend()
    if module is None:
        pytest.skip("the compiled backend is not built here")
    letters = len(default_proteins_alphabet)
    compiled = np.array(module.proteins_matrix(), dtype=np.int64).reshape(letters, letters)
    assert np.array_equal(compiled, default_proteins_matrix.astype(np.int64))


def test_specs_report_the_machine():
    """The reported specs must describe a real accelerator, or say plainly there is none to ask."""
    specs = gpu_specs()
    if not available(Backend.MOJO, Device.GPU):
        assert specs is None
        return
    assert specs is not None
    assert specs.shared_memory_per_multiprocessor > specs.reserved_memory_per_block
    assert specs.largest_allocation > 0
    assert specs.streaming_multiprocessors > 0
    # The tall pairs above are built from these, so a length that did not clear the band would
    # leave those tests measuring the banded path rather than the fallback.
    assert taller_than_the_band() < specs.shared_memory_per_multiprocessor


@pytest.mark.parametrize(
    "request_it, refused",
    [
        pytest.param(
            lambda: needleman_wunsch_gotoh_score("AR", "RA", backend="python", device="gpu"),
            ValueError,
            id="a-reference-backend-on-an-accelerator",
        ),
        pytest.param(
            lambda: needleman_wunsch_gotoh_score("AR", "RA", backend="CPU"),  # type: ignore[arg-type]
            ValueError,
            id="a-device-named-as-a-backend",
        ),
        pytest.param(
            lambda: needleman_wunsch_gotoh_score(
                "AR",
                "RA",
                substitution=TabulatedSubstitutionCosts("AR", np.zeros((2, 2), dtype=np.int8)),
                backend="mojo",
            ),
            NotImplementedError,
            marks=needs_mojo_cpu,
            id="a-table-the-compiled-kernels-cannot-take",
        ),
    ],
)
def test_rejects_what_it_cannot_do(request_it, refused):
    """The dispatcher must refuse rather than quietly doing something else."""
    with pytest.raises(refused):
        request_it()


@pytest.mark.parametrize(
    "build, refused",
    [
        pytest.param(lambda: UniformSubstitutionCosts(match=5), TypeError, id="half-a-record"),  # type: ignore[call-arg]
        pytest.param(lambda: AffineGapCosts(open=-1, extend=-20), ValueError, id="opening-cheaper-than-extending"),
        pytest.param(lambda: AffineGapCosts(open=-1, extend=1), ValueError, id="a-gap-that-pays-to-widen"),
        pytest.param(
            lambda: TabulatedSubstitutionCosts("AR", np.zeros((3, 3), dtype=np.int8)),
            ValueError,
            id="a-table-its-alphabet-cannot-index",
        ),
        pytest.param(lambda: Placement(device=Device.CPU, gpu_id=1), ValueError, id="an-index-the-device-cannot-reach"),
        pytest.param(
            lambda: Placement(device=Device.GPU, gpu_id=-1), ValueError, id="an-accelerator-that-cannot-exist"
        ),
        pytest.param(lambda: Placement(threads=0), ValueError, id="a-parallel-region-with-nobody-in-it"),
    ],
)
def test_costs_cannot_express_a_contradiction(build, refused):
    """The pairing rules that once needed runtime checks are carried by the types."""
    with pytest.raises(refused):
        build()


@needs_colorama
def test_colouring_keeps_the_rows_it_paints():
    """Painting adds escapes and changes nothing else, and ragged rows are refused."""
    first, second = "GIVEQ", "GI-EQ"
    painted = {}
    for background in (Background.DARK, Background.LIGHT):
        left, right = colorize_alignment(first, second, background=background)
        assert (re.sub(r"\033\[[0-9;]*m", "", left), re.sub(r"\033\[[0-9;]*m", "", right)) == (first, second)
        painted[background] = left
    assert painted[Background.DARK] != painted[Background.LIGHT], "the gap colour ignored the background"
    with pytest.raises(ValueError):
        colorize_alignment("GIVEQ", "GI")


def test_the_version_is_declared_once():
    """Three files carry the version, and a release that moves one must move all three."""
    declared = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert affinegaps.__version__ == declared
    assert (REPOSITORY_ROOT / "VERSION").read_text().strip() == declared


# endregion Library Surface

# region Alignment

# The substitution table is BLOSUM62 scaled by five, so anything compared against it must be
# scaled to match or the comparison is vacuous.
BLOSUM_SCALE = default_proteins_scale


class Recurrence(NamedTuple):
    """The four entry points one alignment mode offers, single pair then batched.

    Scoring and traceback are separate kernels in every compiled backend, so one can disagree
    with the other and both must be reachable from a test that has only the mode.
    """

    align: Callable
    score: Callable
    align_batch: Callable
    score_batch: Callable


MODES = {
    "global": Recurrence(
        needleman_wunsch_gotoh_alignment,
        needleman_wunsch_gotoh_score,
        needleman_wunsch_gotoh_alignments,
        needleman_wunsch_gotoh_scores,
    ),
    "local": Recurrence(
        smith_waterman_gotoh_alignment,
        smith_waterman_gotoh_score,
        smith_waterman_gotoh_alignments,
        smith_waterman_gotoh_scores,
    ),
}
"""One record per mode, so a fifth entry point is a field rather than a fifth accessor."""


def costs(match: int, mismatch: int, opening: int, extend: int) -> dict:
    """The two cost records as keyword arguments, so a test can splat one scoring into any call."""
    return {
        "substitution": UniformSubstitutionCosts(match=match, mismatch=mismatch),
        "gaps": AffineGapCosts(open=opening, extend=extend),
    }


# One representative scoring per regime rather than a grid: a cheap gap, an expensive one, and
# unit costs. The grids they replace were overlapping draws from the same family.
EXPENSIVE_GAP: dict = costs(5, -4, -20, -1)
SCORINGS = [
    pytest.param(EXPENSIVE_GAP, id="expensive-gap"),
    pytest.param(costs(2, -1, -2, -1), id="cheap-gap"),
    pytest.param(costs(0, -1, -1, -1), id="unit-cost"),
    pytest.param(costs(1, -1, -5, 0), id="free-extension"),
]
SCORING: dict = EXPENSIVE_GAP
"""The scoring a test uses when the scoring is not what it is varying."""


def random_pair(shortest: int = 5, longest: int = 25, alphabet: str = default_proteins_alphabet):
    """Two independent random sequences over the default alphabet."""
    return (
        "".join(choice(alphabet) for _ in range(randint(shortest, longest))),
        "".join(choice(alphabet) for _ in range(randint(shortest, longest))),
    )


def rescore(first: str, second: str, scoring: dict) -> int:
    """Scores a gapped pair under the affine rule, sharing no code with the dynamic programming."""
    total, in_first, in_second = 0, False, False
    for left, right in zip(first, second, strict=True):
        if left == "-" and right == "-":
            continue
        if left == "-":
            total += scoring["gaps"].extend if in_first else scoring["gaps"].open
            in_first, in_second = True, False
        elif right == "-":
            total += scoring["gaps"].extend if in_second else scoring["gaps"].open
            in_first, in_second = False, True
        else:
            total += scoring["substitution"].match if left == right else scoring["substitution"].mismatch
            in_first = in_second = False
    return total


def assert_path_is_well_formed(first: str, second: str, produced: tuple, scoring: dict) -> None:
    """Both rows have one length, each rebuilds a piece of its input, and they earn their score."""
    gapped_first, gapped_second, score = produced
    assert len(gapped_first) == len(gapped_second)
    assert gapped_first.replace("-", "") in first
    assert gapped_second.replace("-", "") in second
    assert rescore(gapped_first, gapped_second, scoring) == score


@lru_cache(maxsize=4096)
def reference_alignment(mode: str, first: str, second: str, substitution, gaps) -> tuple:
    """The Python answer for one input, memoized so the three compiled backends pay for it once."""
    return MODES[mode].align(first, second, substitution=substitution, gaps=gaps, backend="python")


@lru_cache(maxsize=4096)
def reference_score(mode: str, first: str, second: str, substitution, gaps) -> int:
    """The Python score for one input, memoized beside the alignment for the same reason."""
    return MODES[mode].score(first, second, substitution=substitution, gaps=gaps, backend="python")


def biopython_aligner(mode: str, scoring: dict):
    """A BioPython aligner scaled to match our table, so the comparison can actually fail.

    Unscaled, our score would be larger by construction and the equality could never fail.
    """
    aligner = Align.PairwiseAligner(mode=mode)
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62") * BLOSUM_SCALE
    aligner.open_gap_score = scoring["gaps"].open
    aligner.extend_gap_score = scoring["gaps"].extend
    return aligner


def assert_matches_biopython(placement: dict, mode: str, first: str, second: str) -> None:
    """Our score must equal BioPython's, and in global mode our rows must be one of its optima.

    Global mode only, where both tools spell an alignment over the whole of both inputs.
    """
    gaps = {"gaps": AffineGapCosts(open=-20, extend=-1)}
    aligner = biopython_aligner(mode, gaps)
    assert MODES[mode].score(first, second, **gaps, **placement) == aligner.score(first, second)
    if mode != "global":
        return
    gapped_first, gapped_second, _ = MODES[mode].align(first, second, **gaps, **placement)
    optima = aligner.align(first, second)
    # Enumerating a wide tie costs more than the claim is worth, so a wide one is left alone.
    if len(optima) <= 5000:
        assert (gapped_first, gapped_second) in {(str(optimum[0]), str(optimum[1])) for optimum in optima}


@pytest.fixture
def batch_with_an_oversized_pair():
    """One pair no device kernel can take, beside an ordinary one.

    The kernel indexes its carry by the first sequence, so a tall pair fails it even when the
    matrix is small.
    """
    tall = ("A" * taller_than_the_band(), "ACGT" * 4)
    ordinary = random_pair(20, 60)
    return [tall[0], ordinary[0]], [tall[1], ordinary[1]]


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scoring", SCORINGS)
def test_alignment_output_is_well_formed(backend, mode: str, scoring: dict):
    """Every returned path is well formed, realizes its own score, and matches the score-only kernel.

    None of the three is automatic with affine gaps. A walk that reads only the winning operation at
    each cell can leave a gap run and re-enter it, paying a second opening penalty the score never
    did, so the returned rows score less than the number beside them. A local walk has a second
    version of the same trap: it stops at the first non-positive cell, so the untraced prefixes are
    outside the alignment and must not be flushed into the result. And the score-only kernel is a
    separate entry point in every compiled backend, so it can disagree with the traceback beside it.
    """
    first, second = random_pair()
    recurrence = MODES[mode]
    produced = recurrence.align(first, second, **scoring, **backend)
    assert_path_is_well_formed(first, second, produced, scoring)
    assert recurrence.score(first, second, **scoring, **backend) == produced[2]


@pytest.mark.repeat(randomized_repetitions_count)
def test_alignment_symmetry(backend):
    """Swapping the arguments must not change the score."""
    first, second = random_pair()
    assert needleman_wunsch_gotoh_score(first, second, **SCORING, **backend) == (
        needleman_wunsch_gotoh_score(second, first, **SCORING, **backend)
    )


@pytest.mark.repeat(randomized_repetitions_count)
def test_alignment_against_levenshtein(backend):
    """At unit costs the Gotoh recurrence must reduce to edit distance, rows included."""
    first, second = random_pair(3, 15)
    gapped_first, gapped_second, distance = levenshtein_alignment(first, second)
    unit = costs(0, -1, -1, -1)
    assert -needleman_wunsch_gotoh_score(first, second, **unit, **backend) == distance
    assert len(gapped_first) == len(gapped_second)
    assert (gapped_first.replace("-", ""), gapped_second.replace("-", "")) == (first, second)
    assert levenshtein_alignment("", "") == ("", "", 0)


@pytest.mark.repeat(randomized_repetitions_count)
def test_alignment_gap_expansions(backend):
    """A gap that costs nothing to extend must cost the same however wide it is.

    The scoring makes opening a gap cheaper than a mismatch, so the `W` filler is always gapped
    rather than mismatched; without that the width legitimately changes the score.
    """
    free_extension = costs(5, -10, -1, 0)
    first, second = random_pair(5, 15, alphabet="ACGT")
    cut = len(second) // 2

    widths = {
        width: needleman_wunsch_gotoh_score(
            first, second[:cut] + "W" * width + second[cut:], **free_extension, **backend
        )
        for width in range(1, 6)
    }
    assert len(set(widths.values())) == 1, f"a free-extension gap changed price with its width: {widths}"


@pytest.mark.repeat(randomized_repetitions_count)
def test_alignment_optimum_falls_as_gaps_get_harsher(backend):
    """A harsher gap price lowers the value of every feasible alignment, so the maximum cannot rise."""
    first, second = random_pair()
    scores = [
        needleman_wunsch_gotoh_score(first, second, **costs(5, -4, opening, -1), **backend)
        for opening in (-2, -5, -10, -20, -40)
    ]
    assert scores == sorted(scores, reverse=True), f"a harsher gap raised the optimum: {scores}"


@pytest.mark.repeat(randomized_repetitions_count)
def test_alignment_local_never_scores_below_global(backend):
    """A global path is also a local candidate, and the empty one is always available."""
    first, second = random_pair()
    local = smith_waterman_gotoh_score(first, second, **SCORING, **backend)
    assert local >= 0
    assert local >= needleman_wunsch_gotoh_score(first, second, **SCORING, **backend)


@pytest.mark.parametrize("mode", MODES)
def test_alignment_matches_brute_force_enumeration(backend, mode: str):
    """Checks the recurrence against enumerating every alignment, using no dynamic programming.

    Every pair over a two-letter alphabet whose combined length reaches the budget, both modes, on
    every backend. This is the only kind of oracle that can catch a recurrence which is
    self-consistently wrong on every backend at once, which is what rewriting one risks.
    """

    def best_global(first: str, second: str) -> int:
        return max(rescore(top, bottom, SCORING) for top, bottom in enumerate_alignments(first, second))

    def substrings(text: str):
        return (text[start:stop] for start in range(len(text)) for stop in range(start + 1, len(text) + 1))

    def brute_optimal(first: str, second: str) -> int:
        if mode == "global":
            return best_global(first, second)
        # A local alignment is a global one over a substring of each, and the empty one is always
        # available, so no pair of substrings can force the answer below nothing.
        pieces = (best_global(head, tail) for head in substrings(first) for tail in substrings(second))
        return max(chain([0], pieces))

    # The budget `CONTRIBUTING.md` documents, which `AFFINEGAPS_SCALE` deepens.
    budget = 4 + exhaustive_scale
    scorer = MODES[mode].score
    for length in range(4):
        for first in ["".join(letters) for letters in product("AR", repeat=length)] or [""]:
            for second_length in range(4):
                for second in ["".join(letters) for letters in product("AR", repeat=second_length)] or [""]:
                    if len(first) + len(second) <= budget:
                        assert scorer(first, second, **SCORING, **backend) == brute_optimal(first, second)


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
def test_alignment_against_biopython(backend, mode: str, pair: tuple):
    """Curated pairs, scaled to match, so a change that inflates our scores fails here."""
    assert_matches_biopython(backend, mode, *pair)


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("mode", MODES)
def test_alignment_against_biopython_fuzzy(backend, mode: str):
    """The same equality and the same membership, on random proteins rather than curated pairs."""
    assert_matches_biopython(backend, mode, *random_pair(10, 40))


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scoring", SCORINGS)
def test_alignment_matches_the_reference(compiled_backend, mode: str, scoring: dict):
    """A compiled backend must return exactly what the reference returns, strings included."""
    first, second = random_pair()
    assert MODES[mode].align(first, second, **scoring, **compiled_backend) == reference_alignment(
        mode, first, second, **scoring
    )


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scoring", SCORINGS)
def test_alignment_linear_space_matches_stored(compiled_backend, mode: str, scoring: dict, monkeypatch):
    """Both traceback strategies must agree on the score, and each must return a path that earns it.

    The budget constant is the only seam that forces one strategy over the other. Where several
    alignments tie, a divide-and-conquer join need not land where a single backward walk lands, so
    the strings are not required to match.
    """
    first, second = random_pair(shortest=140, longest=260)
    monkeypatch.setattr(affinegaps, "_STORED_MATRIX_BUDGET", 10**12)
    stored = MODES[mode].align(first, second, **scoring, **compiled_backend)
    monkeypatch.setattr(affinegaps, "_STORED_MATRIX_BUDGET", 0)
    linear = MODES[mode].align(first, second, **scoring, **compiled_backend)
    assert linear[2] == stored[2] == reference_score(mode, first, second, **scoring)
    for produced in (stored, linear):
        assert_path_is_well_formed(first, second, produced, scoring)


@pytest.mark.parametrize("mode", MODES)
def test_alignment_linear_space_beats_the_stored_limit(compiled_backend, mode: str, monkeypatch):
    """Linear space must carry a pair far past what a stored matrix could hold."""
    monkeypatch.setattr(affinegaps, "_STORED_MATRIX_BUDGET", 0)
    first = "".join(choice(default_proteins_alphabet) for _ in range(3000))
    second = "".join(choice(default_proteins_alphabet) for _ in range(3000))
    produced = MODES[mode].align(first, second, **SCORING, **compiled_backend)
    assert_path_is_well_formed(first, second, produced, SCORING)


@pytest.mark.parametrize("mode", MODES)
def test_alignment_batch_matches_single_pair(mode: str):
    """The batched entry points must reproduce the reference pair by pair."""
    pairs = [random_pair(5, 40) for _ in range(24)]
    firsts, seconds = [a for a, _ in pairs], [b for _, b in pairs]
    produced = MODES[mode].align_batch(firsts, seconds, **SCORING)
    expected = [reference_alignment(mode, first, second, **SCORING) for first, second in pairs]
    assert produced == expected


@pytest.mark.parametrize("mode", MODES)
def test_alignment_batch_edge_shapes(mode: str):
    """An empty batch answers with nothing, a batch of one answers as the single call does, and
    sides that do not pair up are refused rather than silently truncated."""
    recurrence = MODES[mode]
    reference = {"backend": "python"}
    assert recurrence.align_batch([], [], **SCORING, **reference) == []
    assert recurrence.score_batch([], [], **SCORING, **reference) == []
    first, second = random_pair(10, 30)
    assert recurrence.align_batch([first], [second], **SCORING, **reference) == [
        reference_alignment(mode, first, second, **SCORING)
    ]
    with pytest.raises(ValueError):
        recurrence.score_batch([first, second], [second], **SCORING, **reference)


@pytest.mark.parametrize("mode", MODES)
def test_alignment_batch_survives_a_pair_the_kernel_refuses(compiled_backend, mode: str, batch_with_an_oversized_pair):
    """A pair the stored kernel refuses must take the linear path alone, not sink the batch."""
    firsts, seconds = batch_with_an_oversized_pair
    produced = MODES[mode].align_batch(firsts, seconds, **SCORING, **compiled_backend)
    for first, second, (left, right, score) in zip(firsts, seconds, produced, strict=True):
        # A global path spans both sequences; a local one spans only the core it found.
        core_first, core_second = left.replace("-", ""), right.replace("-", "")
        assert core_first == first if mode == "global" else core_first in first
        assert core_second == second if mode == "global" else core_second in second
        assert rescore(left, right, SCORING) == score


@pytest.mark.parametrize("mode", MODES)
def test_alignment_score_survives_a_pair_the_kernel_refuses(compiled_backend, mode: str, batch_with_an_oversized_pair):
    """Scoring is the cheaper question, so it must not be the more restricted one.

    The tiled sweep that carries the tall pair must agree with the reference, not with the batch
    it left.
    """
    firsts, seconds = batch_with_an_oversized_pair
    produced = MODES[mode].score_batch(firsts, seconds, **SCORING, **compiled_backend)
    expected = [reference_score(mode, first, second, **SCORING) for first, second in zip(firsts, seconds, strict=True)]
    assert produced == expected
    assert MODES[mode].score(firsts[0], seconds[0], **SCORING, **compiled_backend) == expected[0]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "first, second",
    [
        pytest.param("A" * taller_than_the_band(), "", id="tall-against-nothing"),
        pytest.param("", "ACGT" * 4, id="nothing-against-a-sequence"),
        pytest.param("", "", id="nothing-at-all"),
    ],
)
def test_alignment_score_takes_an_empty_side(compiled_backend, mode: str, first: str, second: str):
    """A rectangle with no interior writes no frontier, so its borders come from the costs alone."""
    assert MODES[mode].score(first, second, **SCORING, **compiled_backend) == reference_score(
        mode, first, second, **SCORING
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("reference_backend", ("python", "numba"))
def test_alignment_table_matches_the_uniform_costs_it_spells(mode: str, reference_backend: str):
    """A matrix whose diagonal is the match score must score what the uniform record scores.

    The compiled backends refuse a table outright, so this is the only place a substitution matrix
    is actually used to produce an alignment rather than to trigger a refusal.
    """
    if not available(reference_backend, "cpu"):
        pytest.skip(f"the {reference_backend} backend does not run here")
    alphabet = "ACGT"
    matrix = np.full((len(alphabet), len(alphabet)), -1, dtype=np.int8)
    np.fill_diagonal(matrix, 2)
    first, second = random_pair(10, 40, alphabet=alphabet)
    gaps = AffineGapCosts(open=-5, extend=-1)
    scored = [
        MODES[mode].score(first, second, substitution=table, gaps=gaps, backend=reference_backend)
        for table in (TabulatedSubstitutionCosts(alphabet, matrix), UniformSubstitutionCosts(2, -1))
    ]
    assert scored[0] == scored[1]


# endregion Alignment

# region Folding

PAIRABLE = frozenset({("A", "U"), ("U", "A"), ("C", "G"), ("G", "C"), ("G", "U"), ("U", "G")})
"""The six ordered letter pairs that can close, written out rather than read from `PAIR_INDEX`.

Sharing the table would let a corrupted row be agreed with instead of caught.
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
        # Tabulated with the helix end factored out, as the mismatch tables are.
        return tabulated + helix_end_penalty(slot)
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


def internal_mismatch(sequence: str, slot: int, left: int, right: int) -> int:
    """The stack an internal loop pays across one of its two closing pairs."""
    return int(
        turner.TERMINAL_MISMATCH_INTERNAL[slot, turner.BASES.index(sequence[left]), turner.BASES.index(sequence[right])]
    )


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
    closure = helix_end_penalty(outer) + helix_end_penalty(reversed_inner)
    mismatches = internal_mismatch(sequence, outer, opening + 1, closing - 1) + internal_mismatch(
        sequence, reversed_inner, inner_close + 1, inner_open - 1
    )
    return int(turner.INTERNAL_INITIATION[size]) + asymmetry + closure + mismatches


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

    Shares no code with `folding.py`. Returns nothing when the structure holds a loop the model
    forbids, because a large number in its place is one that stacking could cancel.
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


def rescore_fold(sequence: str, structure: str) -> int:
    """Decikilocalories for an emitted structure, over its loop decomposition."""
    energy = turner_energy_of(sequence, structure_pairs(structure))
    assert energy is not None, f"the emitted structure holds a loop the model forbids: {structure}"
    return energy


def enumerate_fold_structures(sequence: str):
    """Every nested pair set the sequence admits, as opening and closing indices.

    A grammar over pair sets, never a recurrence over energies, so it consults no table.
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


def random_foldable_rna() -> str:
    """One RNA sequence long enough to form a structure worth checking."""
    return random_rna(randint(8, 45))


def random_brute_forceable_rna() -> str:
    """A sequence whose whole structure space fits the budget, shortened until it does."""
    sequence = random_rna(randint(8, 30))
    while len(sequence) > 1 and count_fold_structures(sequence) > 8000 * exhaustive_scale:
        sequence = sequence[:-1]
    return sequence


@lru_cache(maxsize=4096)
def reference_fold(sequence: str) -> tuple:
    """The reference answer for one sequence, memoized so the compiled backends share one copy."""
    return zuker_fold(sequence, backend="python", device="cpu")


def tabulated_hairpins():
    """Every tabulated hairpin, unpacked from its key and closed inside a stem that cannot slip."""
    for keys, size in ((turner.TRILOOP_KEYS, 3), (turner.TETRALOOP_KEYS, 4), (turner.HEXALOOP_KEYS, 6)):
        for key in keys:
            loop = "".join(turner.BASES[int(key) >> (2 * place) & 3] for place in reversed(range(size + 2)))
            yield pytest.param("GGG" + loop + "CCC", "((((" + "." * size + "))))", id=loop)


FOLD_CASES = (
    pytest.param("", "", 0.0, id="nothing-at-all"),
    pytest.param("GC", "..", 0.0, id="no-partner-in-range"),
    pytest.param("GGGG", "....", 0.0, id="nothing-that-can-pair"),
    pytest.param("AAAAAAAAAAAA", "............", 0.0, id="homopolymer"),
    pytest.param("GGGGCAAAAGCCCC", "(((((....)))))", -9.2, id="published-hairpin"),
    pytest.param("AGGGGCAAAAGCCCCU", "((((((....))))))", -10.8, id="published-hairpin-with-flanks"),
)
"""Sequence, structure and energy in kcal per mol.

The two hairpins were checked against RNAstructure's `efn2` term by term; the flat rows are
arithmetic, since no partner is both legal and in range.
"""


@pytest.mark.parametrize("sequence, structure, energy", FOLD_CASES)
def test_fold_reproduces_the_corpus(backend, sequence: str, structure: str, energy: float):
    """The corpus answer, on every backend, structure and energy alike."""
    assert zuker_fold(sequence, **backend) == (structure, energy)


@pytest.mark.parametrize("sequence, structure, energy", FOLD_CASES)
def test_fold_corpus_is_derived_not_trusted(sequence: str, structure: str, energy: float):
    """Every row re-priced off the Turner tables by a scorer that shares no code with `folding.py`."""
    assert rescore_fold(sequence, structure) == round(energy * 10)


@pytest.mark.repeat(randomized_repetitions_count)
def test_fold_matches_brute_force_enumeration():
    """The recurrence against enumerating every structure, using no dynamic programming.

    Every structure is priced straight off the Turner tables by a scorer sharing no code with
    `folding.py`, so a table read at the wrong offset is caught rather than agreed with.
    """
    sequence = random_brute_forceable_rna()
    _, energy = zuker_fold(sequence)
    assert round(energy * 10) == brute_force_fold(sequence)


@pytest.mark.repeat(randomized_repetitions_count)
def test_fold_output_is_well_formed(backend):
    """Brackets balance, every pair is chemically possible, hairpins are not too tight, and the
    reported energy is the model's energy for the structure actually emitted."""
    sequence = random_foldable_rna()
    structure, energy = zuker_fold(sequence, **backend)
    assert len(structure) == len(sequence)
    for opening, closing in enumerate(structure_partners(structure)):
        if closing <= opening:
            continue
        left = folding.default_rna_alphabet.index(sequence[opening])
        right = folding.default_rna_alphabet.index(sequence[closing])
        assert turner.PAIR_INDEX[left, right] >= 0, f"{sequence[opening]}-{sequence[closing]} cannot pair"
        assert closing - opening - 1 >= folding.MIN_TURN, "a hairpin closed too tightly"
    assert rescore_fold(sequence, structure) == round(energy * 10)


@pytest.mark.repeat(randomized_repetitions_count)
def test_fold_matches_the_reference(compiled_backend):
    """Every compiled backend must reproduce the reference exactly, structure included."""
    sequence = random_foldable_rna()
    assert zuker_fold(sequence, **compiled_backend) == reference_fold(sequence)


@needs_viennarna
@pytest.mark.parametrize("sequence, structure", tabulated_hairpins())
def test_fold_model_matches_viennarna_on_tabulated_hairpins(sequence: str, structure: str):
    """A tabulated loop replaces every term the two models spell differently, so they agree to 0.00."""
    assert rescore_fold(sequence, structure) / 10 == pytest.approx(RNA.energy_of_struct(sequence, structure), abs=0.01)


@needs_viennarna
def test_fold_stays_close_to_viennarna():
    """The mean absolute gap to ViennaRNA across a seeded sample must stay under a recorded bound.

    Folding is held to ViennaRNA rather than to its own past answers, because a frozen corpus
    records what the implementation said when it was written and a genuine fix then arrives looking
    like a regression. Both models score the structure this one emitted, so what the bound measures
    is the energy tables rather than a disagreement over which structure wins. The sample follows
    `AFFINEGAPS_REPETITIONS`, and its generator is local so the draw moves no other test's stream.
    """
    generator = Random(42)
    drawn, total = 20 * randomized_repetitions_count, 0.0
    for _ in range(drawn):
        sequence = "".join(generator.choice(RNA_ALPHABET) for _ in range(generator.randint(20, 60)))
        structure, energy = zuker_fold(sequence)
        total += abs(energy - RNA.energy_of_struct(sequence, structure))
    assert total / drawn < 0.5, "the energy model has drifted from ViennaRNA"


# endregion Folding

# region Cofolding

COFOLD_MATCH, COFOLD_MISMATCH, COFOLD_GAP = 2, -1, -2
"""The covariance scoring the recurrence defaults to, restated so a rescore stays independent."""


def random_rna_pair() -> tuple:
    """A pair of RNA sequences short enough for a recurrence that is quartic in memory."""
    return random_rna(randint(1, 12)), random_rna(randint(1, 12))


def pair_weight(left: str, right: str) -> int:
    """What the covariance table pays for one letter pair, zero where it cannot form."""
    return int(cofolding.default_rna_pair_matrix[RNA_ALPHABET.index(left), RNA_ALPHABET.index(right)])


def column_score(left: str, right: str) -> int:
    """What one aligned column pays, before any pairing credit."""
    return COFOLD_MATCH if left == right else COFOLD_MISMATCH


def rescore_cofold(gapped_first: str, gapped_second: str, structure: str) -> int:
    """Scores an emitted alignment independently of the recurrence that produced it."""
    total = 0
    for left, right in zip(gapped_first, gapped_second, strict=True):
        total += COFOLD_GAP if "-" in (left, right) else column_score(left, right)
    for opening, closing in structure_pairs(structure):
        total += sum(pair_weight(row[opening], row[closing]) for row in (gapped_first, gapped_second))
    return total


def pairable_columns(gapped_first: str, gapped_second: str) -> dict:
    """Columns a Sankoff pair may join, with their weight and the per-sequence hairpin floor applied.

    The floor lives in each sequence's own coordinates, not the alignment's, because that is where
    the recurrence measures a helix's reach.
    """
    rows = (gapped_first, gapped_second)
    # Each row's own index for every column it occupies, which is where the floor is measured.
    places = [
        {column: rank for rank, column in enumerate(index for index, letter in enumerate(row) if letter != "-")}
        for row in rows
    ]
    allowed = {}
    for opening in range(len(gapped_first)):
        for closing in range(opening + 1, len(gapped_first)):
            if any(opening not in place or closing not in place for place in places):
                continue
            if any(place[closing] - place[opening] < cofolding.MIN_CLOSING_REACH for place in places):
                continue
            weights = [pair_weight(row[opening], row[closing]) for row in rows]
            if min(weights) > 0:
                allowed[(opening, closing)] = sum(weights)
    return allowed


def brute_force_cofold(first: str, second: str) -> int:
    """The Sankoff optimum over every alignment and every nested structure, by enumeration.

    Shares the pair matrix with the recurrence but not the algorithm.
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


def sankoff_optimum(first: str, second: str) -> int:
    """The Sankoff optimum by an exact recurrence, reaching pairs the enumeration cannot.

    A second algorithm rather than a third-party tool, because no folding engine optimizes
    covariance — every Sankoff implementation prices Turner free energies instead.
    """
    reach = cofolding.MIN_CLOSING_REACH

    @cache
    def paired(opening_first, closing_first, opening_second, closing_second):
        """A helix closing both rows into the same two columns, or nothing where it cannot."""
        if closing_first - opening_first < reach or closing_second - opening_second < reach:
            return None
        left = pair_weight(first[opening_first], first[closing_first])
        right = pair_weight(second[opening_second], second[closing_second])
        if left <= 0 or right <= 0:
            return None
        columns = column_score(first[opening_first], second[opening_second]) + column_score(
            first[closing_first], second[closing_second]
        )
        inside = free(opening_first + 1, closing_first, opening_second + 1, closing_second)
        return columns + left + right + inside

    @cache
    def free(low_first, high_first, low_second, high_second):
        """One slice of each sequence against the other, under any nesting of the columns."""
        # An exhausted side leaves only gaps to pay for, and two exhausted sides cost nothing.
        if low_first >= high_first and low_second >= high_second:
            return 0
        if low_first >= high_first:
            return COFOLD_GAP + free(low_first, high_first, low_second + 1, high_second)
        if low_second >= high_second:
            return COFOLD_GAP + free(low_first + 1, high_first, low_second, high_second)
        best = max(
            column_score(first[low_first], second[low_second])
            + free(low_first + 1, high_first, low_second + 1, high_second),
            COFOLD_GAP + free(low_first + 1, high_first, low_second, high_second),
            COFOLD_GAP + free(low_first, high_first, low_second + 1, high_second),
        )
        for closing_first in range(low_first + reach, high_first):
            for closing_second in range(low_second + reach, high_second):
                helix = paired(low_first, closing_first, low_second, closing_second)
                if helix is not None:
                    best = max(best, helix + free(closing_first + 1, high_first, closing_second + 1, high_second))
        return best

    return free(0, len(first), 0, len(second))


def weighted_nussinov(sequence: str) -> int:
    """Maximum total pair weight of a nested structure, by the 1978 recurrence."""
    best: dict = {}

    def within(low: int, high: int) -> int:
        # The same steric floor the recurrence applies, so the two differ in algorithm, not chemistry.
        if high - low < cofolding.MIN_CLOSING_REACH + 1:
            return 0
        if (low, high) not in best:
            found = within(low + 1, high)
            for partner in range(low + cofolding.MIN_CLOSING_REACH, high):
                weight = pair_weight(sequence[low], sequence[partner])
                if weight > 0:
                    found = max(found, weight + within(low + 1, partner) + within(partner + 1, high))
            best[(low, high)] = found
        return best[(low, high)]

    return within(0, len(sequence))


@lru_cache(maxsize=4096)
def reference_cofold(first: str, second: str) -> tuple:
    """The reference answer for one pair, memoized so the compiled backends share one copy."""
    return sankoff_cofold(first, second, backend="python", device="cpu")


class CofoldCase(NamedTuple):
    """One cofolding expectation, re-derived on every run by two oracles that share no algorithm."""

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
    # Four G and four C in each row, an all-A loop and no U anywhere, so four pairs is the ceiling
    # and the optimum is arithmetic rather than a stored answer.
    CofoldCase("planted-stem", "GGGGAAAACCCC", "CCCCAAAAGGGG", "GGGGAAAACCCC", "CCCCAAAAGGGG", "((((....))))", 24),
    CofoldCase(
        "hairpin-in-both-rows",
        "GGGGCAAAAGCCCC",
        "GGGGCAAAAGCCCC",
        "GGGGCAAAAGCCCC",
        "GGGGCAAAAGCCCC",
        "(((((....)))))",
        58,
    ),
    # GC in one row and AU in the other at the same columns, so the substitution term penalises
    # every stem position while the pairing term rewards it.
    CofoldCase(
        "compensatory-mutation", "GGGGAAAACCCC", "AAAAGGGGUUUU", "GGGGAAAACCCC", "AAAAGGGGUUUU", "((((....))))", 8
    ),
    CofoldCase("tie-three-optima", "GC", "UAUCU", "G--C-", "UAUCU", ".....", -5),
    CofoldCase("tie-two-optima", "GU", "CGUU", "-GU-", "CGUU", "....", 0),
    CofoldCase("tie-five-optima", "UUAA", "CGCCU", "UUAA-", "CGCCU", ".....", -6),
    CofoldCase("tie-across-equal-lengths", "GACG", "CUAG", "GACG", "CUAG", "....", -1),
    CofoldCase("tie-trailing-gaps", "UAG", "AGGAC", "UAG--", "AGGAC", ".....", -4),
)


@pytest.mark.parametrize("case", COFOLD_CASES, ids=lambda case: case.tag)
def test_cofold_reproduces_the_corpus(backend, case: CofoldCase):
    """The corpus answer, on every backend, both rows and the structure included."""
    assert sankoff_cofold(case.first, case.second, **backend) == (
        case.gapped_first,
        case.gapped_second,
        case.structure,
        case.score,
    )


@pytest.mark.parametrize("case", COFOLD_CASES, ids=lambda case: case.tag)
def test_cofold_corpus_is_derived_not_trusted(case: CofoldCase):
    """Every row re-derived rather than trusted, by an exact recurrence that shares no code with
    the implementation, and by enumeration too where the space is narrow enough to walk.

    The second oracle is what lets the widest rows be checked at all: enumerating a fourteen
    against fourteen alignment space grows as the central Delannoy number.
    """
    assert sankoff_optimum(case.first, case.second) == case.score
    if max(len(case.first), len(case.second)) <= 4 + 2 * exhaustive_scale:
        assert brute_force_cofold(case.first, case.second) == case.score


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_matches_brute_force_enumeration(backend):
    """Sankoff's optimum against enumerating every alignment and every structure over it.

    The alignment count grows as the central Delannoy number, so the inputs stay very short.
    """
    width = 3 + exhaustive_scale
    first = random_rna(randint(1, width))
    second = random_rna(randint(1, width))
    assert sankoff_cofold(first, second, **backend)[3] == brute_force_cofold(first, second)


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_matches_an_exact_recurrence(backend):
    """The same optimum from a second algorithm, on pairs the enumeration is far too slow for."""
    first, second = random_rna(randint(1, 14)), random_rna(randint(1, 14))
    assert sankoff_cofold(first, second, **backend)[3] == sankoff_optimum(first, second)


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_output_is_well_formed(backend):
    """Both rows come back unchanged, every credited pair can form in both of them, and the
    reported score is the score of the alignment and structure actually emitted."""
    first, second = random_rna_pair()
    gapped_first, gapped_second, structure, score = sankoff_cofold(first, second, **backend)
    assert len(gapped_first) == len(gapped_second) == len(structure)
    assert gapped_first.replace("-", "") == first
    assert gapped_second.replace("-", "") == second
    for opening, closing in structure_pairs(structure):
        for row in (gapped_first, gapped_second):
            assert row[opening] != "-" and row[closing] != "-", "a pair closed onto a gap"
            assert pair_weight(row[opening], row[closing]) > 0, f"{row[opening]}-{row[closing]} cannot pair"
    assert rescore_cofold(gapped_first, gapped_second, structure) == score


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_symmetry(backend):
    """Swapping the two sequences cannot change the optimum, because the recurrence is symmetric."""
    first, second = random_rna_pair()
    assert sankoff_cofold(first, second, **backend)[3] == sankoff_cofold(second, first, **backend)[3]


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_matches_the_reference(compiled_backend):
    """Every compiled backend must reproduce the reference exactly, structure included."""
    first, second = random_rna_pair()
    assert sankoff_cofold(first, second, **compiled_backend) == reference_cofold(first, second)


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_without_pairing_is_needleman_wunsch():
    """With nothing able to pair, Sankoff degenerates to global alignment with linear gaps.

    One of two limits where the answer comes from outside this project, and the emitted rows must be
    a member of BioPython's own set of optimal alignments rather than merely score the same.
    Binds the reference alone, because the dispatcher exposes no `pair_scores` keyword.
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
        assert (gapped_first, gapped_second) in {(str(optimum[0]), str(optimum[1])) for optimum in optima}


@pytest.mark.repeat(randomized_repetitions_count)
def test_cofold_with_a_prohibitive_gap_is_nussinov(backend):
    """A gap nobody can afford forces the identity alignment, leaving one folding problem scored
    twice, which an independently written Nussinov answers.

    The other limit answered from outside the recurrence, under the same steric floor it applies, so
    the two stay different algorithms rather than different chemistries.
    """
    sequence = random_rna(randint(1, 12))
    gapped_first, gapped_second, _, score = sankoff_cofold(sequence, sequence, gap=-1000, **backend)
    assert gapped_first == gapped_second == sequence
    assert score == COFOLD_MATCH * len(sequence) + 2 * weighted_nussinov(sequence)


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
    """Laying two solutions side by side is legal in the joined problem, so the joint optimum is
    at least their sum.

    An inequality, never an equality: pairing across the spacer is sometimes strictly better.
    """
    spacer = "A" * 3
    pieces = ((random_rna(4), random_rna(4)), (spacer, spacer), (random_rna(4), random_rna(4)))
    parts = sum(sankoff_cofold(first, second, **backend)[3] for first, second in pieces)
    firsts, seconds = zip(*pieces, strict=True)
    joint = sankoff_cofold("".join(firsts), "".join(seconds), **backend)[3]
    assert joint >= parts


# endregion Cofolding

# region Command Line

VERB_SEQUENCES = {
    # `align` scores over the protein alphabet by default; the folding verbs are RNA-only.
    "align": ["GIVEQCCTSICSLY", "HSQGTFTSDYSKYL"],
    "fold": ["GGGGCAAAAGCCCC"],
    "cofold": ["GGGGCAAAAGCCCC", "GGGGCUUUUGCCCC"],
}
"""Inputs each verb accepts, so a flag is exercised against a sequence its alphabet admits."""

VERB_ARGUMENTS = (
    pytest.param(["align", "GIVEQCCTSICSLYQLENYCN", "HSQGTFTSDYSKYLDSRAEQDFV"], id="align-global"),
    pytest.param(["align", "GIVEQ", "HSQGT", "--local"], id="align-local"),
    pytest.param(["align", "GIVEQ", "HSQGT", "--open", "-5", "--extend", "-2"], id="align-affine-gap"),
    pytest.param(["fold", "GGGGCUUCGGCCCC"], id="fold"),
    pytest.param(["cofold", "GGGGCAAAAGCCCC", "GGGGCUUUUGCCCC"], id="cofold"),
    pytest.param(["cofold", "GGGGCAAAAGCCCC", "GGGGCUUUUGCCCC", "--gap", "-3"], id="cofold-linear-gap"),
    pytest.param(["align", "GIVEQ", "HSQGT", "--gpu-id", "0", "--threads", "2"], marks=needs_gpu, id="align-on-gpu"),
    pytest.param(["fold", "GGGGCUUCGGCCCC", "--gpu-id", "0"], marks=needs_gpu, id="fold-on-gpu"),
)
"""Argument vectors both binaries must answer identically."""


@pytest.fixture
def run_cli(capsys):
    """Drives the Python entry point in this interpreter, so a case costs no process start."""

    def invoke(arguments: list) -> tuple[int, str, str]:
        code = 0
        try:
            affinegaps.main(arguments)
        except SystemExit as requested:
            code = requested.code or 0
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return invoke


def run_cli_in_a_process(arguments: list) -> subprocess.CompletedProcess:
    """The same entry point as its own process, for the cases that must own their streams.

    Loading the Mojo library rewrites `PYTHONPATH` in the C environ, which `os.environ` does not see.
    """
    return subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "affinegaps.py"), *arguments],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def advertised_flags(usage: str) -> set:
    """Every long flag a usage string names, which the matching parser must accept."""
    return set(re.findall(r"--[a-z][a-z-]*", usage))


def advertised_pairs() -> list:
    """Every verb and flag `cli.mojo` advertises, one parameter each so a failure names itself."""
    source = (REPOSITORY_ROOT / "cli.mojo").read_text()
    pairs = []
    for verb, constant in (("align", "USAGE_ALIGN"), ("fold", "USAGE_FOLD"), ("cofold", "USAGE_COFOLD")):
        block = re.search(rf'comptime {constant} = """(.*?)"""', source, re.S)
        assert block, f"{constant} is missing from cli.mojo"
        for flag in sorted(advertised_flags(block.group(1)) - {"--help"}):
            marks = needs_gpu if flag == "--gpu-id" else ()
            pairs.append(pytest.param(verb, flag, marks=marks, id=f"{verb}{flag}"))
    return pairs


@pytest.mark.parametrize("verb,flag", advertised_pairs())
def test_every_advertised_flag_is_accepted(run_cli, verb: str, flag: str):
    """The native usage text and the Python parser must accept the same flags."""
    valued = {"--device": "cpu", "--format": "human", "--color": "never", "--gpu-id": "0", "--threads": "1"}
    extra = [flag] if flag in ("--local", "--verbose") else [flag, valued.get(flag, "-1")]
    # A uniform score is half a record, so the two halves are only ever given together.
    if flag in ("--match", "--mismatch"):
        extra = ["--match", "2", "--mismatch", "-1"]
    code, _, errors = run_cli([verb, *VERB_SEQUENCES[verb], *extra])
    assert code == 0, f"{verb} rejected its own advertised {flag}: {errors}"


@pytest.mark.parametrize("arguments", ([], ["align"], ["fold"], ["cofold"]), ids=("top", "align", "fold", "cofold"))
def test_help_is_offered_everywhere(run_cli, arguments: list):
    """Every parser answers `--help` on stdout and exits clean, so no verb is the odd one out."""
    code, printed, _ = run_cli([*arguments, "--help"])
    assert code == 0
    assert "--help" in printed


@pytest.mark.parametrize(
    "arguments,expected",
    [
        pytest.param([], 2, id="no-verb"),
        pytest.param(["nonsense"], 2, id="unknown-verb"),
        pytest.param(["align", "GIVEQ", "HSQGT", "--bogus"], 2, id="unknown-flag"),
        pytest.param(["align", "GIVEQ", "HSQGT", "--match", "5"], 2, id="half-scoring"),
        pytest.param(["align", "GIVEQ", "HSQGT", "--gpu-id", "0", "--device", "cpu"], 2, id="unreachable-index"),
        pytest.param(["align", "GIVEQ", "HSQGT", "--gpu-id", "-1"], 2, id="negative-accelerator"),
        pytest.param(["align", "GIVEQ", "HSQGT", "--threads", "-1"], 2, id="negative-thread-width"),
        pytest.param(["fold", "GGGGXAAAAGCCCC"], 1, id="bad-character"),
        pytest.param(["align", "GIVEQ", "HSQGT"], 0, id="success"),
    ],
)
def test_exit_codes(run_cli, arguments: list, expected: int):
    """Two is a usage error the parser caught; one is a request the library refused."""
    assert run_cli(arguments)[0] == expected


@pytest.mark.parametrize("verb", ("align", "fold", "cofold"))
def test_an_absent_accelerator_is_refused(verb: str):
    """Naming an absent device is the library refusing a request, not a usage error.

    Its own process, because the resolver raises before the handler that turns a refusal into an
    exit code.
    """
    arguments = [verb, *VERB_SEQUENCES[verb], "--backend", "mojo", "--device", "gpu", "--gpu-id", "4096"]
    outcome = run_cli_in_a_process(arguments)
    assert outcome.returncode == 1, outcome.stdout


@needs_native_binary
@pytest.mark.parametrize("arguments", VERB_ARGUMENTS)
def test_both_binaries_agree(run_cli, arguments: list):
    """Two parsers can drift, so the guard is that they answer identically, not that we were careful."""
    # A vector that names its own placement keeps it; the rest are pinned to the host, which both
    # binaries reach without an accelerator.
    pinned = [] if "--gpu-id" in arguments or "--device" in arguments else ["--device", "cpu"]
    native = subprocess.run([str(NATIVE_BINARY), *arguments, "--format", "json"], capture_output=True, text=True)
    code, printed, errors = run_cli([*arguments, "--format", "json", "--backend", "mojo", *pinned])
    streams = f"native stderr: {native.stderr}\nhosted stderr: {errors}"
    assert native.returncode == 0 and code == 0, streams
    assert json.loads(native.stdout) == json.loads(printed)


@needs_native_binary
@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(["nonsense"], id="unknown-verb"),
        pytest.param(["align", "GIVEQ", "HSQGT", "--bogus"], id="unknown-flag"),
        pytest.param(["align", "GIVEQ", "HSQGT", "--match", "5"], id="half-scoring"),
        pytest.param(["align", "GIVEQ", "HSQGT", "--gpu-id", "-1"], id="negative-accelerator"),
        pytest.param(["align", "GIVEQ", "HSQGT", "--gpu-id", "0", "--device", "cpu"], id="unreachable-index"),
        pytest.param(["align", "GIVEQ", "HSQGT", "--threads", "-1"], id="negative-thread-width"),
        pytest.param(["align", "GIVEQ", "HSQGT", "--device", "nonsense"], id="a-device-that-does-not-exist"),
        pytest.param(["cofold", "GGGGCAAAAGCCCC", "GGGGCUUUUGCCCC", "--gap", "x"], id="a-gap-that-is-not-a-number"),
        pytest.param(["fold", "GGGGXAAAAGCCCC"], id="bad-character"),
    ],
)
def test_both_binaries_agree_on_refusals(run_cli, arguments: list):
    """A refusal must carry the same status out of either binary.

    Two is the parser's answer and one is the library's, so a caller scripting either binary reads
    the same number for the same mistake.
    """
    native = subprocess.run([str(NATIVE_BINARY), *arguments], capture_output=True, text=True)
    assert native.returncode == run_cli(arguments)[0], native.stderr


@pytest.mark.parametrize("arguments", VERB_ARGUMENTS)
def test_json_carries_the_placement(run_cli, arguments: list):
    """Every payload says which recurrence ran and where, so a log of runs is self-describing."""
    code, printed, errors = run_cli([*arguments, "--format", "json"])
    assert code == 0, errors
    payload = json.loads(printed)
    assert payload["operation"] == arguments[0]
    assert payload["backend"] in ("python", "numba", "mojo")
    assert payload["device"] in ("cpu", "gpu")


@needs_gpu
@pytest.mark.parametrize("threads", (1, 2, 8))
def test_thread_width_does_not_move_the_answer(run_cli, threads: int):
    """A width is a placement knob, so the traceback it parallelizes must land on the same alignment."""
    first, second = "GIVEQCCTSICSLYQLENYCN" * 8, "HSQGTFTSDYSKYLDSRAEQDFV" * 8
    code, printed, errors = run_cli(
        ["align", first, second, "--threads", str(threads), "--format", "json", "--backend", "mojo", "--device", "gpu"]
    )
    assert code == 0, errors
    payload = json.loads(printed)
    assert payload["threads"] == threads
    assert payload["score"] == json.loads(run_cli(["align", first, second, "--format", "json"])[1])["score"]


def test_verbose_leaves_stdout_parseable(run_cli):
    """The placement and throughput go to stderr, so a piped payload survives `2>/dev/null`."""
    _, printed, errors = run_cli(["fold", "GGGGCAAAAGCCCC", "--format", "json", "--verbose"])
    assert json.loads(printed)["structure"] == "(((((....)))))"
    assert "throughput" in errors


@pytest.mark.parametrize("choice,escaped", (([], False), (["--color", "never"], False), (["--color", "always"], True)))
def test_color_is_decided_by_the_stream(choice: list, escaped: bool):
    """Colour follows whether stdout is a terminal, not whether colorama imports.

    Its own process, because colouring rewraps the streams and would outlive the case.
    """
    printed = run_cli_in_a_process(["align", "GIVEQ", "HSQGT", *choice]).stdout
    assert ("\033" in printed) is escaped


def test_scoring_flags_reach_the_recurrence(run_cli):
    """A scoring the caller names must move the answer, not be parsed and dropped."""
    named = ["--match", "2", "--mismatch", "-1"]
    _, printed, _ = run_cli(["align", "GIVEQ", "HSQGT", *named, "--format", "json"])
    assert json.loads(printed)["score"] == needleman_wunsch_gotoh_score(
        "GIVEQ", "HSQGT", substitution=UniformSubstitutionCosts(2, -1), gaps=AffineGapCosts(), backend="python"
    )


@pytest.mark.parametrize(
    "arguments, key, expected",
    [
        pytest.param(VERB_ARGUMENTS[0].values[0], "score", 22, id="align"),
        pytest.param(["fold", "GGGGCUUCGGCCCC"], "energy_kcal_per_mol", -9.6, id="fold"),
        pytest.param(VERB_ARGUMENTS[4].values[0], "score", 46, id="cofold"),
    ],
)
def test_verbs_reproduce_the_validated_values(run_cli, arguments: list, key: str, expected):
    """The numbers checked against `efn2` and BioPython must survive any change to the surface."""
    assert json.loads(run_cli([*arguments, "--format", "json"])[1])[key] == expected


# endregion Command Line
