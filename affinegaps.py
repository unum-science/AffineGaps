#!/usr/bin/env python3
"""
Affine Gaps Alignment Toolkit

This single-file library and CLI tool provides robust implementations of sequence alignment algorithms,
including Needleman-Wunsch, Smith-Waterman, and Levenshtein, with support for affine gap penalties.
The toolkit is designed for both programmatic use and command-line operation, making it versatile
for bio-informatics, computational biology, and general sequence alignment tasks.

Key Features:
- Global alignment (Needleman-Wunsch with Gotoh extensions)
- Local alignment (Smith-Waterman with Gotoh extensions)
- Edit distance computation (Levenshtein)
- Customizable substitution matrices and gap penalties
- Optimized for performance with optional NumBa acceleration
- CLI for quick alignment and scoring of sequences

Usage:
1. Library:
    Import and use the library functions for programmatic sequence alignment:
    >>> from affinegaps import needleman_wunsch_gotoh_alignment
    >>> first_gapped, second_gapped, score = needleman_wunsch_gotoh_alignment("GATTACA", "GCATGCU")
    >>> print("Alignment 1:", first_gapped)
    >>> print("Alignment 2:", second_gapped)
    >>> print("Score:", score)

2. CLI:
    One verb per recurrence, reached directly from the command line:
    $ affinegaps align GIVEQCCTSICSLYQLENYCN HSQGTFTSDYSKYLDSRAEQDFV --local
    Sequence 1:  GIVEQCCTSICSLYQLENYCN
    Sequence 2:  HSQGTFTSDYSKYLDSRAEQDFV
    Alignment 1: TSICSLYQLEN
    Alignment 2: TSDYSKY-LDS
    Score:       80

Dependencies:
- Python 3.12+
- NumPy (required)
- NumBa (optional, for acceleration)
- colorama (optional, for colored CLI output)

Author: Ash Vardanian
License: Apache 2.0
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache, lru_cache
from typing import Any

from alignment import (
    _needleman_wunsch_gotoh_kernel,
    _needleman_wunsch_gotoh_score_kernel,
    _reconstruct_alignment,
    _smith_waterman_gotoh_kernel,
    _smith_waterman_gotoh_score_kernel,
    _validate_gotoh_arguments,
    colorize_alignment,
    default_proteins_matrix,
    levenshtein_alignment,
)

from cofolding import default_rna_pair_matrix
from cofolding import sankoff_cofold as _sankoff_cofold_reference
from common import (
    HAS_NUMBA,
    AffineGapCosts,
    Backend,
    Device,
    Placement,
    SubstitutionCosts,
    TabulatedSubstitutionCosts,
    UniformSubstitutionCosts,
    _translate_sequence,
    default_proteins_alphabet,
    default_rna_alphabet,
    hardware_threads,
)
from folding import zuker_fold as _zuker_fold_reference

__version__ = "0.2.5"

# Re-exported so the split into per-recurrence modules stays invisible to callers.
__all__ = [
    "AffineGapCosts",
    "Backend",
    "Device",
    "TabulatedSubstitutionCosts",
    "UniformSubstitutionCosts",
    "available",
    "colorize_alignment",
    "default_proteins_alphabet",
    "default_proteins_matrix",
    "default_rna_alphabet",
    "default_rna_pair_matrix",
    "levenshtein_alignment",
    "needleman_wunsch_gotoh_alignment",
    "needleman_wunsch_gotoh_alignments",
    "needleman_wunsch_gotoh_score",
    "needleman_wunsch_gotoh_scores",
    "sankoff_cofold",
    "smith_waterman_gotoh_alignment",
    "smith_waterman_gotoh_alignments",
    "smith_waterman_gotoh_score",
    "smith_waterman_gotoh_scores",
    "zuker_fold",
]


# region Backends

# Above this many cells the traceback switches to the linear-space recursion, which produces
# identical output. Five matrices at seventeen bytes a cell keeps a stored alignment near 100 MB.
_STORED_MATRIX_BUDGET = 6_000_000


class Algorithm(StrEnum):
    """The four Gotoh entry points, named by what they compute rather than by symbol."""

    GLOBAL_SCORE = "global-score"
    GLOBAL_ALIGNMENT = "global-alignment"
    LOCAL_SCORE = "local-score"
    LOCAL_ALIGNMENT = "local-alignment"


class Result(StrEnum):
    """Whether an entry point returns a number or a pair of gapped strings and a number."""

    SCORE = "score"
    ALIGNMENT = "alignment"


class Mode(StrEnum):
    """Which of the two alignment problems the recurrence solves."""

    GLOBAL = "global"
    LOCAL = "local"


@dataclass(frozen=True)
class _Spec:
    """Everything the dispatcher needs to serve one algorithm on any backend.

    `reference` is the public function, so the batch path never looks a name up in module globals.
    """

    reference: Callable
    result: Result
    mode: Mode


# The compiled module exports one entry point per result shape; every other axis is an argument.
_COMPILED_ENTRY = {Result.SCORE: "gotoh_scores", Result.ALIGNMENT: "gotoh_alignments"}


@lru_cache(maxsize=1)
def _mojo_backend():
    """Imports the compiled module, looking in the local build directory as a fallback.

    Keeping this in one place means callers never manipulate `sys.path` themselves, and a project
    checkout behaves the same as an installed package.
    """
    try:
        import affinegaps_mojo  # type: ignore[import-not-found]
    except ImportError:
        build = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
        if build not in sys.path:
            sys.path.insert(0, build)
        try:
            import affinegaps_mojo  # type: ignore[import-not-found]
        except ImportError:
            return None
    return affinegaps_mojo


def _compiled_module():
    """The compiled module, which `_resolve` has already refused to return without."""
    compiled = _mojo_backend()
    if compiled is None:
        raise RuntimeError("The Mojo backend is not built. See the README for how to build it.")
    return compiled


FASTEST_FIRST: tuple[tuple[Backend, Device], ...] = (
    (Backend.MOJO, Device.GPU),
    (Backend.MOJO, Device.CPU),
    (Backend.NUMBA, Device.CPU),
)
"""The combinations an unnamed backend and device try, in the order they are preferred."""


@cache
def available(backend: Backend = Backend.MOJO, device: Device = Device.CPU) -> bool:
    """Whether a backend and device combination actually runs, tried once and remembered.

    This executes a tiny alignment rather than inferring from an import, because a built extension
    on a machine with an unsupported driver imports cleanly and then fails at every device call.
    """
    if backend is Backend.PYTHON:
        return device is Device.CPU
    if backend is Backend.NUMBA:
        return device is Device.CPU and HAS_NUMBA
    if _mojo_backend() is None:
        return False
    try:
        needleman_wunsch_gotoh_score("AR", "RA", backend=backend, device=device)
    except Exception:
        return False
    return True


def _kernel(function, backend) -> Any:
    """The kernel body a backend asks for: NumBa's compiled dispatcher, or the Python it wrapped."""
    return function if backend is Backend.NUMBA else getattr(function, "py_func", function)


def _resolve_for(substitution, backend: Backend | None, device: Device | None) -> tuple[Backend, Device]:
    """Resolves the backend for a call, knowing what the caller wants scored.

    An unnamed backend adapts, and a table the compiled kernels do not carry is a reason to
    adapt away from them. A named backend is still honoured or refused, never quietly swapped.
    """
    if backend is None and _custom_scoring(substitution):
        backend = Backend.NUMBA if HAS_NUMBA else Backend.PYTHON
    return _resolve(backend, device)


def _resolve(backend: Backend | None, device: Device | None) -> tuple[Backend, Device]:
    """Validates a backend and device pair, filling in whichever the caller left unspecified.

    Unspecified adapts to the machine; specified is honoured or refused. Falling back silently
    would let a benchmark report the GPU while measuring the reference.

    Also the one place a caller's string becomes a member, so everything downstream compares
    members and a typo is refused here rather than mistaken for an unsupported combination.
    """
    backend = None if backend is None else Backend(backend)
    device = None if device is None else Device(device)
    if backend is None and device is None:
        for candidate in FASTEST_FIRST:
            if available(*candidate):
                return candidate
        return (Backend.PYTHON, Device.CPU)
    if backend is None:
        backend = Backend.MOJO
        if device is Device.CPU and not available(Backend.MOJO, Device.CPU):
            backend = Backend.NUMBA if HAS_NUMBA else Backend.PYTHON
    if device is None:
        device = Device.GPU if backend is Backend.MOJO and available(backend, Device.GPU) else Device.CPU

    if backend is not Backend.MOJO:
        if device is not Device.CPU:
            raise ValueError(f"The {backend} backend runs on the CPU only")
        if backend is Backend.NUMBA and not HAS_NUMBA:
            raise RuntimeError("NumBa is not installed. Install the `numba` extra.")
        return (backend, Device.CPU)
    if _mojo_backend() is None:
        raise RuntimeError("The Mojo backend is not built. See the README for how to build it.")
    return (Backend.MOJO, device)


def _custom_scoring(substitution) -> bool:
    """Whether the caller supplied a table the compiled backend does not carry.

    The compiled kernels hold the default matrix and can build a uniform one, so only an explicit
    table is out of reach.
    """
    return isinstance(substitution, TabulatedSubstitutionCosts)


def _reject_custom_scoring(algorithm, backend, options):
    """The compiled backends carry only the default table, plus a uniform match/mismatch pair."""
    if _custom_scoring(options.get("substitution")):
        raise NotImplementedError(f"{algorithm} on the {backend} backend needs match/mismatch or the default matrix")


def _compiled_call(algorithm, backend, device, first, second, options) -> Any:
    """Routes one pair to the compiled backend, where a single pair is a batch of one."""
    return _compiled_batch(algorithm, backend, device, [first], [second], options)[0]


def _compiled_batch(algorithm, backend, device, firsts, seconds, options) -> Any:
    """Routes a whole batch, which is what the device is for.

    The compiled side chooses between the stored and the linear traceback from `stored_budget`,
    so the only axes crossing the boundary are the ones the caller actually named.
    """
    _reject_custom_scoring(algorithm, backend, options)
    spec = _ALGORITHMS[algorithm]
    entry = getattr(_mojo_backend(), _COMPILED_ENTRY[spec.result])
    substitution = options.get("substitution")
    gaps = options.get("gaps") or AffineGapCosts()
    placement = replace(options.get("placement") or Placement(), device=device)
    if spec.result is Result.SCORE:
        return list(entry(list(firsts), list(seconds), spec.mode, placement, substitution, gaps))
    outcome = entry(list(firsts), list(seconds), spec.mode, placement, substitution, gaps, _STORED_MATRIX_BUDGET)
    return [tuple(row) for row in outcome]


def _batch(algorithm, backend, device, firsts, seconds, **options):
    """One batch entry point, on whichever backend resolves."""
    backend, device = _resolve_for(options.get("substitution"), backend, device)
    if device is Device.CPU:
        single = _ALGORITHMS[algorithm].reference
        return [single(a, b, backend=backend, device=device, **options) for a, b in zip(firsts, seconds, strict=True)]
    return _compiled_batch(algorithm, backend, device, firsts, seconds, options)


def needleman_wunsch_gotoh_alignments(firsts, seconds, *, backend=None, device=None, **options):
    """Globally aligns a whole batch, one thread block per pair on the device."""
    return _batch(Algorithm.GLOBAL_ALIGNMENT, backend, device, firsts, seconds, **options)


def smith_waterman_gotoh_alignments(firsts, seconds, *, backend=None, device=None, **options):
    """Locally aligns a whole batch, one thread block per pair on the device."""
    return _batch(Algorithm.LOCAL_ALIGNMENT, backend, device, firsts, seconds, **options)


def needleman_wunsch_gotoh_scores(firsts, seconds, *, backend=None, device=None, **options):
    """Scores a whole batch globally, without reconstructing the alignments."""
    return _batch(Algorithm.GLOBAL_SCORE, backend, device, firsts, seconds, **options)


def smith_waterman_gotoh_scores(firsts, seconds, *, backend=None, device=None, **options):
    """Scores a whole batch locally, without reconstructing the alignments."""
    return _batch(Algorithm.LOCAL_SCORE, backend, device, firsts, seconds, **options)


# endregion Backends


def _gotoh_score(
    algorithm: Algorithm,
    reference,
    first: str,
    second: str,
    substitution,
    gaps,
    backend: Backend | None,
    device: Device | None,
    placement: Placement | None = None,
) -> int:
    """The five steps both score entry points take, which differ only in algorithm and kernel.

    The alignment entry points are not folded in here: reconstructing a path needs the encoded
    sequences and the alphabet after the kernel returns, and the local variant slices them and
    carries its own stopping rule.
    """
    backend, device = _resolve_for(substitution, backend, device)
    if backend is Backend.MOJO:
        options = {"substitution": substitution, "gaps": gaps, "placement": placement}
        return int(_compiled_call(algorithm, backend, device, first, second, options))
    alphabet, matrix, opening, extend = _validate_gotoh_arguments(substitution, gaps)
    return int(
        _kernel(reference, backend)(
            _translate_sequence(first, alphabet),
            _translate_sequence(second, alphabet),
            substitution_matrix=matrix,
            opening=opening,
            extend=extend,
        )
    )


def needleman_wunsch_gotoh_alignment(
    first: str,
    second: str,
    *,
    substitution: SubstitutionCosts | None = None,
    gaps: AffineGapCosts | None = None,
    backend: Backend | None = None,
    device: Device | None = None,
    placement: Placement | None = None,
) -> tuple[str, str, int]:
    """
    Aligns two sequences using Gotoh's affine gap penalty extensions for the
    Needleman-Wunsch global alignment algorithm.
    """
    backend, device = _resolve_for(substitution, backend, device)
    if backend is Backend.MOJO:
        return _compiled_call(
            Algorithm.GLOBAL_ALIGNMENT,
            backend,
            device,
            first,
            second,
            dict(substitution=substitution, gaps=gaps, placement=placement),
        )

    substitution_alphabet, substitution_matrix, opening, extend = _validate_gotoh_arguments(substitution, gaps)

    encoded_first = _translate_sequence(first, substitution_alphabet)
    encoded_second = _translate_sequence(second, substitution_alphabet)
    scores, changes, deletes, inserts = _kernel(_needleman_wunsch_gotoh_kernel, backend)(
        encoded_first,
        encoded_second,
        substitution_matrix=substitution_matrix,
        opening=opening,
        extend=extend,
    )

    first_gapped, second_gapped = _reconstruct_alignment(
        changes,
        scores,
        deletes,
        inserts,
        encoded_first,
        encoded_second,
        opening,
        extend,
        lambda x: substitution_alphabet[x],
        lambda i, j: i > 0 and j > 0,
    )
    return first_gapped, second_gapped, int(scores[-1, -1])


def needleman_wunsch_gotoh_score(
    first: str,
    second: str,
    *,
    substitution: SubstitutionCosts | None = None,
    gaps: AffineGapCosts | None = None,
    backend: Backend | None = None,
    device: Device | None = None,
) -> int:
    """
    Measures the alignment score of two sequences using Gotoh's affine gap penalty extensions for the
    Needleman-Wunsch global alignment algorithm. Uses less memory than the alignment function.
    """
    return _gotoh_score(
        Algorithm.GLOBAL_SCORE, _needleman_wunsch_gotoh_score_kernel, first, second, substitution, gaps, backend, device
    )


def smith_waterman_gotoh_alignment(
    first: str,
    second: str,
    *,
    substitution: SubstitutionCosts | None = None,
    gaps: AffineGapCosts | None = None,
    backend: Backend | None = None,
    device: Device | None = None,
    placement: Placement | None = None,
) -> tuple[str, str, int]:
    """
    Aligns two sequences using the Smith-Waterman algorithm for local alignment.
    """
    backend, device = _resolve_for(substitution, backend, device)
    if backend is Backend.MOJO:
        return _compiled_call(
            Algorithm.LOCAL_ALIGNMENT,
            backend,
            device,
            first,
            second,
            dict(substitution=substitution, gaps=gaps, placement=placement),
        )

    substitution_alphabet, substitution_matrix, opening, extend = _validate_gotoh_arguments(substitution, gaps)

    encoded_first = _translate_sequence(first, substitution_alphabet)
    encoded_second = _translate_sequence(second, substitution_alphabet)
    scores, changes, deletes, inserts, best_place = _kernel(_smith_waterman_gotoh_kernel, backend)(
        encoded_first,
        encoded_second,
        substitution_matrix=substitution_matrix,
        opening=opening,
        extend=extend,
    )

    first_prefix, second_prefix = best_place
    first_gapped, second_gapped = _reconstruct_alignment(
        changes[: first_prefix + 1, : second_prefix + 1],
        scores,
        deletes,
        inserts,
        encoded_first[:first_prefix],
        encoded_second[:second_prefix],
        opening,
        extend,
        lambda x: substitution_alphabet[x],
        lambda i, j: i > 0 and j > 0 and scores[i, j] > 0,
        flush_prefixes=False,
    )
    return first_gapped, second_gapped, int(scores[first_prefix, second_prefix])


def smith_waterman_gotoh_score(
    first: str,
    second: str,
    *,
    substitution: SubstitutionCosts | None = None,
    gaps: AffineGapCosts | None = None,
    backend: Backend | None = None,
    device: Device | None = None,
) -> int:
    """
    Measures the Smith-Waterman local alignment score using Gotoh's affine gap penalty extensions.
    """
    return _gotoh_score(
        Algorithm.LOCAL_SCORE, _smith_waterman_gotoh_score_kernel, first, second, substitution, gaps, backend, device
    )


def sankoff_cofold(
    first: str,
    second: str,
    *,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
    backend: Backend | None = None,
    device: Device | None = None,
    placement: Placement | None = None,
) -> tuple[str, str, str, int]:
    """Exact simultaneous alignment and folding, returning both gapped strings and one structure.

    Sankoff's recurrence credits a base pair only where both sequences can form it, so the signal
    is covariation and no energy model is involved. Memory grows as the fourth power of the
    sequence length, which is inherent to the recurrence and bounds this to a few hundred bases.
    """
    backend, device = _resolve(backend, device)
    if backend is Backend.MOJO:
        compiled = _compiled_module()
        left, right, structure, score = compiled.sankoff_cofold(
            first, second, gap, match, mismatch, replace(placement or Placement(), device=device)
        )
        return (left, right, structure, int(score))
    return _sankoff_cofold_reference(first, second, match=match, mismatch=mismatch, gap=gap)


def zuker_fold(
    sequence: str,
    *,
    backend: Backend | None = None,
    device: Device | None = None,
    placement: Placement | None = None,
) -> tuple[str, float]:
    """Minimum free energy folding of one RNA sequence over the Turner nearest-neighbour model.

    Returns the dot-bracket structure and the free energy in kilocalories per mole. The recurrence
    is exact and works in integer decikilocalories, so a sequence always folds to the same answer.
    """
    backend, device = _resolve(backend, device)
    if backend is Backend.MOJO:
        structure, energy = _compiled_module().zuker_fold(sequence, replace(placement or Placement(), device=device))
        return (structure, float(energy))
    return _zuker_fold_reference(sequence)


# region Dispatch Table

# Defined here rather than beside the enums so `reference` binds the function object itself.
_ALGORITHMS: dict[Algorithm, _Spec] = {
    Algorithm.GLOBAL_SCORE: _Spec(needleman_wunsch_gotoh_score, Result.SCORE, Mode.GLOBAL),
    Algorithm.LOCAL_SCORE: _Spec(smith_waterman_gotoh_score, Result.SCORE, Mode.LOCAL),
    Algorithm.GLOBAL_ALIGNMENT: _Spec(needleman_wunsch_gotoh_alignment, Result.ALIGNMENT, Mode.GLOBAL),
    Algorithm.LOCAL_ALIGNMENT: _Spec(smith_waterman_gotoh_alignment, Result.ALIGNMENT, Mode.LOCAL),
}

# endregion Dispatch Table


# region Command Line


class Verb(StrEnum):
    """The three recurrences the command line exposes, one verb each."""

    ALIGN = "align"
    """Gotoh affine-gap alignment of two sequences."""
    FOLD = "fold"
    """Zuker minimum free energy folding of one RNA sequence."""
    COFOLD = "cofold"
    """Sankoff simultaneous alignment and folding of two RNA sequences."""


def _named_accelerator(device: Device, gpu_id: int | None) -> int:
    """The accelerator a run will use, refusing an index the named device will never reach."""
    if gpu_id is not None and device is not Device.GPU:
        raise ValueError("--gpu-id names an accelerator, which --device cpu will not reach")
    return gpu_id or 0


def _counted(text: str) -> int:
    """An argparse type for a count that cannot be negative, so the parser refuses it rather than the driver."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value}")
    return value


def _placement_parser() -> argparse.ArgumentParser:
    """The flags every verb shares, so they cannot drift apart between verbs."""
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--backend", type=Backend, choices=tuple(Backend), help="Which implementation runs it")
    shared.add_argument("--device", type=Device, choices=tuple(Device), help="Which hardware it runs on")
    shared.add_argument("--gpu-id", type=_counted, help="Which accelerator; implies --device gpu")
    shared.add_argument(
        "--format", type=Format, choices=tuple(Format), default=Format.HUMAN, help="How to print the answer"
    )
    shared.add_argument(
        "--color", type=Coloring, choices=tuple(Coloring), default=Coloring.AUTO, help="Colour the alignment"
    )
    shared.add_argument("--verbose", action="store_true", help="Report the backend and throughput on stderr")
    shared.add_argument("--help", action="help", help="Show this message")
    return shared


def _build_parser() -> argparse.ArgumentParser:
    """One verb per recurrence, each seeing only the flags that apply to it.

    Subcommands rather than a mode flag because folding takes one sequence where the others take
    two, and because an affine `--open` and a linear `--gap` must never be offered together.
    """
    import argparse

    shared = _placement_parser()
    parser = argparse.ArgumentParser(
        prog="affinegaps", description="Exact biosequence dynamic programs, with traceback.", add_help=False
    )
    parser.add_argument("--help", action="help", help="Show this message")
    verbs = parser.add_subparsers(dest="verb", metavar="VERB")

    align = verbs.add_parser(Verb.ALIGN, parents=[shared], add_help=False, help="Gotoh alignment of two sequences")
    align.add_argument("first", help="The first sequence, like insulin GIVEQCCTSICSLYQLENYCN")
    align.add_argument("second", help="The second sequence, like glucagon HSQGTFTSDYSKYLDSRAEQDFV")
    align.add_argument("--local", action="store_true", help="Smith-Waterman instead of Needleman-Wunsch")
    align.add_argument("--match", type=int, help="Uniform match score, instead of scaled BLOSUM62")
    align.add_argument("--mismatch", type=int, help="Uniform mismatch score, instead of scaled BLOSUM62")
    align.add_argument("--open", type=int, help=f"Gap opening penalty, {AffineGapCosts().open} by default")
    align.add_argument("--extend", type=int, help=f"Gap extension penalty, {AffineGapCosts().extend} by default")
    # Only the linear-space traceback runs a parallel region, so only this verb can honour a width.
    align.add_argument(
        "--threads",
        type=_counted,
        default=hardware_threads(),
        help="Host threads the traceback may fork across",
    )

    fold = verbs.add_parser(Verb.FOLD, parents=[shared], add_help=False, help="Zuker folding of one RNA sequence")
    fold.add_argument("sequence", help="The RNA sequence to fold, over ACGU")

    cofold = verbs.add_parser(Verb.COFOLD, parents=[shared], add_help=False, help="Sankoff alignment and folding")
    cofold.add_argument("first", help="The first RNA sequence, over ACGU")
    cofold.add_argument("second", help="The second RNA sequence, over ACGU")
    cofold.add_argument("--match", type=int, default=2, help="Score for aligning two equal bases")
    cofold.add_argument("--mismatch", type=int, default=-1, help="Score for aligning two different bases")
    cofold.add_argument("--gap", type=int, default=-2, help="Linear gap cost; Sankoff has no affine model")
    return parser


def _run_align(args) -> dict:
    """Aligns the pair and returns the record the renderer prints."""
    substitution = None
    if args.match is not None or args.mismatch is not None:
        if args.match is None or args.mismatch is None:
            print("Error: --match and --mismatch must be given together.", file=sys.stderr)
            sys.exit(2)
        substitution = UniformSubstitutionCosts(match=args.match, mismatch=args.mismatch)
    defaults = AffineGapCosts()
    gaps = AffineGapCosts(
        open=defaults.open if args.open is None else args.open,
        extend=defaults.extend if args.extend is None else args.extend,
    )
    aligner = smith_waterman_gotoh_alignment if args.local else needleman_wunsch_gotoh_alignment
    first_gapped, second_gapped, score = aligner(
        args.first,
        args.second,
        substitution=substitution,
        gaps=gaps,
        backend=args.backend,
        device=args.device,
        placement=Placement(device=args.device, gpu_id=args.gpu_id, threads=args.threads),
    )
    return {
        "operation": Verb.ALIGN,
        "mode": "local" if args.local else "global",
        "first": args.first,
        "second": args.second,
        "first_gapped": first_gapped,
        "second_gapped": second_gapped,
        "score": score,
        "cells": len(args.first) * len(args.second),
    }


def _run_fold(args) -> dict:
    """Folds the sequence and returns the record the renderer prints."""
    structure, energy = zuker_fold(
        args.sequence,
        backend=args.backend,
        device=args.device,
        placement=Placement(device=args.device, gpu_id=args.gpu_id),
    )
    return {
        "operation": Verb.FOLD,
        "sequence": args.sequence,
        "structure": structure,
        "energy_kcal_per_mol": energy,
        "cells": len(args.sequence) ** 2,
    }


def _run_cofold(args) -> dict:
    """Aligns and folds the pair together, returning the record the renderer prints."""
    first_gapped, second_gapped, structure, score = sankoff_cofold(
        args.first,
        args.second,
        match=args.match,
        mismatch=args.mismatch,
        gap=args.gap,
        backend=args.backend,
        device=args.device,
        placement=Placement(device=args.device, gpu_id=args.gpu_id),
    )
    return {
        "operation": Verb.COFOLD,
        "first": args.first,
        "second": args.second,
        "first_gapped": first_gapped,
        "second_gapped": second_gapped,
        "structure": structure,
        "score": score,
        "cells": (len(args.first) * len(args.second)) ** 2,
    }


RUNNERS = {Verb.ALIGN: _run_align, Verb.FOLD: _run_fold, Verb.COFOLD: _run_cofold}
"""Which function serves each verb, so the dispatch is a lookup rather than a chain."""

REPORT_ROWS = {
    Verb.ALIGN: (
        ("Sequence 1", "first"),
        ("Sequence 2", "second"),
        ("Alignment 1", "first_gapped"),
        ("Alignment 2", "second_gapped"),
        ("Score", "score"),
    ),
    Verb.FOLD: (("Sequence", "sequence"), ("Structure", "structure"), ("Energy", "energy_kcal_per_mol")),
    Verb.COFOLD: (("Sequence 1", "first"), ("Sequence 2", "second"), ("Structure", "structure"), ("Score", "score")),
}
"""The label and key of every printed row, in order, for each verb."""


class Format(StrEnum):
    """How an answer is printed."""

    HUMAN = "human"
    """Labelled rows, coloured when the terminal wants it."""
    JSON = "json"
    """One object on one line, for a pipe."""


class Coloring(StrEnum):
    """Whether an alignment is coloured."""

    AUTO = "auto"
    """Colour only when standard output is a terminal."""
    ALWAYS = "always"
    """Colour regardless, for a caller that will render the escapes."""
    NEVER = "never"
    """Never colour."""


def _wants_color(choice: Coloring) -> bool:
    """Whether to colour, which `auto` answers by asking if stdout is a terminal.

    The previous rule was whether colorama imports, so a piped alignment carried escape codes.
    """
    if choice is Coloring.NEVER:
        return False
    if choice is Coloring.ALWAYS:
        return True
    return sys.stdout.isatty()


def _render(record: dict, colored: bool) -> str:
    """One report per verb, its labels padded to a width that verb chooses for itself."""
    rows = REPORT_ROWS[record["operation"]]
    shown = dict(record)
    if colored and "first_gapped" in shown:
        try:
            from colorama import init as start_colorama

            start_colorama(autoreset=True, strip=False)
            shown["first_gapped"], shown["second_gapped"] = colorize_alignment(
                record["first_gapped"], record["second_gapped"]
            )
        except ImportError:
            pass
    if "energy_kcal_per_mol" in shown:
        shown["energy_kcal_per_mol"] = f"{record['energy_kcal_per_mol']} kcal/mol"
    width = max(len(label) for label, _ in rows) + 1
    return "\n".join(f"{label + ':':<{width}} {shown[key]}" for label, key in rows)


def _report_placement(record: dict, backend: str, device: str, gpu_id: int, elapsed: float) -> None:
    """Which backend ran and how fast, on stderr so a piped payload stays parseable.

    Elapsed is printed beside the rate because one short problem measures mostly dispatch overhead,
    and a bare cell-update rate would invite reading more into it than it says.
    """
    cells = record["cells"]
    rate = cells / elapsed / 1e6 if elapsed > 0 else float("inf")
    # The index only names something when there is an accelerator for it to name.
    where = f"{device}:{gpu_id}" if device is Device.GPU else str(device)
    print(f"  backend:     {backend} on {where}", file=sys.stderr)
    print(f"  cells:       {cells}", file=sys.stderr)
    print(f"  elapsed:     {elapsed * 1e3:.3f} ms", file=sys.stderr)
    print(f"  throughput:  {rate:.2f} MCUPS", file=sys.stderr)


def main():
    """Parses one verb's arguments, runs it, and prints the answer."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.verb is None:
        parser.print_help(sys.stderr)
        sys.exit(2)
    verb = Verb(args.verb)

    # Naming an accelerator is asking for one, so it settles the device the resolver would guess.
    if args.gpu_id is not None and args.device is None:
        args.device = Device.GPU
    backend, device = _resolve(args.backend, args.device)
    args.backend, args.device = backend, device
    try:
        args.gpu_id = _named_accelerator(device, args.gpu_id)
    except ValueError as contradiction:
        parser.error(str(contradiction))
    try:
        started = time.perf_counter()
        record = RUNNERS[verb](args)
        elapsed = time.perf_counter() - started
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.format is Format.JSON:
        payload = {key: value for key, value in record.items() if key != "cells"}
        payload["backend"], payload["device"] = backend, device
        payload["gpu_id"] = args.gpu_id
        if verb is Verb.ALIGN:
            payload["threads"] = args.threads
        print(json.dumps(payload))
    else:
        print(_render(record, _wants_color(args.color)))
    if args.verbose:
        _report_placement(record, backend, device, args.gpu_id, elapsed)


# endregion Command Line


if __name__ == "__main__":
    main()
