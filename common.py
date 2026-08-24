"""
Scoring records and helpers shared by the reference kernels.

Mirrors `common.mojo`: the parts every recurrence needs, and nothing that presumes a gap model
or a base pair.
"""

import os
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

try:
    import numba as nb
except ImportError:
    nb = None

HAS_NUMBA = nb is not None
"""Whether NumBa is installed, which decides if the reference kernels are compiled."""


def jit_if_available(*jit_args, **jit_kwargs):
    """Compiles with NumBa when it is installed, and leaves the function untouched when it is not."""

    def decorator(func):
        if nb is not None:
            # Cached to disk, so only the first process on a machine pays the compile.
            return nb.jit(*jit_args, cache=True, nogil=True, **jit_kwargs)(func)
        return func

    return decorator


@dataclass(frozen=True)
class AffineGapCosts:
    """Gotoh's two-parameter gap model. Both penalties are negative."""

    open: int = -4 * 5
    extend: int = int(-0.2 * 5)

    def __post_init__(self):
        # Checked here rather than per call, so no backend can be handed a non-affine recurrence.
        if self.open > self.extend:
            raise ValueError("Opening a gap must not cost less than extending it.")
        if self.extend > 0:
            # A rewarded gap leaves the border sentinels undominated, so the seeded layer wins.
            raise ValueError("Gap penalties must not be positive.")


@dataclass(frozen=True)
class UniformSubstitutionCosts:
    """One score for equal symbols and one for unequal, over any alphabet."""

    match: int
    mismatch: int


@dataclass(frozen=True)
class TabulatedSubstitutionCosts:
    """A substitution matrix and the alphabet that indexes it."""

    alphabet: str
    matrix: np.ndarray

    def __post_init__(self):
        # NumBa indexes this with bounds checking off, so a wrong shape reads past the array.
        expected = (len(self.alphabet), len(self.alphabet))
        if self.matrix.shape != expected:
            raise ValueError(f"A {len(self.alphabet)}-letter alphabet needs a {expected} matrix.")


def _translate_sequence(sequence: str, alphabet: str) -> np.ndarray:
    """Maps characters to alphabet indices, raising on anything outside the alphabet."""
    unknown = {letter for letter in sequence if letter not in alphabet}
    if unknown:
        raise ValueError(f"Found characters outside the alphabet {alphabet!r}: {''.join(sorted(unknown))}")
    return np.array([alphabet.index(letter) for letter in sequence], dtype=np.uint8)


# A closed pair rather than four loose parameters, so "a match score and a matrix" cannot be said.
SubstitutionCosts = UniformSubstitutionCosts | TabulatedSubstitutionCosts


class Backend(StrEnum):
    """Which implementation serves a call."""

    PYTHON = "python"
    """The NumPy reference, which every other backend is checked against."""
    NUMBA = "numba"
    """The same reference, compiled."""
    MOJO = "mojo"
    """The shipped kernels."""


class Device(StrEnum):
    """Which hardware a backend runs on."""

    CPU = "cpu"
    """The serial reference sweep."""
    GPU = "gpu"
    """The parallel sweep, on one accelerator."""


class Background(StrEnum):
    """Which terminal background a colouring is chosen for."""

    DARK = "dark"
    """A light-on-dark terminal, where gaps read best in white."""
    LIGHT = "light"
    """A dark-on-light terminal, where gaps read best in black."""


def hardware_threads() -> int:
    """Threads this process may actually run on, which an affinity mask or a cgroup quota narrows.

    The online CPU count is the wrong answer on a shared machine: it counts cores this process has
    been forbidden from touching. Only Linux exposes such a mask.
    """
    if (allowed := getattr(os, "sched_getaffinity", None)) is not None:
        return max(len(allowed(0)), 1)
    return max(os.cpu_count() or 1, 1)


@dataclass(frozen=True)
class Placement:
    """Where a call runs, and which of the machine's resources it may take.

    Carried as one record because the compiled entry points read it the way they read `gaps` and
    `substitution`, and because three loose arguments would not fit what the bindings can infer.
    """

    device: Device = Device.CPU
    """Which hardware serves the call."""
    gpu_id: int = 0
    """Which accelerator, meaningless and so refused unless the device is the GPU."""
    threads: int = field(default_factory=hardware_threads)
    """How many host threads a parallel region may take, at least one."""

    def __post_init__(self):
        # Checked here rather than per call, so no backend can be handed a placement it cannot serve.
        if self.device is not Device.GPU and self.gpu_id != 0:
            raise ValueError("An accelerator index only means something when the device is the GPU.")
        if self.gpu_id < 0:
            raise ValueError("An accelerator index cannot be negative.")
        if self.threads < 1:
            raise ValueError("A parallel region needs at least one thread.")


default_proteins_alphabet: str = "ARNDCQEGHILKMFPSTWYVBZX"
"""The twenty-three protein letters BLOSUM62 is tabulated over."""

default_rna_alphabet: str = "ACGU"
"""The four RNA bases, ordered so a base doubles as its own index."""
