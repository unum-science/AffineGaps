"""
Turner 2004 nearest-neighbour parameters, in integer decikilocalories per mole.

The values are measured constants published in Mathews et al. 2004 and curated in the Nearest
Neighbor Database. Integers keep the recurrence in exact arithmetic, so a fold is reproducible
bit for bit rather than depending on floating-point association order.

Dangle tables are indexed by the pair as seen from outside the helix, so a helix spanning
`(opening, closing)` looks itself up as `PAIR_INDEX[sequence[closing], sequence[opening]]`.

Special-loop tables replace a hairpin's initiation and terminal mismatch outright rather than
adding to them; their keys are the closing pair and loop bases packed base-four, most significant
first, so a kernel scans integers rather than strings.

Not included, and therefore absent from the energies: coaxial stacking, and the special tables for
one-by-one, two-by-one and two-by-two internal loops.
"""

import numpy as np

BASES = "ACGU"
PAIRS = ["AU", "CG", "GC", "UA", "GU", "UG"]
LOOP_LIMIT = 30
FORBIDDEN = 30000

# fmt: off
# Which of the six pair types two bases form, or -1 for none.
PAIR_INDEX = np.array(
[
    [    -1,     -1,     -1,      0],
    [    -1,     -1,      1,     -1],
    [    -1,      2,     -1,      4],
    [     3,     -1,      5,     -1],
],
    dtype=np.int32,
)
# fmt: on

# fmt: off
# STACK[closing][inner]: one helix step.
STACK = np.array(
[
    [    -9,    -22,    -21,    -11,     -6,    -14],
    [   -21,    -33,    -24,    -21,    -14,    -21],
    [   -24,    -34,    -33,    -22,    -15,    -25],
    [   -13,    -25,    -21,    -14,     -5,     13],
    [   -13,    -24,    -21,     -9,    -10,    -13],
    [   -10,    -15,    -14,     -6,      3,     -5],
],
    dtype=np.int32,
)
# fmt: on

# fmt: off
# TERMINAL_MISMATCH_HAIRPIN[closing][first][last]: unpaired bases facing a hairpin.
TERMINAL_MISMATCH_HAIRPIN = np.array(
[
    [    -8,    -10,     -8,    -10],
    [    -6,     -7,     -6,     -7],
    [   -17,    -10,    -16,    -10],
    [    -6,     -8,     -6,    -17],
    [   -15,    -15,    -14,    -15],
    [   -10,    -11,    -10,     -8],
    [   -23,    -15,    -24,    -15],
    [   -10,    -14,    -10,    -21],
    [   -11,    -15,    -13,    -15],
    [   -11,     -7,    -11,     -5],
    [   -25,    -15,    -22,    -15],
    [   -11,    -10,    -11,    -16],
    [    -3,    -10,     -8,    -10],
    [    -6,     -7,     -6,     -7],
    [   -15,    -10,    -16,    -10],
    [    -6,     -8,     -6,    -15],
    [   -10,     -8,    -10,     -8],
    [    -7,     -6,     -7,     -5],
    [   -20,     -8,    -20,     -8],
    [    -7,     -6,     -7,    -14],
    [   -10,     -8,    -11,     -8],
    [    -7,     -6,     -7,     -5],
    [   -14,     -8,    -16,     -8],
    [    -7,     -6,     -7,    -14],
],
    dtype=np.int32,
).reshape(len(PAIRS), 4, 4)
# fmt: on

# fmt: off
# TERMINAL_MISMATCH_INTERNAL[closing][first][last]: unpaired bases facing an internal loop.
TERMINAL_MISMATCH_INTERNAL = np.array(
[
    [     2,      2,     -6,      2],
    [     2,      2,      2,      2],
    [    -8,      2,     -8,      2],
    [     2,      2,      2,     -4],
    [     0,      0,     -8,      0],
    [     0,      0,      0,      0],
    [   -10,      0,    -10,      0],
    [     0,      0,      0,     -6],
    [     0,      0,     -8,      0],
    [     0,      0,      0,      0],
    [   -10,      0,    -10,      0],
    [     0,      0,      0,     -6],
    [     2,      2,     -6,      2],
    [     2,      2,      2,      2],
    [    -8,      2,     -8,      2],
    [     2,      2,      2,     -4],
    [     2,      2,     -6,      2],
    [     2,      2,      2,      2],
    [    -8,      2,     -8,      2],
    [     2,      2,      2,     -4],
    [     2,      2,     -6,      2],
    [     2,      2,      2,      2],
    [    -8,      2,     -8,      2],
    [     2,      2,      2,     -4],
],
    dtype=np.int32,
).reshape(len(PAIRS), 4, 4)
# fmt: on

# fmt: off
# DANGLE_AFTER[pair][base]: one base stacking past the 3' end of a helix.
DANGLE_AFTER = np.array(
[
    [    -8,     -5,     -8,     -6],
    [   -17,     -8,    -17,    -12],
    [   -11,     -4,    -13,     -6],
    [    -7,     -1,     -7,     -1],
    [    -8,     -5,     -8,     -6],
    [    -7,     -1,     -7,     -1],
],
    dtype=np.int32,
)
# fmt: on

# fmt: off
# DANGLE_BEFORE[pair][base]: one base stacking before the 5' end of a helix.
DANGLE_BEFORE = np.array(
[
    [    -3,     -1,     -2,     -2],
    [    -2,     -3,      0,      0],
    [    -5,     -3,     -2,     -1],
    [    -3,     -3,     -4,     -2],
    [    -3,     -1,     -2,     -2],
    [    -3,     -3,     -4,     -2],
],
    dtype=np.int32,
)
# fmt: on

# fmt: off
# Hairpin initiation by loop size, indexed from one.
HAIRPIN_INITIATION = np.array(
[
      30000,   30000,   30000,      54,      56,      57,      54,      60,
         55,      64,      65,      66,      67,      68,      69,      69,
         70,      71,      71,      72,      72,      73,      73,      74,
         74,      75,      75,      75,      76,      76,      77,
],
    dtype=np.int32,
)
# fmt: on

# fmt: off
# Bulge initiation by loop size, indexed from one.
BULGE_INITIATION = np.array(
[
      30000,      38,      28,      32,      36,      40,      44,      46,
         47,      48,      49,      50,      51,      52,      53,      54,
         54,      55,      55,      56,      57,      57,      58,      58,
         58,      59,      59,      60,      60,      60,      61,
],
    dtype=np.int32,
)
# fmt: on

# fmt: off
# Internal loop initiation by total unpaired count.
INTERNAL_INITIATION = np.array(
[
      30000,   30000,   30000,   30000,      11,      20,      20,      21,
         23,      24,      25,      26,      27,      28,      29,      29,
         30,      31,      31,      32,      33,      33,      34,      34,
         35,      35,      35,      36,      36,      37,      37,
],
    dtype=np.int32,
)
# fmt: on

# fmt: off
# Packed 5-mers whose hairpin energy is tabulated outright.
TRILOOP_KEYS = np.array([262, 753], dtype=np.int32)
# fmt: on

# fmt: off
# The energy each triloop key stands for.
TRILOOP_ENERGIES = np.array([68, 69], dtype=np.int32)
# fmt: on

# fmt: off
# Packed 6-mers whose hairpin energy is tabulated outright.
TETRALOOP_KEYS = np.array(
[
       1050,    1290,    1306,    1354,    1418,    1434,    1482,    1498,
       1802,    1818,    1866,    1882,    1946,    1994,    2010,    2042,
],
    dtype=np.int32,
)
# fmt: on

# fmt: off
# The energy each tetraloop key stands for.
TETRALOOP_ENERGIES = np.array([55, 33, 37, 34, 35, 36, 37, 25, 36, 28, 37, 27, 28, 35, 37, 37], dtype=np.int32)
# fmt: on

# fmt: off
# Packed 8-mers whose hairpin energy is tabulated outright.
HEXALOOP_KEYS = np.array([4807, 4835, 4839, 4847], dtype=np.int32)
# fmt: on

# fmt: off
# The energy each hexaloop key stands for.
HEXALOOP_ENERGIES = np.array([23, 31, 24, 13], dtype=np.int32)
# fmt: on

# Scalar rules. The multiloop cost is `offset + per_unpaired * unpaired + per_helix * helices`.
MULTILOOP_OFFSET = 93
MULTILOOP_PER_UNPAIRED = 0
MULTILOOP_PER_HELIX = -6
NINIO_PER_ASYMMETRY = 6
NINIO_CAP = 30
# Charged to any helix end that is not a Watson-Crick CG or GC pair, measured against `efn2`.
TERMINAL_AU = 5
