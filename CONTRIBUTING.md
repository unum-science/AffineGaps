# Contributing to Affine Gaps

## Getting Started

To test, install the development dependencies and run the tests.

```bash
pip install -e . --group test
pytest test.py
```

Alternatively, consider using `uv`:

```sh
uv venv --python 3.12           # Or your preferred Python version
source .venv/bin/activate       # To activate the virtual environment
uv pip install --group test .   # To install the package and its test dependencies
uv run pytest test.py           # To run the tests
```

## House Style

Arm the hooks once per clone, before the first commit:

```sh
pixi run hooks    # git config core.hooksPath .githooks
pixi run format   # mojo format -l 120 *.mojo
```

`pre-commit` gates `mojo format`, `black` and `ruff` on the staged tree, plus three prose rules: big-O written as `$O(n^2)$` rather than backticked, no decorative comment rulers, and a space before every footnote glyph in a Markdown table.
`commit-msg` holds the subject to `Fix:`, `Add:`, `Improve:`, `Chore:`, `Make:`, `Docs:` or `Break:`.
The Mojo gate fails closed when the formatter cannot be resolved, because outside the pixi environment `mojo format` warns, changes nothing and still exits zero.
`.githooks/selftest` gives every check a fixture that must be rejected and one that must pass, so a rule that stops firing fails the suite rather than disappearing quietly.

## Testing the Mojo Backend

The GPU kernels are optional. Build them with `pixi run build` and the same suite picks them up; without a build, every Mojo test skips and the pure-Python suite still runs.

```sh
pixi run test                                  # builds the extension, then runs the suite against it
AFFINEGAPS_BACKENDS=python-cpu pixi run pytest # the reference alone, which is what CI runs
```

Every property test runs against each backend, named `python-cpu`, `numba-cpu`, `mojo-cpu` and `mojo-gpu`.
A backend the machine cannot serve is skipped rather than failed, and the skip reason carries the real cause — a missing build and an unsupported driver are different problems and say so.

The Mojo kernels and the Python reference are held to the same recurrence, the same border initialization and the same tie-breaking, so the suite compares them exhaustively rather than by sampling — every pair of sequences whose combined length reaches `4 + AFFINEGAPS_SCALE` over a two-letter alphabet, both global and local, on the host and on the device.

## Knobs

Four environment variables shape a run, and the first two are deliberately separate: one is breadth, the other is depth.

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `AFFINEGAPS_REPETITIONS` | `10` | Random draws per randomized test |
| `AFFINEGAPS_SCALE` | `1` | Multiplier on every exhaustive oracle's budget, and the highest frozen tier re-derived |
| `AFFINEGAPS_SEED` | unset | Fixes the draws so a failure reproduces |
| `AFFINEGAPS_BACKENDS` | all four | Narrows the backend axis |

`AFFINEGAPS_REPETITIONS=100 AFFINEGAPS_SCALE=1` is a fuzzing run and `AFFINEGAPS_REPETITIONS=1 AFFINEGAPS_SCALE=4` is a release gate, which is why one number cannot express both.

## Properties Under Test

### Every Path Must Achieve Its Own Score

The invariant worth knowing about before touching a traceback.
With affine gaps it is not automatic: a walk that reads only the winning operation at each cell can leave a gap run and re-enter it, paying a second opening penalty the score never did, and the returned strings then score less than the number returned beside them.
Both implementations walk the match, deletion and insertion layers, and `test_alignment_achieves_its_score` re-scores every returned path to enforce it.

Local alignment has a second version of the same trap.
The traceback stops at the first non-positive cell, so the untraced prefixes are outside the alignment and must not be flushed into the result.

### Enumeration, Which Is the Oracle That Matters

Every recurrence is checked against enumerating the whole answer space, sharing no algorithm with the thing it checks.
`test_scores_match_brute_force_enumeration` walks every alignment of every pair up to a combined length of `4 + AFFINEGAPS_SCALE` over a two-letter alphabet.
`test_fold_matches_brute_force_enumeration` walks every nested structure a sequence admits and scores each one straight off the Turner tables, through a scorer that shares no code with `folding.py`.

This is the only kind of oracle that can catch a recurrence which is self-consistently wrong on every backend at once, which is exactly what rewriting a recurrence risks.

### The Frozen Corpus

`FOLD_CASES` and `COFOLD_CASES` hold answers checked case by case before being written down: the minimum-hairpin boundary, all three special-loop tables, the bulge and interior ladder including both ends of Ninio's cap, a single wobble against a Watson-Crick control, energy ties, a multiloop, and the `MAX_LOOP` boundary on both sides.
They assert on every backend and cost microseconds; `AFFINEGAPS_SCALE` decides which of them are additionally re-derived by enumeration.

`TURNER_FINGERPRINT` digests every energy table and scalar.
Edit `turner.py` and that one test fails by name, instead of thirty energies failing at once with no indication why.

### Degenerate Limits

Sankoff collapses to two simpler problems, and both answers come from outside this project.
Give it a pair table where nothing can pair and it is Needleman-Wunsch with linear gaps, which BioPython answers — and the emitted rows must be a member of BioPython's own set of optimal alignments, not merely score the same.
Give it a gap nobody can afford and a sequence against itself, and it is base-pair maximization scored twice, which an independently written Nussinov answers.

### Constructions With a Provable Answer

A planted stem whose optimum is arithmetic, because each row holds exactly four G and four C, the loop is all A, and no U exists anywhere.
A forced gap inside an unpairable stretch, which must move the optimum by the gap price and nothing else.
Gap monotonicity, since a harsher price cannot raise a maximum.
Concatenation across a spacer as an __inequality__, never an equality: the optimum was observed pairing across the spacer, which the side-by-side solution cannot express.

### Symmetry, Levenshtein and Gap Expansion

```bash
pytest test.py -s -x -k symmetry        # argument order cannot change a score
pytest test.py -s -x -k levenshtein     # the negated distance, at the matching costs
pytest test.py -s -x -k gap_expansions  # a free-extension gap cannot change price with its width
pytest test.py -s -x -k biopython       # every score against BioPython, curated and fuzzed
```

## EMBOSS and Other Tools

Seemingly the only correct known open-source implementation is located in `nucleus/embaln.c` file in the EMBOSS package in the `embAlignPathCalcWithEndGapPenalties` and `embAlignGetScoreNWMatrix` functions.
That program was originally [implemented in 1999 by Alan Bleasby](https://www.bioinformatics.nl/cgi-bin/emboss/help/needle) and tweaked in 2000 for better scoring.
That implementation has no SIMD optimizations, branchless-computing tricks, or other modern optimizations, but it's still widely recommended.
If you want to compare the results, you can download the EMBOSS source code and compile it with following commands:

```bash
wget -m 'ftp://emboss.open-bio.org/pub/EMBOSS/'
cd emboss.open-bio.org/pub/EMBOSS/
gunzip EMBOSS-latest.tar.gz
tar xf EMBOSS-latest.tar
cd EMBOSS-latest
./configure
```

Or if you simply want to explore the source:

```bash
cat emboss.open-bio.org/pub/EMBOSS/EMBOSS-6.6.0/nucleus/embaln.c
```
