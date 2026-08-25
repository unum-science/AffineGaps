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
uv run pytest test.py            # To run the tests
```

## House Style

Arm the hooks once per clone, before the first commit:

```sh
pixi run hooks    # git config core.hooksPath .githooks
pixi run format   # mojo format -l 120 *.mojo
```

`pre-commit` gates `mojo format`, `black` and `ruff` on the staged tree, and a set of prose rules — big-O notation, comment rulers, backticked identifiers in prose, footnote spacing in tables.
The hook is the list, so it grows without this file having to.
`commit-msg` holds the subject to `Fix:`, `Add:`, `Improve:`, `Chore:`, `Make:`, `Docs:` or `Break:`.
The Mojo gate fails closed when the formatter cannot be resolved, because outside the pixi environment `mojo format` warns, changes nothing and still exits zero.
`.githooks/selftest` gives every check a fixture that must be rejected and one that must pass, so a rule that stops firing fails the suite rather than disappearing quietly.

## Testing the Mojo Backend

The GPU kernels are optional. Build them with `pixi run build` and the same suite picks them up; without a build, every Mojo test skips and the pure-Python suite still runs.

```sh
pixi run test                                  # builds the extension, then runs the suite against it
AFFINEGAPS_BACKENDS=python-cpu pixi run pytest # the reference alone, which is what CI runs
```

`-k` is not a substitute for that variable.
It filters on test identifiers, so it also drops every test that has no backend axis — the command line, the brute-force oracles, ViennaRNA, the library surface — where the variable narrows the axis and leaves those tests alone.
It is silent about a typo too, where a misspelled `AFFINEGAPS_BACKENDS` refuses to collect at all.

The backends are named `python-cpu`, `numba-cpu`, `mojo-cpu` and `mojo-gpu`, which are the values that variable takes.
A backend the machine cannot serve is skipped rather than failed, and the skip reason carries the real cause — a missing build and an unsupported driver are different problems and say so.


## Knobs

Environment variables shape a run, and the first two are deliberately separate: one is breadth, the other is depth.

| Variable                 | Default     | Meaning                                        |
| :----------------------- | :---------- | :--------------------------------------------- |
| `AFFINEGAPS_REPETITIONS` | `10`        | Random draws per randomized test               |
| `AFFINEGAPS_SCALE`       | `1`         | Multiplier on every exhaustive oracle's budget |
| `AFFINEGAPS_SEED`        | `0`         | Base for every draw                            |
| `AFFINEGAPS_BACKENDS`    | all of them | Narrows the backend axis                       |

`AFFINEGAPS_REPETITIONS=100 AFFINEGAPS_SCALE=1` is a fuzzing run and `AFFINEGAPS_REPETITIONS=1 AFFINEGAPS_SCALE=4` is a release gate, which is why one number cannot express both.
The seed is applied on every run rather than only when it is named, so each backend at one repeat step is handed the same input and a disagreement between two of them is found by construction rather than by luck.
`AFFINEGAPS_SEED=7 pytest test.py` is the run that goes looking somewhere new.
Name the number rather than drawing one: nothing echoes the seed back, so a drawn one leaves a failure nobody can reproduce.

## Running a Subset

```bash
pytest test.py -k enumeration     # the brute-force oracle for all three recurrences
pytest test.py -k symmetry        # argument order, on both recurrences that have it
pytest test.py -k levenshtein     # the unit-cost limit
pytest test.py -k gap_expansions  # free-extension gap widths
pytest test.py -k biopython       # the external alignment oracle
pytest test.py -k cofold          # the Sankoff recurrence alone
```

What each test asserts, and the trap it exists to catch, is in that test's own docstring.
`test.py` is the document; nothing about the suite's methodology is restated here, because a second copy is a copy that drifts.

## Benchmarks

`bench.py` is the whole measurement apparatus, and it reads `data/archive-ii` rather than anything vendored.
Its own docstring carries what it does with that corpus and why.

```sh
pixi run accuracy       # ArchiveII F1 against RNAstructure, per family
pixi run speed          # the folding ladder against both reference folders
python bench.py accuracy --cofold-pairs 20   # homologous pairs, scored as two structures
```

`--objective` decides whether `Fold` runs at its own defaults or at the model this project implements, and the two answers differ enough that quoting one without naming it would be misleading.
