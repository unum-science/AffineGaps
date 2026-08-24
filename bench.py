"""
The benchmark and accuracy suite: one corpus reader, one metric, one set of rival invocations.

Accuracy is measured on ArchiveII, reading `.ct` and `.lis` only. The `.seq` files beside them
carry truncated and mislabelled titles and a dirtier alphabet than the structures they describe,
so every sequence is reconstructed from its own reference.

Folding scores one predicted structure against the molecule's measured one. Cofolding scores two,
because Sankoff consumes a homologous pair and emits a structure for each, which is how the
comparable tools are evaluated and needs no reference alignment.

Every third-party flag here was chosen from that tool's own help or source. The reasoning lives in
the plan; what matters at the call site is that two objectives are matched deliberately rather
than by a default that happened to agree.

    python bench.py accuracy --per-family 120 --backend mojo --device gpu
    python bench.py speed --lengths 128,256,512
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path

import numpy as np
from seqfold import dot_bracket, fold as seqfold_fold

import affinegaps
import cofolding
import folding
from common import Backend, Device
from turner import PAIR_INDEX

default_alphabet = "ACGU"
"""The four RNA bases, ordered so a base doubles as its own index."""

default_archive_root = Path("data/archive-ii")
"""Where the corpus is reached from, as a symlink onto the shared filesystem."""

bench_prefix = Path(__file__).resolve().parent / ".pixi" / "envs" / "bench"
"""Where the reference folders are installed, as their own pixi environment."""

data_tables = bench_prefix / "share" / "rnastructure" / "data_tables"
"""RNAstructure aborts without this on `DATAPATH`, `scorer` included."""

default_seed: int = 20260824
"""Fixes the draw, because the published table recorded none and its subsets are unrecoverable."""

default_length_limit: int = 400
"""Longest sequence the published accuracy table admits, recovered from its own per-family counts."""

default_family_sample: int = 120
"""Most sequences drawn from any one family, so a large family cannot dominate the average."""

accuracy_families: tuple[str, ...] = ("tRNA", "5s", "srp", "RNaseP", "tmRNA", "grp1")
"""The six families a 400 nt cap leaves with enough sequences to average over."""

cofolding_families: tuple[str, ...] = ("tRNA", "5s", "srp")
"""The three with no pseudoknots at all, and short enough for the Sankoff sweep to reach."""


# region Corpus


@dataclass(frozen=True)
class Structure:
    """One reference molecule: its sequence and the pairing that was measured for it."""

    name: str
    """The filename stem, which is the only reliable identifier the corpus carries."""
    family: str
    """Which of the ten families the stem belongs to."""
    sequence: str
    """Uppercase ACGU, reconstructed from the structure's own second column."""
    partners: tuple[int, ...]
    """Zero-based partner of each position, or `-1` where the position is unpaired."""

    def pairs(self) -> frozenset[tuple[int, int]]:
        """Every base pair once, ordered so the opening position comes first."""
        return frozenset((i, j) for i, j in enumerate(self.partners) if 0 <= i < j)


def read_structure(path: Path) -> Structure:
    """Parses one `.ct` file, raising rather than guessing when the file disagrees with itself."""
    lines = path.read_text(errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"{path.name} is empty")

    # The 37 telomerase files are space-aligned where the other 3,938 are tab-separated.
    declared = int(lines[0].split(maxsplit=1)[0])
    bases: list[str] = []
    partners: list[int] = []
    for line in lines[1 : declared + 1]:
        fields = line.split()
        if len(fields) < 6:
            raise ValueError(f"{path.name} row {len(bases) + 1} has {len(fields)} columns, expected 6")
        bases.append(fields[1].upper())
        partners.append(int(fields[4]) - 1)

    if len(partners) != declared:
        raise ValueError(f"{path.name} declares {declared} bases and holds {len(partners)}")
    for position, partner in enumerate(partners):
        if partner >= 0 and partners[partner] != position:
            raise ValueError(f"{path.name} pairs {position + 1} to {partner + 1} but not back")

    sequence = "".join(bases)
    unknown = set(sequence) - set("ACGU")
    if unknown:
        raise ValueError(f"{path.name} holds letters outside ACGU: {''.join(sorted(unknown))}")

    stem = path.stem
    return Structure(name=stem, family=stem.split("_", 1)[0], sequence=sequence, partners=tuple(partners))


def read_family_index(family: str, root: Path = default_archive_root) -> tuple[str, ...]:
    """The stems one family admits, which is narrower than globbing `.ct` and deliberately so.

    The 35 files outside every index are superseded whole-molecule 16S and 23S entries and the five
    with no base pairs at all, on which sensitivity has no denominator.
    """
    return tuple((root / f"{family}.lis").read_text().split())


def sample_family(
    family: str,
    *,
    root: Path = default_archive_root,
    length_limit: int = default_length_limit,
    count: int = default_family_sample,
    seed: int,
) -> list[Structure]:
    """A reproducible subset of one family: index membership, then a length cap, then a draw.

    The seed is required because the published table was drawn without recording one, and its exact
    subsets are therefore unrecoverable.
    """
    admitted = [read_structure(root / f"{stem}.ct") for stem in read_family_index(family, root)]
    admitted = sorted((e for e in admitted if len(e.sequence) <= length_limit), key=lambda e: e.name)
    if len(admitted) <= count:
        return admitted
    return random.Random(seed).sample(admitted, count)


# endregion Corpus

# region Metrics


class PairMatching(StrEnum):
    """How a predicted pair is matched against a reference pair."""

    EXACT = "exact"
    """Both ends must land on the reference positions."""
    FLEXIBLE = "flexible"
    """Either end may slip by one, which is what `scorer` counts by default."""


def pairs_of_dot_bracket(structure: str) -> frozenset[tuple[int, int]]:
    """The pairs a dot-bracket string names, which is what both recurrences emit."""
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for position, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(position)
        elif symbol == ")":
            if not stack:
                raise ValueError("Unbalanced dot-bracket structure")
            pairs.append((stack.pop(), position))
    if stack:
        raise ValueError("Unbalanced dot-bracket structure")
    return frozenset(pairs)


def crossing_pairs(pairs: frozenset[tuple[int, int]]) -> int:
    """How many pairs participate in a crossing, which is what a nested recurrence cannot reach."""
    ordered = sorted(pairs)
    crossing = set()
    for index, (opening, closing) in enumerate(ordered):
        for later_opening, later_closing in ordered[index + 1 :]:
            if opening < later_opening < closing < later_closing:
                crossing.add((opening, closing))
                crossing.add((later_opening, later_closing))
    return len(crossing)


def _matches(pair: tuple[int, int], against: frozenset[tuple[int, int]], matching: PairMatching) -> bool:
    """Whether one pair finds a counterpart, allowing one position of slippage when asked."""
    i, j = pair
    if matching is PairMatching.EXACT:
        return pair in against
    return any(candidate in against for candidate in ((i, j), (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)))


@dataclass(frozen=True)
class Comparison:
    """One predicted structure scored against one reference.

    The two numerators differ under flexible matching, because a reference pair covered by some
    prediction and a predicted pair covering some reference are counted separately.
    """

    found: int
    """Reference pairs that some predicted pair reaches."""
    reference_total: int
    """Pairs the reference holds, which is sensitivity's denominator."""
    covered: int
    """Predicted pairs that reach some reference pair."""
    predicted_total: int
    """Pairs the prediction holds, which is precision's denominator."""

    @property
    def sensitivity(self) -> float:
        """Share of the reference the prediction recovered, or zero when the reference is empty."""
        return self.found / self.reference_total if self.reference_total else 0.0

    @property
    def precision(self) -> float:
        """Share of the prediction the reference supports, or zero when nothing was predicted."""
        return self.covered / self.predicted_total if self.predicted_total else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of the two, which is zero whenever either one is."""
        total = self.sensitivity + self.precision
        return 2 * self.sensitivity * self.precision / total if total else 0.0


def compare(
    predicted: frozenset[tuple[int, int]],
    reference: frozenset[tuple[int, int]],
    matching: PairMatching = PairMatching.FLEXIBLE,
) -> Comparison:
    """Scores one structure against another, counting both numerators separately."""
    return Comparison(
        found=sum(1 for pair in reference if _matches(pair, predicted, matching)),
        reference_total=len(reference),
        covered=sum(1 for pair in predicted if _matches(pair, reference, matching)),
        predicted_total=len(predicted),
    )


def _mean(values: list[float]) -> float:
    """Arithmetic mean, or zero over nothing."""
    return sum(values) / len(values) if values else 0.0


def _standard_error(values: list[float]) -> float:
    """Standard error of a mean, or zero when a single observation cannot express one."""
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


# endregion Metrics

# region Rivals


class FoldObjective(StrEnum):
    """Which energy function `Fold` is asked for."""

    PUBLISHED = "published"
    """Its own defaults, which price coaxial stacking and forbid lonely pairs."""
    MATCHED = "matched"
    """Coaxial stacking off and lonely pairs allowed, which is this project's model."""


def _binary(name: str) -> Path:
    """Locates one reference tool, preferring the bench environment over anything on `PATH`."""
    local = bench_prefix / "bin" / name
    if local.exists():
        return local
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(f"{name} is not installed; `pixi install -e bench` provides it")
    return Path(found)


def _environment() -> dict[str, str]:
    """The tools' environment, which RNAstructure needs pointing at its tables."""
    return {**os.environ, "DATAPATH": str(data_tables)}


def write_sequence(path: Path, name: str, sequence: str) -> Path:
    """Writes RNAstructure's three-line `.seq`, regenerated clean rather than copied.

    The corpus's own `.seq` files carry lowercase letters, which RNAstructure reads as an
    instruction to forbid those bases from pairing.
    """
    path.write_text(f";\n{name}\n{sequence}1\n")
    return path


def version_of(name: str) -> str:
    """What a tool reports about itself, recorded beside every measurement it produces."""
    binary = _binary(name)
    for flag in ("--version", "-V", "--help"):
        result = subprocess.run([str(binary), flag], env=_environment(), capture_output=True, text=True)
        match = re.search(r"\b\d+\.\d+(?:\.\d+)?\b", f"{result.stdout}\n{result.stderr}")
        if match:
            return match.group(0)
    return "unknown"


def dot_bracket_of_ct(path: Path) -> str:
    """Reads the first structure out of a `.ct` file, ignoring any suboptimal ones after it."""
    lines = path.read_text(errors="ignore").splitlines()
    declared = int(lines[0].split(maxsplit=1)[0])
    symbols = ["."] * declared
    for line in lines[1 : declared + 1]:
        fields = line.split()
        position, partner = int(fields[0]) - 1, int(fields[4]) - 1
        if partner >= 0:
            symbols[position] = "(" if position < partner else ")"
    return "".join(symbols)


def fold_with_rnastructure(
    sequence: str,
    name: str,
    workspace: Path,
    objective: FoldObjective = FoldObjective.MATCHED,
) -> str:
    """Folds one sequence with `Fold`, returning dot-bracket over the same positions."""
    source = write_sequence(workspace / f"{name}.seq", name, sequence)
    target = workspace / f"{name}.ct"
    command = [str(_binary("Fold")), str(source), str(target), "-mfe"]
    if objective is FoldObjective.MATCHED:
        # `--disablecoax` drops the one term that costs; `-i` stops it forbidding lonely pairs.
        command += ["--disablecoax", "-i"]
    subprocess.run(command, env=_environment(), capture_output=True, text=True, check=True)
    return dot_bracket_of_ct(target)


def fold_with_viennarna(sequence: str) -> str:
    """Folds one sequence with `RNAfold` at its default dangle model, which carries no coax."""
    # `--noDP` is rejected unless `--partfunc` is also given, so only the plot is suppressed.
    result = subprocess.run(
        [str(_binary("RNAfold")), "-d2", "--noPS"],
        input=f"{sequence}\n",
        env=_environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        candidate = line.split(" ", 1)[0]
        if candidate and set(candidate) <= set(".()"):
            return candidate
    raise ValueError("RNAfold emitted no structure")


def fold_with_seqfold(sequence: str) -> str:
    """Folds one sequence with SeqFold, which reads RNA and DNA and picks its tables from that.

    Feeding it thymine silently selects the DNA model, so a sequence meant as RNA must spell uracil.
    """
    return dot_bracket(sequence, seqfold_fold(sequence))


def cofold_with_dynalign(
    first: str,
    second: str,
    workspace: Path,
    *,
    separation: int | None = None,
    gap_penalty: float = 0.4,
) -> tuple[str, str]:
    """Runs `dynalign` unbanded and unpruned, which is slower than its defaults and is the point.

    `singlefold_subopt_percent` is raised because the default folds each sequence alone first and
    forbids every pair outside 30 percent of that optimum; the value is read into a `short`, so a
    large finite number is the practical setting. `gap` and `maxpercent` are not keys, whatever
    they look like: the real names are `fgap` and `percent`.
    """
    first_path = write_sequence(workspace / "first.seq", "first", first)
    second_path = write_sequence(workspace / "second.seq", "second", second)
    first_target, second_target = workspace / "first.ct", workspace / "second.ct"
    configuration = workspace / "dynalign.conf"
    configuration.write_text(
        "\n".join(
            (
                f"inseq1 = {first_path}",
                f"inseq2 = {second_path}",
                f"outct = {first_target}",
                f"outct2 = {second_target}",
                f"aout = {workspace / 'alignment.ali'}",
                f"imaxseparation = {separation if separation is not None else max(len(first), len(second))}",
                f"fgap = {gap_penalty}",
                "singlefold_subopt_percent = 100",
                "maxtrace = 1",
                "optimal_only = 1",
                "num_processors = 1",
            )
        )
        + "\n"
    )
    subprocess.run(
        [str(_binary("dynalign")), str(configuration)], env=_environment(), capture_output=True, text=True, check=True
    )
    return dot_bracket_of_ct(first_target), dot_bracket_of_ct(second_target)


# endregion Rivals

# region Accuracy


@dataclass(frozen=True)
class FamilyAccuracy:
    """One family's verdict, for one contestant against the references."""

    family: str
    """Which family was drawn from."""
    sequences: int
    """How many structures were scored."""
    sensitivity: float
    """Mean share of reference pairs recovered."""
    precision: float
    """Mean share of predicted pairs the reference supports."""
    f1: float
    """Mean harmonic mean, averaged per sequence rather than over pooled counts."""
    unreachable_pairs: float
    """Mean share of reference pairs in a crossing, which no nested recurrence can produce."""


@dataclass(frozen=True)
class FamilyDifference:
    """How far apart two contestants sat on one family, with the uncertainty of that verdict."""

    family: str
    """Which family was drawn from."""
    difference: float
    """Mean per-sequence F1 difference, positive where this project won."""
    standard_error: float
    """How far the verdict would move on another sample of the same size."""


def score_folding(
    structures: list[Structure],
    predictions: list[str],
    matching: PairMatching = PairMatching.FLEXIBLE,
) -> tuple[FamilyAccuracy, list[float]]:
    """Scores one contestant's structures against their references, keeping the per-sequence F1."""
    sensitivities, precisions, scores, unreachable = [], [], [], []
    for reference, predicted in zip(structures, predictions, strict=True):
        truth = reference.pairs()
        verdict = compare(pairs_of_dot_bracket(predicted), truth, matching)
        sensitivities.append(verdict.sensitivity)
        precisions.append(verdict.precision)
        scores.append(verdict.f1)
        unreachable.append(crossing_pairs(truth) / len(truth) if truth else 0.0)
    summary = FamilyAccuracy(
        family=structures[0].family if structures else "",
        sequences=len(structures),
        sensitivity=_mean(sensitivities),
        precision=_mean(precisions),
        f1=_mean(scores),
        unreachable_pairs=_mean(unreachable),
    )
    return summary, scores


def project_structure(gapped: str, structure: str) -> str:
    """Maps a structure over alignment columns onto one sequence's own positions.

    A pair survives only where neither of its columns is a gap in this sequence, which is what
    makes the two halves of a Sankoff answer separately scoreable.
    """
    column_of_position = [column for column, letter in enumerate(gapped) if letter != "-"]
    position_of_column = {column: position for position, column in enumerate(column_of_position)}
    symbols = ["."] * len(column_of_position)
    for opening, closing in pairs_of_dot_bracket(structure):
        if opening in position_of_column and closing in position_of_column:
            symbols[position_of_column[opening]] = "("
            symbols[position_of_column[closing]] = ")"
    return "".join(symbols)


def sweep_folding(options, fold_one) -> dict:
    """Folds every sampled sequence with both contestants and reports the gap between them."""
    ours, theirs, differences = [], [], []
    with tempfile.TemporaryDirectory() as scratch:
        workspace = Path(scratch)
        for family in accuracy_families:
            drawn = sample_family(
                family,
                root=options.root,
                length_limit=options.length_limit,
                count=options.per_family,
                seed=options.seed,
            )
            if not drawn:
                continue
            mine = [fold_one(entry.sequence) for entry in drawn]
            rival = [fold_with_rnastructure(e.sequence, e.name, workspace, options.objective) for e in drawn]
            my_summary, my_scores = score_folding(drawn, mine, options.matching)
            their_summary, their_scores = score_folding(drawn, rival, options.matching)
            gaps = [a - b for a, b in zip(my_scores, their_scores, strict=True)]
            ours.append(my_summary)
            theirs.append(their_summary)
            differences.append(FamilyDifference(family, _mean(gaps), _standard_error(gaps)))
            print(
                f"{family:>10}  n={my_summary.sequences:<4} ours F1={my_summary.f1:.3f}  "
                f"Fold F1={their_summary.f1:.3f}  delta={_mean(gaps):+.3f} +/- {_standard_error(gaps):.3f}  "
                f"unreachable={my_summary.unreachable_pairs:.1%}",
                flush=True,
            )
    return {
        "affinegaps": [asdict(entry) for entry in ours],
        "rnastructure": [asdict(entry) for entry in theirs],
        "difference": [asdict(entry) for entry in differences],
    }


def sweep_cofolding(options, cofold_two) -> dict:
    """Cofolds homologous pairs and scores both emitted structures against their own references."""
    results = []
    for family in cofolding_families:
        drawn = sample_family(
            family, root=options.root, length_limit=options.cofold_limit, count=default_family_sample, seed=options.seed
        )
        if len(drawn) < 2:
            continue
        generator = random.Random(options.seed + len(family))
        scores = []
        for _ in range(options.cofold_pairs):
            first, second = generator.sample(drawn, 2)
            gapped_first, gapped_second, structure, _ = cofold_two(first.sequence, second.sequence)
            for entry, gapped in ((first, gapped_first), (second, gapped_second)):
                predicted = pairs_of_dot_bracket(project_structure(gapped, structure))
                scores.append(compare(predicted, entry.pairs(), options.matching).f1)
        results.append(FamilyAccuracy(family, len(scores), 0.0, 0.0, _mean(scores), 0.0))
        print(f"{family:>10}  structures={len(scores):<4} cofold F1={_mean(scores):.3f}", flush=True)
    return {"affinegaps_cofold": [asdict(entry) for entry in results]}


# endregion Accuracy


# region Accelerator


class GpuPolicy(StrEnum):
    """Whether a measurement may share the card with somebody else's work."""

    REQUIRE_IDLE = "require-idle"
    """Refuse to measure while another process holds the GPU, which is the only honest default."""
    SHARE = "share"
    """Measure anyway and record the contention, for a run whose numbers are not being published."""


@dataclass(frozen=True)
class GpuActivity:
    """What the card was actually doing while a rung ran, sampled from DCGM rather than inferred.

    A wall clock cannot tell a slow kernel from a busy neighbour or from a card that never left its
    idle clock, and both have produced wrong numbers here.
    """

    sm_activity: float
    """Share of elapsed time at least one warp was resident on an SM."""
    sm_occupancy: float
    """Share of the resident warp slots that were filled, which is the real parallelism."""
    dram_activity: float
    """Share of elapsed time the memory interface was moving data."""
    power_watts: float
    """Board power, which reads as throttling long before the clock does."""
    clock_mhz: int
    """Streaming-multiprocessor clock, to catch a rung measured from an idle card."""


def _nvidia_query(fields: str) -> list[str]:
    """One `nvidia-smi` query, as the rows it printed."""
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def foreign_gpu_processes() -> list[str]:
    """Compute processes on the card that are not this one."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    mine = str(os.getpid())
    found = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid = line.split(",", 1)[0].strip()
        if pid != mine:
            found.append(line.strip())
    return found


def measure_when_idle(policy: GpuPolicy, call, attempts: int = 3):
    """Runs one rung, and runs it again if somebody joined the card while it was running.

    The guard at the start of a sweep cannot see a neighbour that arrives ten minutes in, and a
    single poisoned rung is worse than a missing one because it still looks like a measurement.
    """
    for attempt in range(attempts):
        before = set(foreign_gpu_processes())
        answer = call()
        after = set(foreign_gpu_processes())
        if policy is GpuPolicy.SHARE or not (before | after):
            return answer
        print(f"  the card was shared during that rung, retrying ({attempt + 1}/{attempts})", flush=True)
    raise RuntimeError("the GPU never stayed idle long enough to measure; pass --gpu-policy share to accept it")


def require_idle_gpu(policy: GpuPolicy, *, utilization_ceiling: int = 10) -> None:
    """Refuses to measure a card somebody else is using, because a shared clock is not a result."""
    if policy is GpuPolicy.SHARE:
        return
    intruders = foreign_gpu_processes()
    if intruders:
        listing = "\n  ".join(intruders)
        raise RuntimeError(
            f"another process is on the GPU, so these numbers would measure it too:\n  {listing}\n"
            "Wait for it, or pass --gpu-policy share to measure anyway."
        )
    rows = _nvidia_query("utilization.gpu")
    busy = [int(row) for row in rows if row.isdigit() and int(row) > utilization_ceiling]
    if busy:
        raise RuntimeError(
            f"the GPU is at {busy[0]}% with no compute process visible, which usually means a "
            "graphics client or a container neighbour. Pass --gpu-policy share to measure anyway."
        )


def ramp_clocks(fold_one, seconds: float = 10.0) -> int:
    """Runs the recurrence until the card leaves its idle clock, and reports where it settled.

    An H100 idles near 345 MHz against a 1980 MHz ceiling, so a short rung measured from cold reads
    as much as twice its warm cost.
    """
    sequence = "".join(random.Random(1).choice(default_alphabet) for _ in range(2048))
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        fold_one(sequence)
    rows = _nvidia_query("clocks.sm")
    return int(rows[0]) if rows and rows[0].isdigit() else 0


def sample_activity(call, seconds_hint: float) -> tuple[float, GpuActivity | None]:
    """Times one call while DCGM watches the card, so a number arrives with its own witness."""
    fields = "1002,1003,1005,155,100"  # SM activity, occupancy, DRAM activity, power, SM clock
    sampler = None
    if shutil.which("dcgmi") is not None and seconds_hint > 0.2:
        sampler = subprocess.Popen(
            ["dcgmi", "dmon", "-e", fields, "-d", "100"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    started = time.perf_counter()
    call()
    elapsed = time.perf_counter() - started
    if sampler is None:
        return elapsed, None
    sampler.terminate()
    readings = []
    for line in (sampler.stdout.read() if sampler.stdout else "").splitlines():
        parts = line.split()
        if len(parts) >= 7 and parts[0] == "GPU":
            try:
                readings.append([float(entry) for entry in parts[2:7]])
            except ValueError:
                continue
    if not readings:
        return elapsed, None
    columns = list(zip(*readings, strict=True))
    means = [sum(column) / len(column) for column in columns]
    return elapsed, GpuActivity(
        sm_activity=means[0],
        sm_occupancy=means[1],
        dram_activity=means[2],
        power_watts=means[3],
        clock_mhz=int(means[4]),
    )


# endregion Accelerator


# region Candidates

# One evaluation of the innermost recurrence, which is what a rate is per. It is not one table
# entry: a folding cell weighs hundreds of interior loops and a Sankoff cell scans a partner pair.
# The count is a property of the sequence and the recurrence, so a third-party clock divided by it
# is that tool rated against this project's work rather than its own.


def _pairable(codes, minimum_span: int):
    """Which ordered position pairs may close, once the loop floor is applied."""
    length = codes.shape[0]
    reach = np.subtract.outer(np.arange(length), np.arange(length))
    return (PAIR_INDEX[codes[:, None], codes[None, :]] >= 0) & (-reach >= minimum_span)


def _partner_visits(closing) -> int:
    """How many partner reads a table indexed by start and window performs in total.

    One start reaches a partner once per window long enough to contain it, so a partner at `p`
    is read `length - p` times.
    """
    length = closing.shape[0]
    return int((closing * (length - np.arange(length))[None, :]).sum())


def folding_candidates(sequence: str) -> int:
    """Innermost evaluations one fold performs: the interior scan, then the partner scans.

    The interior scan is counted from its bounds over every cell rather than only over cells whose
    ends can pair, which is how the published ladders were rated and what keeps a rate comparable
    across them. `multiloop` and `closable` each walk a cell's partners; `exterior` is one cell per
    start, so it walks each start's partners once.
    """
    length = len(sequence)
    codes = np.array([default_alphabet.index(letter) for letter in sequence], dtype=np.int64)

    windows = np.arange(1, length + 1)
    room = np.minimum(folding.MAX_LOOP, windows - 2 - folding.MIN_PAIRED_WINDOW)
    combinations = np.where(room >= 0, (room + 1) * (room + 2) // 2, 0)
    interior = int(((length - windows + 1) * combinations).sum())

    closing = _pairable(codes, folding.MIN_CLOSING_REACH)
    return interior + 2 * _partner_visits(closing) + int(closing.sum())


def cofolding_candidates(first: str, second: str) -> int:
    """Innermost evaluations one Sankoff sweep performs, which is one partner pair per candidate.

    Every cell pairs a window of the first sequence with one of the second, so the total factorises
    into each sequence's own partner visits rather than needing the whole four-dimensional walk.
    """

    def visits(sequence: str) -> int:
        codes = np.array([default_alphabet.index(letter) for letter in sequence], dtype=np.int64)
        pairs = cofolding.default_rna_pair_matrix
        length = codes.shape[0]
        reach = np.subtract.outer(np.arange(length), np.arange(length))
        closing = (pairs[codes[:, None], codes[None, :]] > 0) & (-reach >= cofolding.MIN_CLOSING_REACH)
        return _partner_visits(closing)

    return visits(first) * visits(second)


# endregion Candidates


# region Speed


def _timed(call, repeats: int) -> float:
    """Fastest of a few runs, because a shared machine's slowest run measures its neighbours."""
    best = math.inf
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - started)
    return best


def sweep_cofolding_speed(options, cofold_two, fold_one) -> dict:
    """Times the Sankoff sweep across a ladder of lengths, on a card nobody else is using."""
    require_idle_gpu(options.gpu_policy)
    settled = ramp_clocks(fold_one) if options.device == "gpu" else 0
    if settled:
        print(f"clocks settled at {settled} MHz", flush=True)
    generator = random.Random(options.seed)
    rows = []
    for length in options.lengths:
        first = "".join(generator.choice(default_alphabet) for _ in range(length))
        second = "".join(generator.choice(default_alphabet) for _ in range(length))
        candidates = cofolding_candidates(first, second)
        rung = partial(_timed, partial(cofold_two, first, second), options.repeats)
        taken = measure_when_idle(options.gpu_policy, rung)
        _, activity = sample_activity(partial(cofold_two, first, second), taken)
        row = {"length": length, "candidates": candidates, "affinegaps": taken}
        shown = [
            f"length={length}",
            f"candidates={candidates:,}",
            f"affinegaps={taken:.4g} ({candidates / taken / 1e9:.3g} GCUPS)",
        ]
        if activity is not None:
            row["activity"] = asdict(activity)
            shown.append(
                f"[sm {activity.sm_activity:.2f} occ {activity.sm_occupancy:.2f} "
                f"dram {activity.dram_activity:.2f} {activity.power_watts:.0f}W {activity.clock_mhz}MHz]"
            )
        rows.append(row)
        print("  ".join(shown), flush=True)
    return {"speed": rows}


def sweep_speed(options, fold_one) -> dict:
    """Times folding against both reference folders across a ladder of lengths."""
    require_idle_gpu(options.gpu_policy)
    settled = ramp_clocks(fold_one) if options.device == "gpu" else 0
    if settled:
        print(f"clocks settled at {settled} MHz", flush=True)
    generator = random.Random(options.seed)
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        workspace = Path(scratch)
        for length in options.lengths:
            sequence = "".join(generator.choice(default_alphabet) for _ in range(length))
            candidates = folding_candidates(sequence)
            rung = partial(_timed, partial(fold_one, sequence), options.repeats)
            taken = measure_when_idle(options.gpu_policy, rung)
            _, activity = sample_activity(partial(fold_one, sequence), taken)
            row = {"length": length, "candidates": candidates, "affinegaps": taken}
            if activity is not None:
                row["activity"] = asdict(activity)
            if not options.skip_rivals:
                row["viennarna"] = _timed(partial(fold_with_viennarna, sequence), options.repeats)
                row["rnastructure"] = _timed(
                    partial(fold_with_rnastructure, sequence, "ladder", workspace, options.objective), options.repeats
                )
                if length <= options.seqfold_limit:
                    row["seqfold"] = _timed(partial(fold_with_seqfold, sequence), options.repeats)
            rows.append(row)
            # Every contestant is rated against this project's candidate count, so a rate compares
            # tools on the same work rather than each on its own recurrence.
            shown = [f"length={length}", f"candidates={candidates:,}"]
            for key in ("affinegaps", "viennarna", "rnastructure", "seqfold"):
                if key in row:
                    shown.append(f"{key}={row[key]:.4g} ({candidates / row[key] / 1e9:.3g} GCUPS)")
            if activity is not None:
                shown.append(
                    f"[sm {activity.sm_activity:.2f} occ {activity.sm_occupancy:.2f} "
                    f"dram {activity.dram_activity:.2f} {activity.power_watts:.0f}W {activity.clock_mhz}MHz]"
                )
            print("  ".join(shown), flush=True)
    return {"speed": rows}


# endregion Speed


def main() -> int:
    """Runs one sweep and writes the verdict as JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("accuracy", "speed"))
    parser.add_argument("--root", type=Path, default=default_archive_root)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--length-limit", type=int, default=default_length_limit)
    parser.add_argument("--per-family", type=int, default=default_family_sample)
    parser.add_argument("--cofold-pairs", type=int, default=0, help="homologous pairs per family, 0 to skip")
    parser.add_argument("--cofold-limit", type=int, default=128, help="longest sequence the Sankoff sweep may take")
    parser.add_argument("--objective", type=FoldObjective, choices=list(FoldObjective), default=FoldObjective.MATCHED)
    parser.add_argument("--matching", type=PairMatching, choices=list(PairMatching), default=PairMatching.FLEXIBLE)
    parser.add_argument("--backend", default="python", choices=("python", "numba", "mojo"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--lengths", default="128,256,512,1024", help="comma-separated ladder for the speed mode")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-folding", action="store_true")
    parser.add_argument("--skip-rivals", action="store_true")
    parser.add_argument("--gpu-policy", type=GpuPolicy, choices=list(GpuPolicy), default=GpuPolicy.REQUIRE_IDLE)
    parser.add_argument("--recurrence", default="folding", choices=("folding", "cofolding"))
    # Its cost grows near the fourth power, so it leaves the ladder long before the others do.
    parser.add_argument("--seqfold-limit", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    options.lengths = [int(entry) for entry in options.lengths.split(",") if entry]

    backend, device = Backend(options.backend), Device(options.device)

    def fold_one(sequence: str) -> str:
        return affinegaps.zuker_fold(sequence, backend=backend, device=device)[0]

    def cofold_two(first: str, second: str):
        return affinegaps.sankoff_cofold(first, second, backend=backend, device=device)

    report: dict = {
        "seed": options.seed,
        "backend": f"{options.backend}-{options.device}",
        "objective": str(options.objective),
        "matching": str(options.matching),
        "rnastructure": version_of("Fold"),
        "viennarna": version_of("RNAfold"),
    }
    if options.mode == "speed":
        if options.recurrence == "cofolding":
            report |= sweep_cofolding_speed(options, cofold_two, fold_one)
        else:
            report |= sweep_speed(options, fold_one)
    else:
        report |= {"length_limit": options.length_limit, "per_family": options.per_family}
        if not options.skip_folding:
            report |= sweep_folding(options, fold_one)
        if options.cofold_pairs:
            report |= sweep_cofolding(options, cofold_two)
    if options.output:
        options.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
