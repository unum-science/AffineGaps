# AffineGaps

![Affine Gaps Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/AffineGaps.jpg?raw=true)

__AffineGaps__ collects __less-wrong__ implementations of the classical biosequence dynamic programs, exact and with traceback, on the GPU.
It covers Osamu Gotoh's 1982 affine gap penalty [paper](https://doc.aporc.org/attach/Course001Papers/gotoh1982.pdf) for the Needleman-Wunsch and Smith-Waterman algorithms - with several algorithmic corrections to the original paper, and David Sankoff's 1985 simultaneous alignment and folding of RNA.
Unlike potentially faster algorithms in [StringZilla](https://github.com/ashvardanian/StringZilla), beyond scoring — AffineGaps also reconstructs the alignment strings, __also on the GPU__.
Gotoh reconstructs in __linear memory__; Sankoff provably cannot, and the reason is worth reading below.
A NumPy reference implementation ships beside every Mojo kernel and serves as the parity oracle.

- __`alignment`__ — Gotoh, global and local, in $O(nm)$ time and $O(\min(n, m))$ memory via Hirschberg, reconstructing the aligned sequences.
- __`folding`__ — Zuker minimum free energy over the Turner model, in $O(n^3)$ time and $O(n^2)$ memory, reconstructing a dot-bracket structure.
- __`cofolding`__ — Sankoff, simultaneous alignment and folding, in $O(n^3 m^3)$ time and $O(n^2 m^2)$ memory, reconstructing the aligned sequences plus one shared structure.

## Less Wrong

As reported in the "Are all global alignment algorithms and implementations correct?" [paper](https://www.biorxiv.org/content/10.1101/031500v1.full.pdf) by Tomas Flouri, Kassian Kobert, Torbjørn Rognes, and Alexandros Stamatakis:

> In 1982 Gotoh presented an improved algorithm with lower time complexity.
> Gotoh’s algorithm is frequently cited...
> While implementing the algorithm, we discovered two mathematical mistakes in Gotoh’s paper that induce sub-optimal sequence alignments.
> First, there are minor indexing mistakes in the dynamic programming algorithm which become apparent immediately when implementing the procedure.
> Hence, we report on these for the sake of completeness.
> Second, there is a more profound problem with the dynamic programming matrix initialization.
> This initialization issue can easily be missed and find its way into actual implementations.
> This error is also present in standard text books.
> Namely, the widely used books by Gusfield and Waterman.
> To obtain an initial estimate of the extent to which this error has been propagated, we scrutinized freely available undergraduate lecture slides.
> We found that 8 out of 31 lecture slides contained the mistake, while 16 out of 31 simply omit parts of the initialization, thus giving an incomplete description of the algorithm.
> Finally, by inspecting ten source codes and running respective tests, we found that five implementations were incorrect.

During my exploration of existing implementations, I've noticed several bugs:

- several libraries initialize the header row/columns of penalty matrices with ±∞, causing overflows on the first iteration.
- initialize matrices to zeros, ignoring the first gap opening cost.
- combining opening and expansion costs where only the opening cost should be applied.
- even the most correct `needle` from EMBOSS uses `float` representation, which would obviously be numerically unstable on very long sequences.

## Benchmarks

Each recurrence is measured against the tool people already reach for — WFA2 for alignment, RNAstructure's `Fold` for folding.

### Alignment Speed Against WFA2

One __NVIDIA H100 80GB HBM3__ against __WFA2__ on one CPU core, both reconstructing the alignment rather than only scoring it.
DNA pairs at 10 percent divergence, the range long-read work lives in, with every score checked to agree exactly before timing.

```
                   wfa2 faster  ←│→  affinegaps faster
    1 kbp                       ▍│                          1.1x
   10 kbp                        │████▋                     2.2x
  100 kbp                        │█████████████████        17.2x
  300 kbp                        │██████████████████████   39.8x
    1 Mbp                        │██████████████████████  out of memory
```

The last row is not a timing.
WFA2 reconstructs exactly, and its traceback grows with the square of the alignment score, so a megabase pair at this divergence exhausts tens of gigabytes before finishing, where the linear-space recursion holds a few frontiers and finishes in seconds.

That same growth is what decides every other row, and it makes the crossover a property of the data rather than of the sequence length.
On a 30 kilobase pair, mutated harder each row:

```
                   wfa2 faster  ←│→  affinegaps faster
     0.1%     ███████████████████│                         55.5x
       1%            ███████████▊│                         12.0x
       5%                        │█▊                        1.5x
      10%                        │███████▌                  5.0x
      25%                        │███████████████▊         28.3x
      50%                        │████████████████████▋    79.7x
    99.9%                        │██████████████████████  104.6x
```

A full sweep never notices how different the sequences are, so reach for WFA2 on near-identical inputs and for this on divergent or very long ones.

Linear memory is meant literally, and it is what makes the long end reachable at all.
A four-megabase pair reconstructs inside a gigabyte of device memory, and doubling the pair grows that by a third rather than by four, so an 80-gigabyte card has room for sequences far longer than anything measured here.

### Folding Accuracy Against RNAstructure

Accuracy is reported on __ArchiveII__, the Mathews lab set of 3,975 known structures across ten families that is the standard benchmark for thermodynamic folders.
It plays the role here that BLOSUM62 and Biopython play on the protein side: an external reference this project is measured against rather than tuned on.
The set is not vendored; the harness reads it from `data/archive-ii`.

Both columns score predicted base pairs against the __known structure__ in each file, so `Fold` is a second contestant rather than the target.
F1 is the harmonic mean of sensitivity and positive predictive value, dimensionless and bounded by zero and one.
The last column is the mean per-sequence difference in F1 with its standard error, negative where `Fold` wins, because comparing two averages over different sequences hides more than it shows.

| family         | sequences | AffineGaps F1 | RNAstructure F1 |     ΔF1 ± s.e. |
| :------------- | --------: | ------------: | --------------: | -------------: |
| tRNA           |       120 |         0.571 |           0.677 | −0.107 ± 0.028 |
| 5S rRNA        |       120 |         0.615 |           0.610 | +0.005 ± 0.024 |
| SRP RNA        |       120 |         0.626 |           0.630 | −0.003 ± 0.019 |
| RNase P        |       120 |         0.461 |           0.513 | −0.053 ± 0.013 |
| tmRNA          |       120 |         0.379 |           0.390 | −0.011 ± 0.012 |
| group I intron |        38 |         0.471 |           0.496 | −0.025 ± 0.027 |

__Only tRNA and RNase P differ by more than their uncertainty.__
On the other four families the reduced model is statistically indistinguishable from the complete one.
`Fold`'s own numbers sit where the literature puts the Turner model, so the comparison is calibrated rather than flattering.

## Installation

There is no package-registry release; the repository is the distribution.
Where Mojo ships a toolchain — Linux on x86-64 or ARM, and macOS on Apple silicon — the compiled kernels are built during the install and travel with the package.
Everywhere else you get the NumPy reference alone, from the same command.

### As a Python Package

```bash
uv pip install git+https://github.com/unum-science/AffineGaps.git
uv pip install 'affinegaps[numba] @ git+https://github.com/unum-science/AffineGaps.git'
```

Pin a tag or a commit when the build has to be reproducible:

```bash
uv pip install 'affinegaps @ git+https://github.com/unum-science/AffineGaps.git@v0.2.5'
```

Two optional extras, neither of them required: `numba` accelerates the NumPy reference, and `color` paints the command-line output.

### As a Command-Line Tool

`uv tool install` puts `affinegaps` on your path without touching the current environment:

```bash
uv tool install git+https://github.com/unum-science/AffineGaps.git
```

The same tool is also built natively from a checkout, with no Python involved at run time:

```bash
git clone https://github.com/unum-science/AffineGaps.git && cd AffineGaps
pixi run install      # `affinegaps` on the PATH, compiled kernels included
pixi run build-cli    # or build/affinegaps, a standalone binary
```

Both accept the same verbs and print the same thing, so install whichever suits you rather than both.
To run it once without installing anything, `uvx` fetches, builds and runs it in a throwaway environment:

```bash
uvx --from git+https://github.com/unum-science/AffineGaps.git affinegaps align GATTACA GACTATA
```

### As a Pixi Dependency

A downstream [pixi](https://pixi.sh) project takes the same git dependency, pinned in `pixi.lock` beside everything else:

```bash
pixi add --pypi 'affinegaps @ git+https://github.com/unum-science/AffineGaps.git'
```

### Requirements for the GPU

The compiled kernels run on the CPU anywhere Mojo builds them.
Reaching the GPU additionally needs an NVIDIA device with driver 580 or newer, and `available("mojo", "gpu")` reports whether this machine has one — it tries a real alignment rather than assuming.

## Using the Library

### Backends and Devices

Every entry point runs elsewhere by naming where the work goes.
Two axes: `backend` chooses which implementation, `device` chooses which hardware.
Naming neither picks the fastest the machine offers.

```python
needleman_wunsch_gotoh_alignment(first, second)                     # fastest available
needleman_wunsch_gotoh_alignment(first, second, backend="python")   # the reference
zuker_fold(sequence, backend="mojo")                                # compiled, host
sankoff_cofold(first, second, backend="mojo", device="gpu")
```

Unspecified adapts; specified is honoured or refused.
Asking for a backend that is not built raises rather than quietly running something else, so a measurement can never report the GPU while timing the reference.

There is no knob for how the traceback stores its state.
Below a size threshold it keeps a decision per cell, above it recurses in linear space, and the two produce identical output — so the choice is cost, never correctness.

### Aligning Two Sequences

To obtain the alignment of two sequences, use the `needleman_wunsch_gotoh_alignment` function.

```python
from affinegaps import needleman_wunsch_gotoh_alignment

insulin = "GIVEQCCTSICSLYQLENYCN"
glucagon = "HSQGTFTSDYSKYLDSRAEQDFV"
aligned_insulin, aligned_glucagon, score = needleman_wunsch_gotoh_alignment(insulin, glucagon)

print("Alignment 1:", aligned_insulin)  # ---GIVEQCCTSICSLYQLENYCN----
print("Alignment 2:", aligned_glucagon) # HSQGTF----TSDYSKY-LDSRAEQDFV
print("Score:", score)                  # 22
```

If you only need the alignment score, `needleman_wunsch_gotoh_score` uses less memory and works faster.
Batches are where the device earns its keep, one thread block per pair:

```python
from affinegaps import needleman_wunsch_gotoh_alignments, available

if available("mojo", "gpu"):
    results = needleman_wunsch_gotoh_alignments(firsts, seconds, backend="mojo", device="gpu")
```

By default a BLOSUM62 substitution matrix scaled by five is used.
Costs come in two records: the gap model, and either a uniform pair of scores or a table.

```python
import numpy as np
from affinegaps import AffineGapCosts, UniformSubstitutionCosts, TabulatedSubstitutionCosts

aligned_insulin, aligned_glucagon, score = needleman_wunsch_gotoh_alignment(
    insulin, glucagon,
    substitution=UniformSubstitutionCosts(match=1, mismatch=-1),
    gaps=AffineGapCosts(open=-2, extend=-1),
)

alphabet = "ARNDCQEGHILKMFPSTWYVBZX"
substitutions = np.full((len(alphabet), len(alphabet)), -1, dtype=np.int8)
np.fill_diagonal(substitutions, 1)

aligned_insulin, aligned_glucagon, score = needleman_wunsch_gotoh_alignment(
    insulin, glucagon,
    substitution=TabulatedSubstitutionCosts(alphabet, substitutions),
    gaps=AffineGapCosts(open=-2, extend=-1),
)
```

A uniform cost carries its match and its mismatch together, and a table carries its matrix and the alphabet indexing it, so neither can be half-specified and the two cannot be combined.
That is similar to the following usage example of BioPython:

```python
from Bio import Align
from Bio.Align import substitution_matrices

aligner = Align.PairwiseAligner(mode="global")
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = open_gap_score
aligner.extend_gap_score = extend_gap_score
```

### Folding One Sequence

Zuker's recurrence over the Turner nearest-neighbour model, exact, with the structure reconstructed rather than just its energy.

```python
from affinegaps import zuker_fold

structure, energy = zuker_fold("GGGGCAAAAGCCCC")

print(structure)  # (((((....)))))
print(energy)     # -9.2, kilocalories per mole
```

The recurrence works in integer decikilocalories, so a sequence always folds to the same answer rather than depending on floating-point association order.

The parameters in `turner.py` are the published Turner 2004 constants.
The kernels use __stacking, loop initiation by size, terminal mismatches on hairpins and internal loops, the tabulated tri-, tetra- and hexaloops, dangling ends, Ninio's asymmetry correction, the linear multiloop rule and the terminal AU penalty__.

They do __not__ use coaxial stacking, or the special tables for one-by-one, two-by-one and two-by-two internal loops.
Dangles are charged to both neighbours of every helix placed in an exterior loop or a multiloop, which needs no extra states in the recurrence but slightly over-counts where two helices abut.

So this remains __an exact implementation of a documented model__ rather than a drop-in replacement for a complete Turner folder.
Every number traces to a table you can read, and the remaining omissions — coaxial stacking above all — are additive refinements that fit behind the same interface.

### Aligning and Folding Together

Sankoff's recurrence aligns two RNA sequences and folds them at the same time, crediting a base pair only where __both__ sequences can form it.
The signal is covariation rather than thermodynamics, so no energy model is involved and no parameter tables ship with it.

```python
from affinegaps import sankoff_cofold

first = "GGGGCAAAAGCCCC"
second = "GGGGCUUUUGCCCC"
aligned_first, aligned_second, structure, score = sankoff_cofold(first, second)

print(aligned_first)   # GGGGCAAAAGCCCC
print(aligned_second)  # GGGGCUUUUGCCCC
print(structure)       # (((((....)))))
print(score)           # 46, the optimum of the recurrence rather than a free energy
```

The structure is dot-bracket over the __alignment columns__, so one string describes the pairing both sequences agree on.

The table is indexed by a window of each sequence, so it holds $O(n^2 m^2)$ cells, and __that memory is inherent rather than an implementation limit__.
A bifurcation at one layer reads every layer beneath it, so nothing can ever be retired.
Measured at $n = 24$, even a perfect freeing oracle leaves 75.5% of the table live at peak, which is why no Hirschberg-style band exists here and why the linear-memory claim is scoped to alignment.
The consolation is that __traceback costs nothing extra__: the whole table is resident regardless, so reconstruction is a walk rather than a second pass.

Time is $O(n^3 m^3)$ and binds before memory does.
On one idle H100 the exact sweep is comfortable to roughly 200 bases, which covers tRNA, 5S rRNA, microRNA precursors and most riboswitches, and gets expensive immediately after.
For longer sequences the banded approximations remain the right tool; this one exists to be __exact__, and to be the oracle they can be measured against.

## Using the Command Line

One verb per recurrence, and the three take different shapes:

```bash
$ affinegaps align GIVEQCCTSICSLYQLENYCN HSQGTFTSDYSKYLDSRAEQDFV
$ affinegaps fold GGGGCAAAAGCCCC
$ affinegaps cofold GGGGCAAAAGCCCC GGGGCUUUUGCCCC
```

Folding takes one sequence where the other two take a pair, and alignment's gap is affine where cofolding's is linear, so a single command would have offered `--open` and `--gap` side by side and invited a reader to mix them.

Every verb accepts `--device cpu|gpu`, `--gpu-id`, `--format human|json`, `--color auto|always|never`, `--verbose` and `--help`.
Naming `--gpu-id` is asking for an accelerator, so it settles a device left unnamed and is refused alongside `--device cpu`.
`--backend python|numba|mojo` exists on the Python entry point alone, because the compiled binary is already the backend.
`--verbose` adds the backend and device, the cell count, the elapsed time and the rate in MCUPS — on stderr, so a piped payload stays parseable.

```bash
$ affinegaps fold GGGGCAAAAGCCCC --format json
> {"operation": "fold", "sequence": "GGGGCAAAAGCCCC", "structure": "(((((....)))))", "energy_kcal_per_mol": -9.2, "backend": "mojo", "device": "cpu", "gpu_id": 0}
```

### Aligning Two Sequences

To compute the optimal global alignment of insulin and glucagon sequences with the BLOSUM62 substitution matrix scaled by five:

```bash
$ affinegaps align GIVEQCCTSICSLYQLENYCN HSQGTFTSDYSKYLDSRAEQDFV
>
> Sequence 1:  GIVEQCCTSICSLYQLENYCN
> Sequence 2:  HSQGTFTSDYSKYLDSRAEQDFV
> Alignment 1: ---GIVEQCCTSICSLYQLENYCN----
> Alignment 2: HSQGTF----TSDYSKY-LDSRAEQDFV
> Score:       22
```

To compute the local alignment of the same sequences, pass `--local`.
Only the highest-scoring subalignment comes back, trimmed at both ends:

```bash
$ affinegaps align GIVEQCCTSICSLYQLENYCN HSQGTFTSDYSKYLDSRAEQDFV --local
>
> Sequence 1:  GIVEQCCTSICSLYQLENYCN
> Sequence 2:  HSQGTFTSDYSKYLDSRAEQDFV
> Alignment 1: TSICSLYQLEN
> Alignment 2: TSDYSKY-LDS
> Score:       80
```

`--match` and `--mismatch` replace the scaled BLOSUM62 with a uniform pair and must be given together, `--open` and `--extend` set the affine gap, and `--threads` bounds the host threads the linear-space traceback may fork across.
That last one defaults to the threads this process may actually run on, which an affinity mask or a cgroup quota narrows below the core count, and alignment is the only verb with a parallel region to bound.

### Folding One Sequence

```bash
$ affinegaps fold GGGGCAAAAGCCCC
>
> Sequence:  GGGGCAAAAGCCCC
> Structure: (((((....)))))
> Energy:    -9.2 kcal/mol
```

The verb takes no scoring flags, because the energies come from the Turner tables rather than the command line.

### Aligning and Folding Together

```bash
$ affinegaps cofold GGGGCAAAAGCCCC GGGGCUUUUGCCCC
>
> Sequence 1: GGGGCAAAAGCCCC
> Sequence 2: GGGGCUUUUGCCCC
> Structure:  (((((....)))))
> Score:      46
```

`--match`, `--mismatch` and `--gap` set the covariation scoring.
The gap is linear and there is no `--open`, because Sankoff has no affine model.

## Related Work

Three groups, one per module, each naming what the tool does and where this differs.

### Alignment

- [WFA2](https://github.com/smarco/WFA2-lib) — exact and gap-affine, with time proportional to the alignment score rather than to the product of the lengths. That makes it the faster choice on near-identical sequences and the one that exhausts memory on divergent or very long ones, which is what the benchmark above measures.
- [parasail](https://github.com/jeffdaily/parasail) — vectorised Smith-Waterman and Needleman-Wunsch for the CPU, with no accelerator path.
- [SeqAn](https://github.com/seqan/seqan3) — a general sequence-analysis library whose aligner covers the same recurrences among much else.
- [Biopython](https://biopython.org) — `PairwiseAligner` implements the same recurrences in Python, and is the readability reference rather than the speed one.
- [EMBOSS](http://emboss.open-bio.org) — `needle` is seemingly the only other open-source implementation that gets the initialization right, in `embAlignPathCalcWithEndGapPenalties` and `embAlignGetScoreNWMatrix` inside `nucleus/embaln.c`. It was [written in 1999 by Alan Bleasby](https://www.bioinformatics.nl/cgi-bin/emboss/help/needle) and rescored in 2000, carries no vectorisation, and is still widely recommended. It scores in `float`, which drifts on long sequences.
- [StringZilla](https://github.com/ashvardanian/StringZilla) — faster still, but it scores without reconstructing an alignment.

### Folding

- [RNAstructure](https://rna.urmc.rochester.edu/RNAstructure.html) — the Mathews lab suite whose `Fold` program implements the complete Turner model, including the coaxial stacking and the special internal-loop tables omitted here. It is the contestant in the accuracy table above.
- [ViennaRNA](https://www.tbi.univie.ac.at/RNA/) — `RNAfold` and a partition function over the same thermodynamic model, so it answers how probable a pairing is rather than only which structure is optimal.
- [LinearFold](https://github.com/LinearFold/LinearFold) — beam search in linear time, trading exactness for a sweep that scales to whole messenger RNAs.

### Cofolding

- [LocARNA](https://github.com/s-will/LocARNA) — Sankoff restricted by base-pair probabilities computed beforehand, which is the standard way to make the recurrence affordable.
- Dynalign, shipped inside RNAstructure — Sankoff with a bound on how far the two alignments may diverge, cheap enough for sequences well past the reach of the exact sweep.
- [Foldalign](https://rth.dk/resources/foldalign/) — a banded local variant, aimed at finding shared structured motifs rather than folding whole sequences together.

Every tool in the last group approximates.
This one does not, which is what makes it useful to them: an exact answer at a few hundred bases is the oracle a band can be measured against.
