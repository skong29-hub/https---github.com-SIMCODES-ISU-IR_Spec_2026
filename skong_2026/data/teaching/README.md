# Teaching subset

The small, committed slice of the dataset that `notebooks/teaching_notebook.ipynb` runs on. It
exists so the notebook works on a fresh clone, with no raw xTB tables.

## Files

| Path | Size | Contents |
| --- | --- | --- |
| `molecules.csv` | ~1 MB | 360 rows: `amino_acid, metal, length, provenance, total_free_energy` then the 791 raw stick intensities `spec_50 … spec_4000` |

Exactly the same columns as `data/processed/molecules.csv`, so one loader reads both. Broadening
is applied at read time, not baked into the file.

## How to load it

```python
from irspectra.data import processed
meta, X, sticks = processed.load_teaching()
```

## What is in it

360 molecules drawn from `data/processed/` (i.e. the lowest-free-energy conformer of each
molecule), stratified over peptide length and metal:

| length | Co2+ | Ni2+ | Cu2+ | Zn2+ | apo |
| --- | --- | --- | --- | --- | --- |
| 1 (monomer) | 20 | 20 | 20 | 20 | 20 |
| 2 (dimer) | 20 | 20 | 20 | 20 | 20 |
| 3 (trimer) | 40 | 40 | 40 | 40 | 0 |

Trimers get the largest share because they are the zero-shot transfer target in Steps 5 and 6.
The full trimer cohort contains no apo peptides, so neither does this one. All monomers are
included, since the monomer set is only 100 molecules in total.

## How it was built

```bash
python tools/make_teaching_subset.py     # needs data/processed/ to exist
```

The sample is drawn with a fixed seed (`SEED = 20260805` in that script), so re-running it
reproduces the same 360 molecules. Change `PER_METAL` there to make the subset bigger or smaller.

## What it is not

**One spectrum per molecule, not per conformer.** The full study represents each molecule by all
of its converged conformers (~25,000 spectra for monomers + dimers alone) and must therefore
group by molecule when splitting, so conformers of the same molecule never straddle a train/test
boundary. This subset has a single spectrum per molecule, so grouping is a no-op here.

That simplification costs accuracy but not the story. On this subset you should see roughly
**0.77 macro-F1** in-domain and **0.52** zero-shot on trimers, against **0.81** and **0.63** for
the full per-conformer pipeline. Chance is 0.25. The length drop — the point of the project — is
clearly visible either way.

Section 4 of the teaching notebook explains the grouping requirement, and Exercise 5 walks
through switching to `conformers.load_conformers()`, where it becomes mandatory.
