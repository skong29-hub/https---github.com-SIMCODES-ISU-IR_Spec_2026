# Data

## What is committed — the dataset every script reads

**No pipeline script ever opens a raw xTB table.** They all read these files, which is what makes
a fresh clone reproducible. Every one has the same layout as the raw tables it replaces: metadata
columns, then the 791 raw stick intensities `spec_50 … spec_4000`.

| Path | Rows | Size | Contents |
| --- | --- | --- | --- |
| `processed/molecules.csv` | 9,060 | ~22 MB | one row per molecule — the lowest-free-energy conformer. `amino_acid, metal, length, provenance, total_free_energy` + `spec_*` |
| `processed/conformers_monodi.csv.gz` | 24,685 | ~13 MB | **every** mono+dimer metal conformer. `amino_acid, metal, length, provenance, total_free_energy` + `spec_*` |
| `processed/conformers_trimers_cap10.csv.gz` | 69,520 | ~44 MB | trimer conformers, capped at 10 per molecule |
| `teaching/molecules.csv` | 360 | ~1 MB | stratified subset for `notebooks/teaching_notebook.ipynb`, same columns as `processed/molecules.csv` |
| `processed/dataset_manifest.json` | — | ~1 kB | what was **simulated** vs what **shipped**, per peptide length, plus the build settings |

### Why the manifest exists

The trimer table ships capped, so counting its rows under-reports what xTB actually produced —
104,169 trimer conformers become 69,520 on disk. Once the build finishes, that pre-cap number
exists nowhere else in the repo. `dataset_manifest.json` records it, so anything quoting a
dataset size (the README, `figures/graphical_overview.svg`) reads the manifest instead of either
hardcoding the number or silently reporting the smaller one. Read it with:

```python
from irspectra.data import processed
man = processed.load_manifest()
man["conformers_simulated_total"]   # 128,854 — the real xTB output
man["conformers_committed_total"]   #  94,205 — what this repo contains
```

### The `provenance` column

Every table carries `provenance`, the quantum-chemistry method behind the spectrum. It is the
constant string `xtb` in all four files today, so it looks like dead weight — it is not. The
project also holds DFT spectra (`dft_*_spectra_*.csv` among the raw tables), and the whole point
of using xTB is that it reaches trimers where DFT stops at dimers. If the two are ever combined,
`provenance` is what keeps a classifier from learning *method* instead of *metal*: xTB and DFT
put their band positions in systematically different places, which is a far stronger signal than
the metal identity we actually want. Carrying the column now means that merge is a concat rather
than a schema migration.

Two notes on the other choices:

- **Why gzip for the conformer tables.** The trimer set is 170 MB as plain text, over GitHub's
  100 MB per-file limit. 83% of every stick spectrum is zeros, so gzip is dramatic.
  `pandas.read_csv` decompresses `.gz` transparently — no loader code knows the difference.
- **Why mono+di is uncapped but trimers are capped at 10.** Step 4's cross-validation is an
  *uncapped* per-conformer run, so the full mono+di set has to be present for the 0.81 headline
  to be reproducible. The full trimer set is 104,169 conformers and does not fit. As a result
  `--cap` on trimers means "cap further": `--cap 10` reproduces the committed `_cap10` outputs
  exactly, and omitting it still gives 10 per molecule.

## What is not committed

The raw xTB tables, which only `tools/build_dataset.py` reads:

| Path | Size |
| --- | --- |
| `energy_all_metal_onemers.csv` | ~4 MB |
| `energy_all_metal_dimers.csv` | ~92 MB |
| `energy_all_metal_trimers.csv` | ~423 MB |
| `energy_all_aa_singles.csv`, `energy_all_aa_dimers.csv` | ~4 MB each |

They are archived separately (Zenodo / institutional storage). Put them directly in `data/` —
i.e. `data/energy_all_metal_dimers.csv`. A `data/raw/` subdirectory is **not** searched.

## Rebuilding the dataset from raw

Only needed if the raw tables change. With them in `data/`:

```bash
python tools/build_dataset.py                  # -> processed/*.csv(.gz)
python tools/make_teaching_subset.py           # -> teaching/molecules.csv
```

`build_dataset.py` is the single point of contact with the raw data. If you point it elsewhere,
pass `--raw-dir`.

## The spectral grid

791 bins, 50 to 4000 cm-1, 5 cm-1 spacing. Broadening is a Gaussian with FWHM 15 cm-1; each spectrum is normalized to unit area.

## Cohort (single spectrum per molecule)

| length | Co2+ | Ni2+ | Cu2+ | Zn2+ | apo |
| --- | --- | --- | --- | --- | --- |
| 1 (monomer) | 20 | 20 | 20 | 20 | 20 |
| 2 (dimer) | 399 | 399 | 399 | 399 | 379 |
| 3 (trimer) | 1751 | 1750 | 1741 | 1743 | 0 |
