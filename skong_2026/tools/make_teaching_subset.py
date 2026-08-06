#!/usr/bin/env python3
"""
make_teaching_subset.py -- carve a small, committable subset out of data/processed/ for
notebooks/teaching_notebook.ipynb.

The full molecule table is 9,060 rows x 791 spectral bins (~22 MB), fine to commit but slow to
teach with. This draws a stratified sample -- every monomer, plus a fixed number of dimers and
trimers per metal -- so the teaching notebook demonstrates all six pipeline steps, including the
mono+di -> trimer length transfer that is the point of the project, from a ~1 MB file.

Stratification is by (length, metal) with a fixed seed, so the subset is reproducible and its
class balance is even -- which matters because the full trimer set has no apo peptides and the
monomer set has only 20 molecules per class.

The output has exactly the same columns as data/processed/molecules.csv, so processed.load_*
reads both through one code path.

    python tools/make_teaching_subset.py
      -> data/teaching/molecules.csv   (metadata + the 791 raw stick intensities)
"""
import os

import numpy as np
import pandas as pd

from irspectra import paths
from irspectra.data import processed
from irspectra.data import spectra

# arbitrary (the build date) but frozen: data/teaching/ is committed, so a new
# seed rewrites a binary blob in git and silently changes every figure in the teaching notebook
SEED = 20260805
# molecules to keep per metal, by peptide length. None = keep every molecule of that length.
PER_METAL = {1: None,     # all 20/metal -- the monomer set is small enough to ship whole
             2: 20,
             3: 40}       # trimers are the Step 5-6 transfer target, so keep the most of them


def draw_subset(meta, seed=SEED):
    """Pick a stratified subset of molecule row indices, balanced over (length, metal).

    Args:
        meta (pandas.DataFrame): The processed molecule table from processed.load_processed().
        seed (int): Seed for the per-group sample. Defaults to SEED.

    Returns:
        numpy.ndarray: Sorted row indices into ``meta`` / ``X`` / ``sticks``.
    """
    rng = np.random.RandomState(seed)
    keep = []
    for (length, metal), grp in meta.groupby(["length", "metal"], sort=True):
        n = PER_METAL.get(length)
        idx = grp.index.to_numpy()
        keep.append(idx if n is None or n >= len(idx) else rng.choice(idx, size=n, replace=False))
    return np.sort(np.concatenate(keep))


def main():
    """Build data/teaching/ from data/processed/ and report what was written.

    Writes:
        data/teaching/molecules.csv: Metadata then the 791 raw stick intensities, exactly the
            layout of data/processed/molecules.csv so the same loader reads both.

    Raises:
        FileNotFoundError: If data/processed/molecules.csv is absent; build it with
            ``python tools/build_dataset.py``.
    """
    meta, _, sticks = processed.load_processed()
    idx = draw_subset(meta)

    sub = meta.loc[idx].reset_index(drop=True)
    spec = pd.DataFrame(sticks[idx], columns=spectra.SPEC_COLS).reset_index(drop=True)
    out = pd.concat([sub, spec], axis=1)

    os.makedirs(paths.TEACHING_DIR, exist_ok=True)
    csv_path = os.path.join(paths.TEACHING_DIR, "molecules.csv")
    out.to_csv(csv_path, index=False, float_format="%.6g")

    print(f"  drew {len(sub)} of {len(meta)} molecules (seed {SEED})")
    print("  class balance (rows = length, cols = metal):")
    tbl = sub.groupby(["length", "metal"]).size().unstack(fill_value=0)
    print("    " + tbl.to_string().replace("\n", "\n    "))
    print(f"  wrote {csv_path}  {out.shape[0]} rows x {out.shape[1]} cols "
          f"({os.path.getsize(csv_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
