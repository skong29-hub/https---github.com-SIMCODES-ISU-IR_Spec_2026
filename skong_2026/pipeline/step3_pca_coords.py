#!/usr/bin/env python3
"""
step3_pca_coords.py -- Step 3 PCA fit: project every mono+dimer conformer onto the first
10 principal components and save the coordinates.

This is the FIRST half of Step 3 and the only place the PCA is actually fitted. The
downstream consumers re-plot from what this writes and never refit:
  pipeline/step3_pca_means.py     reads pca_coords.csv + evr.json -> the molecule-means scatter
  notebooks/figure_notebook.ipynb  reads the same two files for its Step 3 panel
(pipeline/step3_pca_loadings.py is independent -- it refits a 3-component PCA for the
loading vectors, which need components_ rather than the projected coordinates.)

Same data as every other Step 3 view: per-conformer, full source, metal complexes only,
monomers + dimers, broadened and area-normalized by conformers.load_conformers().

    python pipeline/step3_pca_coords.py
      -> output/step3/pca_coords.csv  (molecule, metal, length, PC1..PC10)
      -> output/step3/evr.json        (explained-variance %, n_conformers)
"""
import os
import json
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from irspectra import paths
from irspectra.data import conformers

OUT = paths.output_dir("step3")
N_COMPONENTS = 10  # only PC1/PC2 get plotted; keep 10 so evr.json can report cumulative variance
LENGTHS = (1, 2)          # mono + dimer; trimers stay unseen (they are the Step 5 target)


def main():
    """Fit a 10-component PCA on the mono+dimer conformers and save the coordinates.

    Writes:
        output/step3/pca_coords.csv: One row per conformer -- the ``molecule`` group id,
            its metal and length, then PC1..PC10.
        output/step3/evr.json: ``evr_percent`` (explained variance of each component, in
            percent) and ``n_conformers``.

    Raises:
        ValueError: Propagated from conformers.load_conformers() if the mono/dimer conformer
            CSVs are missing from data/.
    """
    meta, X = conformers.load_conformers(lengths=LENGTHS)
    print(f"PCA on {len(X)} conformers x {X.shape[1]} features", flush=True)

    pca = PCA(n_components=N_COMPONENTS).fit(X)
    coords = pca.transform(X)
    evr = pca.explained_variance_ratio_ * 100

    os.makedirs(OUT, exist_ok=True)
    df = meta[["molecule", "metal", "length"]].copy()
    pcs = pd.DataFrame(coords, columns=[f"PC{k + 1}" for k in range(N_COMPONENTS)], index=df.index)
    pd.concat([df, pcs], axis=1).to_csv(os.path.join(OUT, "pca_coords.csv"), index=False)
    with open(os.path.join(OUT, "evr.json"), "w") as f:
        json.dump({"evr_percent": [float(v) for v in evr], "n_conformers": int(len(X))}, f, indent=2)

    total = float(np.sum(evr))
    print(f"  PC1 {evr[0]:.1f}%  PC2 {evr[1]:.1f}%  | first {N_COMPONENTS} PCs = {total:.1f}% of variance")
    print(f"wrote {OUT}/pca_coords.csv (+ evr.json)")


if __name__ == "__main__":
    main()
