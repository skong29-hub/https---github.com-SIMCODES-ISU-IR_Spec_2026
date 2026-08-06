#!/usr/bin/env python3
"""
step3_pca_loadings.py -- Step 3 PCA "loadings as spectra": which wavenumbers drive PC1 and PC2.
Same data as step3 (mono+di metal complexes, per-conformer, full source). Fits PCA, plots the PC1 and
PC2 loading vectors (components_) against wavenumber with the standard IR bands shaded.

    python pipeline/step3_pca_loadings.py   # -> figures/step3_pca_loadings.svg (+ output/step3/pca_loadings.csv)
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["savefig.transparent"] = True
from sklearn.decomposition import PCA

from irspectra import paths
from irspectra.data import conformers
from irspectra.data import spectra
WN = spectra.WN
OUT = paths.output_dir("step3")
FIG = paths.FIGURES_DIR
COLORS = ["#15807D", "#C1651B"]     # PC1 teal, PC2 orange -> match figure_notebook's step2 cell


def main():
    """Fit a 3-component PCA on the mono+dimer conformers and plot PC1/PC2 loadings.

    Loads the per-conformer full-source mono+dimer set, fits the PCA, then writes
    the two leading loading vectors as both a table and a two-panel figure with
    the standard IR bands shaded and labelled.

    Writes:
        output/step3/pca_loadings.csv: PC1 and PC2 loadings indexed by wavenumber.
        figures/step3_pca_loadings.svg: The two-panel loadings figure.

    Raises:
        ValueError: Propagated from conformers.load_conformers() if the mono/dimer
            conformer CSVs are missing from data/.
    """
    meta, X = conformers.load_conformers(lengths=(1, 2))          # per-conformer, full source (matches step3)
    print(f"PCA on {len(X)} conformers x {X.shape[1]} features", flush=True)
    pca = PCA(n_components=3).fit(X)
    evr = pca.explained_variance_ratio_ * 100
    comp = pca.components_                                      # (3, 791) loading vectors

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(comp[:2].T, columns=["PC1_loading", "PC2_loading"], index=pd.Index(WN, name="wavenumber")
                 ).to_csv(os.path.join(OUT, "pca_loadings.csv"))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for k, ax in enumerate(axes):
        for lo, hi, _ in spectra.BANDS:
            ax.axvspan(lo, hi, color="0.85", alpha=0.55, lw=0, zorder=0)
        ax.axhline(0, color="0.6", lw=1, zorder=1)
        ax.plot(WN, comp[k], color=COLORS[k], lw=1.8, zorder=3)
        ax.set_ylabel(f"PC{k + 1} loading", fontsize=15, labelpad=8)
        ax.set_title(f"PC{k + 1}  ({evr[k]:.1f}% of variance)", fontsize=15, loc="left", fontweight="bold", pad=8)
        for sp in ax.spines.values():
            sp.set_linewidth(1.8)
        ax.tick_params(axis="both", which="major", length=6, width=1.5, labelsize=12)
    y0, y1 = axes[0].get_ylim()
    for lo, hi, label in spectra.BANDS:
        axes[0].text((lo + hi) / 2, y1, label, rotation=90, va="top", ha="center", fontsize=8, color="0.35")
    axes[0].set_xlim(WN.min(), WN.max())
    axes[1].set_xlabel(r"wavenumber (cm$^{-1}$)", fontsize=15, labelpad=8)
    fig.suptitle("Feature Loadings as Spectra — Which Bands Drive Each PC", fontsize=17, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIG, "step3_pca_loadings.svg")
    fig.savefig(out, dpi=600, transparent=True, bbox_inches="tight")
    print(f"wrote {out} | PC1 {evr[0]:.1f}%  PC2 {evr[1]:.1f}%  (+ output/step3/pca_loadings.csv)")


if __name__ == "__main__":
    main()
