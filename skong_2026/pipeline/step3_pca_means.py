#!/usr/bin/env python3
r"""
step3_pca_means.py -- Step 3 PCA scatter with the molecule-MEANS overlaid on the conformer
clouds. Re-plots from the coords saved by pipeline/step3_pca_coords.py
(output/step3/pca_coords.csv + evr.json) so it does NOT recompute the PCA -- run that script
first. Faint conformer clouds + bold per-molecule means, metal-color legend only.

Molecule mean = per-molecule average of the PC coords, which for a linear PCA equals projecting the mean
spectrum. -> figures/step3_pca_metals_means.svg
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams["savefig.transparent"] = True

from irspectra import paths
from irspectra.viz.palette import METAL_COLORS
from irspectra.viz.palette import METAL_LABELS
from irspectra.config import METAL_ORDER
OUT = paths.output_dir("step3")
FIG = paths.FIGURES_DIR


def main():
    """Re-plot the Step-3 PCA scatter with per-molecule means over the conformer clouds.

    Reads the saved PCA coordinates rather than recomputing the decomposition,
    averages the PC coords per molecule (equivalent to projecting the mean
    spectrum, since PCA is linear), and draws a subsampled faint conformer cloud
    under bold molecule means.

    Reads:
        output/step3/pca_coords.csv, output/step3/evr.json: Written by the Step-3
            PCA script.

    Writes:
        figures/step3_pca_metals_means.svg: The publication figure.
        figures/_v_pca_means.png: A white-background preview.

    Raises:
        FileNotFoundError: If the Step-3 PCA outputs have not been generated yet.
    """
    d = pd.read_csv(os.path.join(OUT, "pca_coords.csv"))
    evr = json.load(open(os.path.join(OUT, "evr.json")))["evr_percent"]
    means = d.groupby("molecule").agg(PC1=("PC1", "mean"), PC2=("PC2", "mean"),
                                      metal=("metal", "first")).reset_index()
    # cosmetic only -> same cloud every rerun (unrelated to config.SEEDS)
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(10, 8))
    for m in METAL_ORDER:                                   # faint conformer clouds (subsampled)
        idx = np.where(d.metal.values == m)[0]
        # cap the cloud at 1200/metal: at alpha 0.09 more points just saturate and the SVG balloons
        sub = rng.choice(idx, size=min(1200, len(idx)), replace=False)
        ax.scatter(d.PC1.values[sub], d.PC2.values[sub], s=6, color=METAL_COLORS[m], alpha=0.09,
                   edgecolors="none", zorder=2)
    for m in METAL_ORDER:                                   # molecule means, bold on top
        sel = means.metal == m
        ax.scatter(means.loc[sel, "PC1"], means.loc[sel, "PC2"], s=20, color=METAL_COLORS[m], alpha=0.9,
                   edgecolors="#2b2b2b", linewidths=0.35, zorder=3)
    ax.set_xlabel(f"PC1 ({evr[0]:.1f}% var)", fontsize=18, labelpad=15)
    ax.set_ylabel(f"PC2 ({evr[1]:.1f}% var)", fontsize=18, labelpad=15)
    ax.set_title("PCA of Mono + Dimer Metal Complexes\n"
                 "molecule means (solid) over conformer clouds (faint)", fontsize=19, pad=18)
    for sp in ax.spines.values():
        sp.set_linewidth(2.5)
    ax.tick_params(axis="both", which="major", length=8, width=2, labelsize=12)
    h = [Line2D([0], [0], marker="o", linestyle="", markersize=12, markeredgecolor="none",
                markerfacecolor=METAL_COLORS[m], label=METAL_LABELS[m]) for m in METAL_ORDER]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4, frameon=False,
              fontsize=15, columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout()
    out = os.path.join(FIG, "step3_pca_metals_means.svg")
    fig.savefig(out, dpi=600, transparent=True, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "_v_pca_means.png"), facecolor="white", dpi=100, bbox_inches="tight")
    print(f"wrote {out} | {len(d)} conformers, {len(means)} molecule-means")


if __name__ == "__main__":
    main()
