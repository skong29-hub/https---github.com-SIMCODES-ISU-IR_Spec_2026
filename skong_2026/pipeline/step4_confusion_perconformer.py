#!/usr/bin/env python3
r"""
step4_confusion_perconformer.py -- Step 4 per-metal confusion for the per-conformer model, with
every molecule capped to K=10 conformers so the per-metal counts are balanced.

Out-of-fold StratifiedGroupKFold(5) grouped by molecule, 10 hardcoded seeds, row-normalized per
seed then averaged, so each cell is a mean recall fraction.

    python pipeline/step4_confusion_perconformer.py
      -> output/step4/confusion_perconformer_cap10.csv
      -> figures/step4_confusion_perconformer_cap10.svg
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["savefig.transparent"] = True
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, accuracy_score

from irspectra import paths
from irspectra.data import conformers
from irspectra.config import METAL_ORDER, SEEDS, N_FOLDS, RF_PARAMS, RF_PARAMS_SOURCE
from irspectra.viz.panels import plot_confusion

OUT = paths.output_dir("step4")
FIG = paths.FIGURES_DIR
# must stay byte-identical to figure_notebook.ipynb cell 12's S4C title -> both draw the same panel
TITLE = "Per-conformer model — per-metal confusion\n(out-of-fold, mean of 10 seeds)"
K = 10  # Co/Ni have ~20 conformers/molecule, Cu/Zn ~10 -> cap 10 balances, drops no Cu/Zn


def main():
    """Build the per-conformer per-metal confusion matrix on the K=10-capped set.

    Caps every molecule to K conformers so per-metal counts are balanced, then
    for each of the 10 hardcoded seeds collects out-of-fold predictions from a
    StratifiedGroupKFold(5) grouped by molecule, row-normalizes the confusion
    matrix, and averages the seeds. Each cell is a mean recall fraction.

    Writes:
        output/step4/confusion_perconformer_cap{K}.csv: The mean row-normalized
            confusion matrix (``confusion_perconformer_cap10.csv`` at the default K=10).
        figures/step4_confusion_perconformer_cap{K}.svg: The heatmap figure.

    Raises:
        ValueError: Propagated from conformers.load_conformers() if the mono/dimer
            conformer CSVs are missing from data/.
    """
    # cap_seed fixed at 0 while SEEDS varies -> every seed sees the SAME capped rows, so the
    # spread below is fold/model variance only, not conformer-resampling noise
    meta, Xc = conformers.load_conformers(lengths=(1, 2), cap=K, cap_seed=0)
    yc = meta["metal"].to_numpy(); gc = meta["molecule"].to_numpy()
    print(f"K={K} cap -> {len(yc)} conformers | per-metal {pd.Series(yc).value_counts().to_dict()}", flush=True)
    print(f"RF params [{RF_PARAMS_SOURCE}]: {RF_PARAMS}", flush=True)

    per_seed, accs = [], []
    for seed in SEEDS:
        skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.empty(len(yc), dtype=object)
        for tr, te in skf.split(Xc, yc, gc):
            rf = RandomForestClassifier(random_state=seed, n_jobs=-1, **RF_PARAMS).fit(Xc[tr], yc[tr])
            oof[te] = rf.predict(Xc[te])
        cm = confusion_matrix(yc, oof, labels=METAL_ORDER).astype(float)
        per_seed.append(cm / cm.sum(axis=1, keepdims=True))
        accs.append(accuracy_score(yc, oof))
        print(f"  seed {seed}: out-of-fold acc {accs[-1]:.3f}", flush=True)

    cm_mean = np.array(per_seed).mean(axis=0)
    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, f"confusion_perconformer_cap{K}.csv")   # filename follows K, not hardcoded
    pd.DataFrame(cm_mean, index=METAL_ORDER, columns=METAL_ORDER).to_csv(csv_path)

    fig, _ = plot_confusion(cm_mean, TITLE, cmap="Blues")
    out = os.path.join(FIG, f"step4_confusion_perconformer_cap{K}.svg")
    fig.savefig(out, dpi=600, transparent=True, bbox_inches="tight")
    print(f"\nout-of-fold acc = {np.mean(accs):.4f} | recall {{ {', '.join(f'{m}:{cm_mean[i,i]:.2f}' for i,m in enumerate(METAL_ORDER))} }}")
    print(f"wrote {out} (+ {os.path.basename(csv_path)})")


if __name__ == "__main__":
    main()
