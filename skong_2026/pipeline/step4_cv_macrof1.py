#!/usr/bin/env python3
r"""
step4_cv_macrof1.py -- per-conformer metal-classification CV scored in MACRO-F1. 10 hardcoded seeds,
freeze 20% of MOLECULES (grouped), StratifiedGroupKFold(5) on the 80%, each fold-model scored on the
frozen 20% test; per-seed = mean of the 5 folds. Single arm (per-conformer, full source) so it matches
the poster's macro-F1 numbers.

    python pipeline/step4_cv_macrof1.py
      -> output/step4/cv_results_macrof1.csv (+ summary) and figures/step4_cv_macrof1.svg
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["savefig.transparent"] = True
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from irspectra import paths
from irspectra.modeling import metrics
from irspectra.data import conformers
from irspectra.config import SEEDS, TEST_FRACTION, N_FOLDS, RF_PARAMS, RF_PARAMS_SOURCE, ci95

OUT = paths.output_dir("step4")
FIG = paths.FIGURES_DIR


def main():
    """Score the per-conformer metal classifier in macro-F1 over 10-seed Monte-Carlo CV.

    Per seed: freeze 20% of MOLECULES as a grouped outer test set, run
    StratifiedGroupKFold(5) on the remaining 80%, score each fold-model on the
    frozen test set, and take the mean of the 5 folds as that seed's macro-F1.

    Writes:
        output/step4/cv_results_macrof1.csv: Per-seed and per-fold macro-F1.
        output/step4/summary_macrof1.json: Mean, 95% CI, std and per-seed scores.
        figures/step4_cv_macrof1.svg: Per-seed scores with fold-range error bars.
        figures/_v_cvf1.png: A white-background preview.

    Raises:
        ValueError: Propagated from conformers.load_conformers() if the mono/dimer
            conformer CSVs are missing from data/.
    """
    meta, X = conformers.load_conformers(lengths=(1, 2))
    y = meta["metal"].to_numpy(); g = meta["molecule"].to_numpy()
    print(f"per-conformer: {len(y)} samples / {meta['molecule'].nunique()} molecules", flush=True)
    print(f"RF params [{RF_PARAMS_SOURCE}]: {RF_PARAMS}", flush=True)

    rows = []
    for seed in SEEDS:
        tr, te = next(iter(GroupShuffleSplit(1, test_size=TEST_FRACTION,
                                             random_state=seed).split(X, y, g)))
        Xtr, ytr, gtr = X[tr], y[tr], g[tr]; Xte, yte = X[te], y[te]
        fold = []
        skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        # val fold discarded on purpose -- the 5 folds only give 5 training subsets -> 5 models,
        # each scored on the frozen test set below (fold spread = model variance, not CV error)
        for fit_idx, _ in skf.split(Xtr, ytr, gtr):
            rf = RandomForestClassifier(random_state=seed, n_jobs=-1, **RF_PARAMS).fit(Xtr[fit_idx], ytr[fit_idx])
            fold.append(metrics.macro_f1(yte, rf.predict(Xte)))
        row = {"seed": seed, "macro_f1": float(np.mean(fold)), "test_molecules": int(len(np.unique(g[te])))}
        row.update({f"fold{k}": float(v) for k, v in enumerate(fold)})
        rows.append(row)
        print(f"  seed {seed}: macro-F1 {row['macro_f1']:.3f}  folds {[round(v,3) for v in fold]}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "cv_results_macrof1.csv"), index=False)
    m = df["macro_f1"].to_numpy()
    ci = ci95(m)
    json.dump({"metric": "macro_f1", "representation": "per-conformer", "mean_macro_f1": float(m.mean()),
               "ci95": float(ci), "std": float(m.std(ddof=1)), "n_seeds": int(len(m)),
               "per_seed": {int(s): float(v) for s, v in zip(df["seed"], m)}, "rf_params": RF_PARAMS},
              open(os.path.join(OUT, "summary_macrof1.json"), "w"), indent=2)

    foldcols = [f"fold{k}" for k in range(N_FOLDS)]
    lo = df["macro_f1"] - df[foldcols].min(axis=1); hi = df[foldcols].max(axis=1) - df["macro_f1"]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.errorbar(df["seed"], df["macro_f1"], yerr=[lo, hi], fmt="o", ms=11, color="#2A78D6",
                capsize=5, elinewidth=1.5, capthick=1.5, zorder=3, label="per-seed macro-F1 (bars = fold range)")
    ax.axhline(m.mean(), ls="-", color="#2A78D6", lw=2, zorder=2)
    ax.fill_between([-0.5, 9.5], m.mean() - ci, m.mean() + ci, color="#2A78D6", alpha=0.13, zorder=1,
                    label=f"mean {m.mean():.3f} ± {ci:.3f} (95% CI)")
    # chance = 1/4: only true because the 4 metals are ~balanced
    ax.axhline(0.25, ls=":", color="0.5", lw=1.8, zorder=2, label="chance = 0.25")
    ax.set_xticks(range(10)); ax.set_xlim(-0.5, 9.5); ax.set_ylim(0, 1)
    ax.set_xlabel("seed", fontsize=18, labelpad=12)
    ax.set_ylabel("macro-F1 on frozen 20% test", fontsize=18, labelpad=12)
    ax.set_title("Metal classification — 10-seed Monte-Carlo CV\n(monomer + dimer, per-conformer)", fontsize=22, pad=16)
    for sp in ax.spines.values(): sp.set_linewidth(2.5)
    ax.tick_params(axis="both", which="major", length=8, width=2, labelsize=13)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "step4_cv_macrof1.svg"), dpi=600, transparent=True, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "_v_cvf1.png"), facecolor="white", dpi=100, bbox_inches="tight")
    print(f"\nmean macro-F1 = {m.mean():.4f} ± {ci:.4f} over {len(m)} seeds")
    print("wrote figures/step4_cv_macrof1.svg (+ cv_results_macrof1.csv, summary)")


if __name__ == "__main__":
    main()
