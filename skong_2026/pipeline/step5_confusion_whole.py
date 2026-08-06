#!/usr/bin/env python3
"""
step5_confusion_whole.py -- per-metal confusion for the mono+di model evaluated ZERO-SHOT on the WHOLE
unseen-trimer set (all trimer conformers), the companion to step5_confusion_part.py's matched-size "part".

Per-seed protocol is step5_zeroshot.py's, verbatim -- see that module docstring. Deltas vs
step5_confusion_part.py: the test set is EVERY trimer conformer (not a size-matched subsample), the
output is the row-normalized confusion averaged over folds then seeds, there is no --aggregate flag
(per-conformer only), and this script also renders its own heatmap svg.

    python pipeline/step5_confusion_whole.py --cap 10   # -> output/step5/confusion_trimer_whole_cap10.csv + svg
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["savefig.transparent"] = True
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, accuracy_score

from irspectra import paths
from irspectra.modeling import protocol
from irspectra.config import METAL_ORDER, SEEDS, TEST_FRACTION, N_FOLDS, RF_PARAMS, RF_PARAMS_SOURCE
from irspectra.viz.panels import plot_confusion

OUT = paths.output_dir("step5")
FIG = paths.FIGURES_DIR


def main():
    """Build the zero-shot confusion matrix on the WHOLE unseen-trimer set.

    Per-seed protocol is step5_zeroshot.py's, verbatim -- see that module
    docstring. Parses ``--cap``; the deltas are the test set (every trimer
    conformer) and the output: row-normalized confusion averaged over folds then
    seeds, plus a heatmap svg.

    Writes (with a ``_cap{K}`` suffix when capping):
        output/step5/confusion_trimer_whole*.csv: Mean row-normalized confusion.
        output/step5/confusion_trimer_whole*_summary.json: Per-metal recall and
            mean zero-shot accuracy.
        figures/step5_confusion_trimer_whole*.svg: The heatmap figure.

    Raises:
        SystemExit: Propagated from argparse on an invalid command line.
        ValueError: Propagated from protocol.load_monodi_and_trimers() if the conformer
            CSVs are missing from data/processed/.
    """
    ap = argparse.ArgumentParser()
    protocol.add_cap_flag(ap)
    a = ap.parse_args()
    sfx = protocol.suffix_for(a)
    meta_md, X_md, meta_tr, X_tr = protocol.load_monodi_and_trimers(a)
    y_md = meta_md["metal"].to_numpy(); g_md = meta_md["molecule"].to_numpy()
    y_tr = meta_tr["metal"].to_numpy()
    print(f"mono+di {len(y_md)} conf | trimers {len(y_tr)} conf (WHOLE test)", flush=True)
    print(f"RF params [{RF_PARAMS_SOURCE}]: {RF_PARAMS}", flush=True)

    per_seed, accs = [], []
    for seed in SEEDS:
        # holdout discarded: split only so the train pool matches step5_zeroshot.py's exactly
        tr_idx, _ = next(iter(GroupShuffleSplit(1, test_size=TEST_FRACTION,
                                                random_state=seed).split(X_md, y_md, g_md)))
        Xpool, ypool, gpool = X_md[tr_idx], y_md[tr_idx], g_md[tr_idx]
        fold_cms, fold_acc = [], []
        skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        # val fold discarded on purpose -- 5 folds give 5 training subsets -> 5 models, each scored on
        # the external whole-trimer set below (fold spread = model variance, not CV error)
        for fit_idx, _ in skf.split(Xpool, ypool, gpool):
            rf = RandomForestClassifier(random_state=seed, n_jobs=-1, **RF_PARAMS).fit(Xpool[fit_idx], ypool[fit_idx])
            pred = rf.predict(X_tr)                                # predict ALL trimers
            cm = confusion_matrix(y_tr, pred, labels=METAL_ORDER).astype(float)
            fold_cms.append(cm / cm.sum(axis=1, keepdims=True))
            fold_acc.append(accuracy_score(y_tr, pred))
        per_seed.append(np.mean(fold_cms, axis=0)); accs.append(float(np.mean(fold_acc)))
        print(f"  seed {seed}: zero-shot acc {accs[-1]:.3f}", flush=True)

    arr = np.array(per_seed); cm_mean = arr.mean(axis=0)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(cm_mean, index=METAL_ORDER, columns=METAL_ORDER).to_csv(os.path.join(OUT, f"confusion_trimer_whole{sfx}.csv"))
    recall = {m: float(cm_mean[i, i]) for i, m in enumerate(METAL_ORDER)}
    json.dump({"test_set": "trimer_whole (all unseen trimer conformers)", "representation": "per-conformer",
               "model": "mono+di RF (tuned), 5-fold x 10-seed", "metals": METAL_ORDER,
               "per_metal_recall_mean": recall, "accuracy_mean": float(np.mean(accs)),
               "n_seeds": len(accs), "confusion_mean_rownorm": cm_mean.tolist()},
              open(os.path.join(OUT, f"confusion_trimer_whole{sfx}_summary.json"), "w"), indent=2)

    fig, _ = plot_confusion(
        cm_mean,
        "Mono+di model, zero-shot on all (whole) trimers\n"
        "(per-conformer, per-metal confusion, mean of 10 seeds)",
        cmap="Oranges", figsize=(7.0, 6.2), title_fontsize=16)
    out = os.path.join(FIG, f"step5_confusion_trimer_whole{sfx}.svg")
    fig.savefig(out, dpi=600, transparent=True, bbox_inches="tight")
    print(f"\nrecall: {{ {', '.join(f'{m}:{recall[m]:.2f}' for m in METAL_ORDER)} }} | zero-shot acc {np.mean(accs):.3f}")
    print(f"wrote {out} (+ csv + summary)")


if __name__ == "__main__":
    main()
