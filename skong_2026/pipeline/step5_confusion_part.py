#!/usr/bin/env python3
"""
step5_confusion_part.py -- per-metal confusion matrix for the mono+di model evaluated
ZERO-SHOT on the matched-size unseen-trimer test set (Step 5's "trimer_part").
"part" = a trimer subsample the same size as the mono+di holdout; see
step5_confusion_whole.py for the same matrix over ALL trimer conformers.

Per-seed protocol is step5_zeroshot.py's, verbatim -- see that module docstring. The only
delta is the output: score only the matched-size trimer subsample, row-normalize the
confusion per fold, average over folds then seeds. Each cell = mean recall fraction.

    python pipeline/step5_confusion_part.py --cap 10   # -> output/step5/confusion_trimer_part_cap10.csv
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

from irspectra import paths
from irspectra.modeling import protocol
from irspectra.config import METAL_ORDER, SEEDS, TEST_FRACTION, N_FOLDS, RF_PARAMS, RF_PARAMS_SOURCE

OUT = paths.output_dir("step5")


def main():
    """Build the zero-shot confusion matrix on the matched-size unseen-trimer set.

    Per-seed protocol is step5_zeroshot.py's, verbatim -- see that module
    docstring. Parses ``--aggregate`` and ``--cap``; the only delta is the
    output: row-normalize the confusion per fold, average over folds then seeds.

    Writes (with a ``_meanavg`` / ``_cap{K}`` suffix reflecting the flags):
        output/step5/confusion_trimer_part*.csv: Mean row-normalized confusion.
        output/step5/confusion_trimer_part*_summary.json: Per-class recall mean
            and std, plus the run configuration.

    Raises:
        SystemExit: Propagated from argparse on an invalid command line.
        ValueError: Propagated from protocol.load_monodi_and_trimers() if the conformer
            CSVs are missing from data/processed/.
    """
    ap = argparse.ArgumentParser()
    protocol.add_aggregate_flag(ap)
    protocol.add_cap_flag(ap)
    a = ap.parse_args()
    suffix = protocol.suffix_for(a)
    rep = protocol.representation(a)

    print(f"loading mono+di + trimers ({rep}) ...", flush=True)
    meta_md, X_md, meta_tr, X_tr = protocol.load_monodi_and_trimers(a)
    y_md = meta_md["metal"].to_numpy(); g_md = meta_md["molecule"].to_numpy()
    y_tr = meta_tr["metal"].to_numpy()
    unit = "conf" if a.aggregate == "none" else "mol"
    print(f"mono+di {len(y_md)} {unit} | trimers {len(y_tr)} {unit}", flush=True)
    print(f"RF params [{RF_PARAMS_SOURCE}]: {RF_PARAMS}", flush=True)

    per_seed = []
    for seed in SEEDS:
        tr_idx, te_idx = next(iter(GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                                                     random_state=seed).split(X_md, y_md, g_md)))
        Xpool, ypool, gpool = X_md[tr_idx], y_md[tr_idx], g_md[tr_idx]
        n_test = len(te_idx)
        # fixed matched-size trimer subsample for this seed (same #conformers as the mono+di holdout)
        sub = np.random.RandomState(seed).choice(len(y_tr), size=min(n_test, len(y_tr)), replace=False)
        Xsub, ysub = X_tr[sub], y_tr[sub]

        fold_cms = []
        skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        # val fold discarded on purpose -- 5 folds give 5 training subsets -> 5 models, each scored on
        # the external trimer subsample below (fold spread = model variance, not CV error)
        for fit_idx, _ in skf.split(Xpool, ypool, gpool):
            rf = RandomForestClassifier(random_state=seed, n_jobs=-1, **RF_PARAMS)
            rf.fit(Xpool[fit_idx], ypool[fit_idx])
            cm = confusion_matrix(ysub, rf.predict(Xsub), labels=METAL_ORDER).astype(float)
            fold_cms.append(cm / cm.sum(axis=1, keepdims=True))     # row-normalize this fold
        per_seed.append(np.mean(fold_cms, axis=0))                  # per-seed = mean of 5 folds
        print(f"  seed {seed}: recall "
              f"{ {m: round(per_seed[-1][i, i], 3) for i, m in enumerate(METAL_ORDER)} }", flush=True)

    arr = np.array(per_seed)                                        # (10, 4, 4)
    cmn_mean, cmn_std = arr.mean(axis=0), arr.std(axis=0, ddof=1)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(cmn_mean, index=METAL_ORDER, columns=METAL_ORDER).to_csv(
        os.path.join(OUT, f"confusion_trimer_part{suffix}.csv"))
    recall = {m: float(cmn_mean[i, i]) for i, m in enumerate(METAL_ORDER)}
    with open(os.path.join(OUT, f"confusion_trimer_part{suffix}_summary.json"), "w") as f:
        json.dump({"test_set": "trimer_part (matched-size unseen trimers)",
                   "model": f"mono+di {rep} RF (tuned), 5-fold x 10-seed", "aggregate": a.aggregate,
                   "metals": METAL_ORDER, "per_class_recall_mean": recall,
                   "per_class_recall_std": {m: float(cmn_std[i, i]) for i, m in enumerate(METAL_ORDER)},
                   "n_seeds": len(per_seed), "confusion_mean_rownorm": cmn_mean.tolist()}, f, indent=2)
    print("\nper-class recall (mean over 10 seeds):", {m: round(r, 3) for m, r in recall.items()})
    print(f"wrote {OUT}/confusion_trimer_part{suffix}.csv (+ summary)")


if __name__ == "__main__":
    main()
