#!/usr/bin/env python3
"""
step5_zeroshot.py -- Step 5 zero-shot / length transfer. Per-conformer metal RF
trained on mono+di, evaluated on three test sets, 10-seed Monte-Carlo CV.

Per seed (10 HARDCODED seeds -> reproducible):
  1. lock out 20% of mono+di MOLECULES (grouped -> no conformer leakage); 80% = train pool.
  2. 5-fold StratifiedGroupKFold on the 80% -> 5 RF fold-models.
  3. score each fold-model (macro-F1) on:
       monodi        -- the frozen 20% mono+di holdout
       trimer_whole  -- ALL trimer conformers
       trimer_part   -- a random trimer-conformer subsample the SAME NUMBER of conformers
                        as the mono+di holdout (fixed per seed)
     average the 5 folds -> that seed's macro-F1 per test set.
Each bar = mean over the 10 seeds. Error bars come from the spread across seeds
(bars monodi & trimer_part resample their test set each seed; trimer_whole is the
fixed whole set). Uses the Bayesian-Optimization-tuned RF params
(output/step2/best_params.json) via RF_PARAMS.

    python pipeline/step5_zeroshot.py --cap 10   # -> output/step5/zeroshot_results_cap10.csv
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from irspectra import paths
from irspectra.modeling import protocol, metrics
from irspectra.config import METAL_ORDER, SEEDS, TEST_FRACTION, N_FOLDS, RF_PARAMS, RF_PARAMS_SOURCE, ci95

OUT = paths.output_dir("step5")
COLS = ["monodi", "trimer_whole", "trimer_part"]


def main():
    """Run the Step-5 zero-shot / length-transfer evaluation over 10 seeds.

    Per-seed protocol is the module docstring's, verbatim -- see above. Parses
    ``--aggregate`` and ``--cap`` from the command line; per-metal F1 is logged
    alongside macro-F1.

    Writes (with a ``_meanavg`` / ``_cap{K}`` suffix reflecting the flags):
        output/step5/zeroshot_results*.csv: Per-seed macro and per-metal F1.
        output/step5/zeroshot_summary*.json: Mean and 95% CI per test set.

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
    unit = "conformers" if a.aggregate == "none" else "molecules"
    print(f"mono+di: {len(y_md)} {unit} | trimers: {len(y_tr)} {unit}", flush=True)
    print(f"RF params [{RF_PARAMS_SOURCE}]: {RF_PARAMS}", flush=True)

    rows = []
    for seed in SEEDS:
        tr_idx, te_idx = next(iter(GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                                                     random_state=seed).split(X_md, y_md, g_md)))
        Xpool, ypool, gpool = X_md[tr_idx], y_md[tr_idx], g_md[tr_idx]
        Xmd_te, ymd_te = X_md[te_idx], y_md[te_idx]
        n_test = len(te_idx)
        # size-matched control -> rules out 'trimer_whole is lower only because it is bigger'
        rng = np.random.RandomState(seed)
        sub = rng.choice(len(y_tr), size=min(n_test, len(y_tr)), replace=False)

        macro = {c: [] for c in COLS}
        permetal = {c: [] for c in COLS}
        skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        # val fold discarded on purpose -- 5 folds give 5 training subsets -> 5 models, each scored on
        # the three external test sets below (fold spread = model variance, not CV error)
        for fit_idx, _ in skf.split(Xpool, ypool, gpool):
            rf = RandomForestClassifier(random_state=seed, n_jobs=-1, **RF_PARAMS)
            rf.fit(Xpool[fit_idx], ypool[fit_idx])
            pred_tr = rf.predict(X_tr)                       # all trimers (reused for whole + part)
            evals = {"monodi": (ymd_te, rf.predict(Xmd_te)),
                     "trimer_whole": (y_tr, pred_tr),
                     "trimer_part": (y_tr[sub], pred_tr[sub])}
            for c, (yt, yp) in evals.items():
                macro[c].append(metrics.macro_f1(yt, yp))
                permetal[c].append(metrics.per_metal_f1(yt, yp))
        row = {"seed": seed, "n_test_conformers": int(n_test)}
        for c in COLS:
            row[c] = float(np.mean(macro[c]))                            # macro-F1
            for m, v in zip(METAL_ORDER, np.mean(np.array(permetal[c]), axis=0)):
                row[f"{c}__{m}"] = float(v)                              # per-metal F1
        rows.append(row)
        print(f"  seed {seed}: monodi {row['monodi']:.3f}  "
              f"trimer_whole {row['trimer_whole']:.3f}  trimer_part {row['trimer_part']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"zeroshot_results{suffix}.csv"), index=False)

    def mci(c):
        """Mean and 95% CI half-width across seeds for one test set's scores.

        Args:
            c (str): Column name in the per-seed results frame, one of COLS.

        Returns:
            tuple: ``(mean, ci95)`` as floats, the CI from the seed-to-seed
            standard error.
        """
        v = df[c].to_numpy()
        return float(v.mean()), ci95(v)
    summary = {c: dict(zip(["mean", "ci95"], mci(c))) for c in COLS}
    summary.update({"n_seeds": int(len(df)), "metric": "macro_f1", "aggregate": a.aggregate,
                    "mean_test_conformers": float(df["n_test_conformers"].mean()),
                    "rf_params": RF_PARAMS})
    with open(os.path.join(OUT, f"zeroshot_summary{suffix}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nmacro-F1 (mean ± 95% CI over 10 seeds):")
    for c in COLS:
        print(f"  {c:14s} {summary[c]['mean']:.3f} ± {summary[c]['ci95']:.3f}")
    print(f"wrote {OUT}/zeroshot_results{suffix}.csv (+ summary)")


if __name__ == "__main__":
    main()
