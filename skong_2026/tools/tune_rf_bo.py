#!/usr/bin/env python3
"""
tune_rf_bo.py -- Bayesian-Optimization hyperparameter tuning for the Step 4
per-conformer metal RandomForest, using Optuna's GPSampler (a Gaussian-Process
surrogate + acquisition function -- classic Bayesian optimization).

Non-negotiables baked in:
  * PER-CONFORMER data, mono+di only (never trimers -- those are the Step 5 target).
  * Objective = mean MACRO-F1 over a 5-fold StratifiedGroupKFold GROUPED BY MOLECULE,
    so conformers never leak between the tuning-train and tuning-validation folds.
  * n_estimators is FIXED (small during the search, larger for the final model) --
    more trees never overfit a forest, so tuning it only wastes compute.

Tune once -> writes output/step2/best_params.json, which irspectra.config loads into its
module-level RF_PARAMS -> every step4/5/6 script picks it up automatically.

    python tools/tune_rf_bo.py --n-trials 40 --subsample-molecules 800
    python tools/tune_rf_bo.py --subsample-molecules 0        # tune on all 1,676 molecules
"""
import os
import json
import argparse
import warnings
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from irspectra import paths
from irspectra.modeling import metrics
from irspectra.data import conformers

OUT = paths.output_dir("step2")
SEARCH_TREES = 150       # forest size during the search (kept modest for speed)
FINAL_TREES = 300        # forest size written for the final model
N_SPLITS = 5


def load_monodi(n_molecules, seed=0):
    """Per-conformer mono+di; optionally keep a random subset of MOLECULES (grouped,
    so whole molecules are kept and the CV grouping stays valid).

    Trimers are never loaded -- they are the Step-5 zero-shot target.

    Args:
        n_molecules (int): Keep a random subset of this many molecules. Falsy
            (0 or None), or a value at least the total molecule count, keeps
            every molecule.
        seed (int): Seed for the molecule subsample. Defaults to 0.

    Returns:
        tuple: ``(X, y, groups)`` -- the broadened spectra (numpy.ndarray, shape
        (n_conformers, 791)), the metal labels (numpy.ndarray of str) and the
        per-conformer molecule group ids (numpy.ndarray of str).

    Raises:
        ValueError: Propagated from conformers.load_conformers() if the mono/dimer
            conformer CSVs are missing from data/.
    """
    meta, X = conformers.load_conformers(lengths=(1, 2))
    y = meta["metal"].to_numpy()
    groups = meta["molecule"].to_numpy()
    if n_molecules and n_molecules < meta["molecule"].nunique():
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(meta["molecule"].unique(), size=n_molecules, replace=False))
        mask = np.array([g in keep for g in groups])
        X, y, groups = X[mask], y[mask], groups[mask]
    return X, y, groups


def objective(trial, X, y, groups):
    """Optuna objective: mean macro-F1 of one RandomForest hyperparameter draw.

    Suggests max_features, min_samples_leaf, min_samples_split, max_depth and
    class_weight, then scores them with a 5-fold StratifiedGroupKFold GROUPED BY
    MOLECULE so conformers never leak between the tuning-train and
    tuning-validation folds. n_estimators is fixed at SEARCH_TREES -- more trees
    never overfit a forest, so tuning it only wastes compute.

    Args:
        trial (optuna.trial.Trial): The trial supplying the suggested values.
        X (numpy.ndarray): Per-conformer spectra, shape (n_conformers, 791).
        y (numpy.ndarray): Metal labels, one per conformer.
        groups (numpy.ndarray): Molecule group ids, one per conformer.

    Returns:
        float: Mean macro-F1 across the 5 grouped folds; Optuna maximizes this.
    """
    max_features      = trial.suggest_float("max_features", 0.02, 0.4)
    min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 50, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 40, log=True)
    max_depth         = trial.suggest_int("max_depth", 8, 60, log=True)
    class_weight      = trial.suggest_categorical("class_weight", ["none", "balanced", "balanced_subsample"])
    cw = None if class_weight == "none" else class_weight

    # folds + RF pinned to 0, not --seed: all trials see identical folds, so score gaps are
    # hyperparameters and not resampling noise
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    scores = []
    for tr, va in cv.split(X, y, groups):                    # grouped by molecule -> no conformer leak
        rf = RandomForestClassifier(n_estimators=SEARCH_TREES, n_jobs=-1, random_state=0,
                                    max_features=max_features, min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split, max_depth=max_depth,
                                    class_weight=cw)
        rf.fit(X[tr], y[tr])
        scores.append(metrics.macro_f1(y[va], rf.predict(X[va])))
    return float(np.mean(scores))


def main():
    """Tune the per-conformer metal RandomForest by Gaussian-Process Bayesian optimization.

    Parses ``--n-trials``, ``--subsample-molecules``, ``--seed`` and ``--out``
    from the command line, loads the per-conformer mono+di data, and runs an
    Optuna GPSampler study maximizing grouped 5-fold macro-F1. The winning
    params are written with n_estimators raised to FINAL_TREES, and are picked up
    automatically by the step scripts through irspectra.config.RF_PARAMS.

    Writes:
        output/step2/best_params.json (or ``--out``): Best params plus the search
            score, sampler, trial count and a description of the tuning data.

    Raises:
        SystemExit: Propagated from argparse on an invalid command line.
        ValueError: Propagated from conformers.load_conformers() if the mono/dimer
            conformer CSVs are missing from data/.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=40)
    # 800 of 1,676: the full set is hours x 40 trials, and the
    # param ranking is stable at half the molecules -> search on 800, refit the winner on everything
    ap.add_argument("--subsample-molecules", type=int, default=800,
                    help="tune on a random grouped subset of this many molecules (0 = all 1,676)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUT, "best_params.json"))
    a = ap.parse_args()

    X, y, groups = load_monodi(a.subsample_molecules, seed=a.seed)
    print(f"tuning data: {len(y)} conformers, {len(np.unique(groups))} molecules, "
          f"classes {sorted(np.unique(y))}", flush=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    sampler = optuna.samplers.GPSampler(seed=a.seed, n_startup_trials=10)   # GP-based Bayesian optimization
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def log_cb(st, tr):
        """Optuna callback printing each finished trial's score and the best so far.

        Args:
            st (optuna.study.Study): The running study.
            tr (optuna.trial.FrozenTrial): The trial that just completed.
        """
        print(f"  trial {tr.number:>2d}: macro-F1 {tr.value:.4f}   (best {st.best_value:.4f})", flush=True)

    study.optimize(lambda t: objective(t, X, y, groups), n_trials=a.n_trials,
                   show_progress_bar=False, callbacks=[log_cb])

    best = dict(study.best_params)
    if best.get("class_weight") == "none":
        best["class_weight"] = None
    result = {
        "best_params": {**best, "n_estimators": FINAL_TREES},
        "search_macro_f1": float(study.best_value),
        "sampler": "optuna.GPSampler (Gaussian-Process Bayesian optimization)",
        "n_trials": a.n_trials, "search_trees": SEARCH_TREES, "final_trees": FINAL_TREES,
        "objective": "mean macro-F1, 5-fold StratifiedGroupKFold grouped by molecule",
        "tuned_on": f"per-conformer mono+di, {len(y)} conformers / {len(np.unique(groups))} molecules",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nbest search macro-F1 = {study.best_value:.4f}")
    print("best params:", result["best_params"])
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
