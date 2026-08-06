#!/usr/bin/env python3
r"""
step6_frontier.py -- Step 6 data-value frontier. Per-conformer metal RF; grow the
TRAINING set with trimer molecules and watch macro-F1 on a FIXED trimer TEST set climb.

Design (confirmed with the user):
  * Fixed trimer TEST set: 20% of trimer MOLECULES held out (grouped, fixed seed) -- never trained on.
    This one test set scores every point on the curve.
  * Trimer TRAIN pool: the other 80% of trimer molecules.
  * Training = ALL mono+di + a growing slice of the trimer pool.
  * x-axis = trimer molecules added, in +5%-of-pool steps: 0, 5, ..., 100%.
  * Two arms:
      unique    -- add that % of DISTINCT trimer molecules (new information).
      duplicate -- a FIXED base (the first 5% of pool molecules) whose conformer rows are REPLICATED
                   to match the unique arm's added-conformer count at each step (same info, more volume).
  * 10 HARDCODED seeds (reproducible); each point = mean +/- 95% CI over the seeds.
  * --tune none      : reuse the Step-2 tuned params (best_params.json) at every point   [Figure A]
    --tune per_point : re-tune the RF (Optuna GPSampler Bayesian optimization) ONCE per (point, arm),
                       reused across the 10 seeds                                          [Figure B]

Per-seed macro-F1 AND per-metal F1 are saved for every (pct, arm, seed) row so downstream
graphs can be rebuilt without re-running.

  --dup-mode   tile (default, Fig A/B: one fixed base replicated) | resample (Fig C: a NEW
               per-seed base sampled WITH replacement -> widening error bars)
  --test-mode  fixed (default, Fig A/B: one trimer test for all seeds) | per_seed (Fig C: a
               DIFFERENT 20% grouped trimer test each seed, from hardcoded TEST_SEEDS)

    python pipeline/step6_frontier.py --tune none                         # Figure A
    python pipeline/step6_frontier.py --tune per_point                    # Figure B
    python pipeline/step6_frontier.py --dup-mode resample --test-mode per_seed   # Figure C
"""
import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from irspectra import paths
from irspectra.modeling import protocol, metrics
from irspectra.config import METAL_ORDER, SEEDS, TEST_SEEDS, TEST_FRACTION, RF_PARAMS

OUT = paths.output_dir("step6")
TEST_SEED = TEST_SEEDS[0]                       # fixed trimer test holdout (test-mode fixed; Fig A/B)
# arbitrary but frozen, drawn OUTSIDE the seed loop -> tile arm varies only by the RF
BASE_SEED = 7                                   # fixed duplicate base selection (tile mode)
# offset by seed; kept far from SEEDS 0-9 so a resample base never repeats an RF draw
DUP_BASE_SEED = 4242                            # base seed for the per-seed resample duplicate (Fig C)
# search-only forest; winner refits at RF_PARAMS' 300. more trees never overfit a
# forest, so tuning n_estimators just burns compute
SEARCH_TREES = 150


def bo_tune(X, y, groups, n_trials, subsample_molecules, seed=0):
    """Small GP Bayesian-Optimization tune (grouped 5-fold macro-F1) for one augmented
    training set.

    Tunes on a grouped molecule subsample for speed, so whole molecules are kept
    and the CV grouping stays valid.

    Args:
        X (numpy.ndarray): Training spectra, shape (n_samples, 791).
        y (numpy.ndarray): Metal labels, one per sample.
        groups (numpy.ndarray): Molecule group ids, one per sample.
        n_trials (int): Number of Optuna trials to run.
        subsample_molecules (int): Tune on a random grouped subset of this many
            molecules. Falsy, or at least the molecule count, uses them all.
        seed (int): Seed for both the subsample and the GPSampler. Defaults to 0.

    Returns:
        tuple: ``(best_params, best_value)`` -- the winning RandomForest kwargs
        with n_estimators fixed to the Step-2 final count, and the best mean
        macro-F1 reached during the search.
    """
    # subsample molecules (grouped) to keep the search cheap
    if subsample_molecules and len(np.unique(groups)) > subsample_molecules:
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(np.unique(groups), size=subsample_molecules, replace=False))
        m = np.array([g in keep for g in groups])
        X, y, groups = X[m], y[m], groups[m]

    def objective(trial):
        """Optuna objective: mean grouped 5-fold macro-F1 for one params draw.

        Args:
            trial (optuna.trial.Trial): The trial supplying the suggested values.

        Returns:
            float: Mean macro-F1 across the 5 molecule-grouped folds.
        """
        p = dict(max_features=trial.suggest_float("max_features", 0.02, 0.4),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True),
                 min_samples_split=trial.suggest_int("min_samples_split", 2, 40, log=True),
                 max_depth=trial.suggest_int("max_depth", 8, 60, log=True),
                 class_weight=trial.suggest_categorical("class_weight", ["none", "balanced", "balanced_subsample"]))
        cw = None if p["class_weight"] == "none" else p["class_weight"]
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        sc = []
        for tr, va in cv.split(X, y, groups):
            rf = RandomForestClassifier(n_estimators=SEARCH_TREES, n_jobs=-1, random_state=0,
                                        max_features=p["max_features"], min_samples_leaf=p["min_samples_leaf"],
                                        min_samples_split=p["min_samples_split"], max_depth=p["max_depth"],
                                        class_weight=cw).fit(X[tr], y[tr])
            sc.append(metrics.macro_f1(y[va], rf.predict(X[va])))
        return float(np.mean(sc))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.GPSampler(seed=seed, n_startup_trials=6))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params)
    if best.get("class_weight") == "none":
        best["class_weight"] = None
    best["n_estimators"] = RF_PARAMS.get("n_estimators", 300)
    return best, float(study.best_value)


def make_split(X_tr, y_tr, g_tr, test_seed, dup_base_frac):
    """One 20% grouped trimer test split + its 80% pool, plus a tile-mode fixed base for the pool.

    The base molecules are drawn with the module-level BASE_SEED rather than
    ``test_seed``, so the tile-mode duplicate arm uses the same base across
    seeds.

    Args:
        X_tr (numpy.ndarray): Trimer spectra, shape (n_conformers, 791).
        y_tr (numpy.ndarray): Trimer metal labels, one per conformer.
        g_tr (numpy.ndarray): Trimer molecule group ids, one per conformer.
        test_seed (int): Seed for the 20% grouped GroupShuffleSplit holdout.
        dup_base_frac (float): Duplicate base size as a fraction of the pool's
            molecules; always rounds up to at least 1 molecule.

    Returns:
        dict: The split, with keys ``Xtest`` / ``ytest`` (the held-out trimer
        test set), ``Xp`` / ``yp`` / ``gp`` (the 80% pool), ``pool_mols`` (unique
        pool molecules), ``m2i`` (molecule -> conformer indices into ``Xp``),
        ``nb`` (base molecule count) and ``base_conf`` (the tile-mode base's
        conformer indices).
    """
    p_i, t_i = next(iter(GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                                           random_state=test_seed).split(X_tr, y_tr, g_tr)))
    Xp, yp, gp = X_tr[p_i], y_tr[p_i], g_tr[p_i]
    pmols = np.unique(gp)
    m2i = {m: np.where(gp == m)[0] for m in pmols}                  # conformer indices into Xp
    nb = max(1, round(dup_base_frac * len(pmols)))
    base_mols = np.random.RandomState(BASE_SEED).choice(pmols, size=nb, replace=False)
    base_conf = np.concatenate([m2i[m] for m in base_mols])
    return dict(Xtest=X_tr[t_i], ytest=y_tr[t_i], Xp=Xp, yp=yp, gp=gp,
                pool_mols=pmols, m2i=m2i, nb=nb, base_conf=base_conf)


def main():
    """Trace the Step-6 data-value frontier: macro-F1 vs. trimer molecules added.

    Parses the run configuration from the command line (``--tune``,
    ``--step-pct``, ``--n-seeds``, ``--bo-trials``,
    ``--bo-subsample-molecules``, ``--dup-mode``, ``--dup-base-frac``,
    ``--arms``, ``--test-mode``, ``--aggregate``, ``--cap``), then grows the
    training set from all mono+di data across a 0-100% grid of the trimer pool
    and scores each point on a held-out trimer test set. The ``unique`` arm adds
    distinct molecules (new information); the ``duplicate`` arm replicates a
    fixed base to the same conformer count (same information, more volume).

    Writes (``{tag}`` encodes the flags -- tuned/fixed plus resampledup,
    perseedtest, meanavg and cap suffixes):
        output/step6/frontier_{tag}.csv: One row per (pct, arm, seed) with macro
            and per-metal F1.
        output/step6/frontier_{tag}_config.json: The resolved run configuration.
        output/step6/frontier_{tag}_params.csv: Per-point tuned params, written
            only under ``--tune per_point``.

    Raises:
        SystemExit: From argparse on an invalid command line, or when
            ``--test-mode per_seed`` is given more seeds than TEST_SEEDS has.
        ValueError: Propagated from protocol.load_monodi_and_trimers() if the conformer
            CSVs are missing from data/processed/.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", choices=["none", "per_point"], default="none")
    ap.add_argument("--step-pct", type=int, default=5, help="grid increment in %% of the trimer pool")
    ap.add_argument("--n-seeds", type=int, default=len(SEEDS), help="use SEEDS[:n] (10 = full)")
    ap.add_argument("--bo-trials", type=int, default=15)
    ap.add_argument("--bo-subsample-molecules", type=int, default=400)
    ap.add_argument("--dup-mode", choices=["tile", "resample"], default="tile",
                    help="duplicate arm: 'tile' = ONE fixed base replicated in order (Fig A/B); "
                         "'resample' = a NEW per-seed base sampled WITH replacement (Fig C)")
    ap.add_argument("--dup-base-frac", type=float, default=0.05,
                    help="duplicate base size as a fraction of the trimer pool molecules")
    ap.add_argument("--arms", default="unique,duplicate",
                    help="comma list of arms to run")
    ap.add_argument("--test-mode", choices=["fixed", "per_seed"], default="fixed",
                    help="fixed = ONE trimer test set for all seeds (Fig A/B); per_seed = a DIFFERENT "
                         "20%% grouped trimer test each seed (hardcoded TEST_SEEDS; Fig C)")
    protocol.add_aggregate_flag(ap)
    protocol.add_cap_flag(ap)
    a = ap.parse_args()
    seeds = SEEDS[:a.n_seeds]
    run_arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    # this script's tag carries two extra segments the shared suffix knows nothing about, so the
    # arm/test-mode parts are built here and the shared _meanavg/_cap tail is appended
    tag = "tuned" if a.tune == "per_point" else "fixed"
    if a.dup_mode == "resample":
        tag += "_resampledup"                       # never overwrite the tile-mode Fig A/B results
    if a.test_mode == "per_seed":
        tag += "_perseedtest"
    tag += protocol.suffix_for(a)

    rep = protocol.representation(a)
    print(f"loading mono+di + trimers ({rep}) ...", flush=True)
    meta_md, X_md, meta_tr, X_tr = protocol.load_monodi_and_trimers(a)
    y_md = meta_md["metal"].to_numpy(); g_md = meta_md["molecule"].to_numpy()
    y_tr = meta_tr["metal"].to_numpy(); g_tr = meta_tr["molecule"].to_numpy()

    # trimer test seed per model-seed: fixed reuses TEST_SEED; per_seed uses distinct hardcoded seeds
    if a.test_mode == "per_seed":
        if len(seeds) > len(TEST_SEEDS):
            raise SystemExit(f"--test-mode per_seed needs <= {len(TEST_SEEDS)} seeds")
        seed_test_seed = {s: TEST_SEEDS[i] for i, s in enumerate(seeds)}
    else:
        seed_test_seed = {s: TEST_SEED for s in seeds}
    splits = {s: make_split(X_tr, y_tr, g_tr, seed_test_seed[s], a.dup_base_frac) for s in seeds}
    ref = splits[seeds[0]]                          # reference split for grid sizing + tuning
    print(f"mono+di {len(y_md)} conf | trimer pool ~{len(ref['yp'])} conf / {len(ref['pool_mols'])} mol "
          f"| trimer test ~{len(ref['ytest'])} conf | test-mode {a.test_mode}", flush=True)

    def dup_add(sp, target, base_seed):
        """Duplicate-arm rows for split `sp` at the given target conformer count.

        tile     -> sp's fixed BASE_SEED base, replicated in order (Fig A/B; base constant across
                    seeds, so the arm's only seed variance is the RF -> tight error bars).
        resample -> a NEW base of sp['nb'] molecules drawn for THIS seed, conformer rows sampled
                    WITH replacement to `target` (Fig C; per-seed base + amplification -> variance
                    that grows with target -> widening error bars). Hardcoded -> reproducible.

        Args:
            sp (dict): A split from make_split().
            target (int): Number of conformer rows to produce; 0 adds nothing.
            base_seed (int): Offset added to DUP_BASE_SEED for the per-seed
                resample base. Ignored in tile mode.

        Returns:
            numpy.ndarray: ``target`` conformer indices into ``sp['Xp']``, with
            repeats.
        """
        if target == 0:
            return np.array([], int)
        if a.dup_mode == "resample":
            drng = np.random.RandomState(DUP_BASE_SEED + base_seed)
            bm = drng.choice(sp["pool_mols"], size=sp["nb"], replace=False)
            bc = np.concatenate([sp["m2i"][m] for m in bm])
            return drng.choice(bc, size=target, replace=True)
        return np.tile(sp["base_conf"], int(np.ceil(target / len(sp["base_conf"]))))[:target]

    grid = list(range(0, 101, a.step_pct))
    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, f"frontier_{tag}.csv")
    json.dump({"tag": tag, "tune": a.tune, "dup_mode": a.dup_mode, "test_mode": a.test_mode,
               "dup_base_frac": a.dup_base_frac, "arms": run_arms, "seeds": seeds,
               "test_seeds": [int(seed_test_seed[s]) for s in seeds], "grid": grid,
               "BASE_SEED": BASE_SEED, "DUP_BASE_SEED": DUP_BASE_SEED, "TEST_FRACTION": TEST_FRACTION},
              open(os.path.join(OUT, f"frontier_{tag}_config.json"), "w"), indent=2)
    print(f"grid {grid}  arms {run_arms}  seeds {seeds}  tune={a.tune}  dup={a.dup_mode}  "
          f"test={a.test_mode}  base~{ref['nb']} mol  -> {out_csv}", flush=True)

    rows, tune_records = [], []
    for pct in grid:
        n_mols_ref = round(pct / 100 * len(ref["pool_mols"]))
        # tuned mode: one BO tune per (pct, arm) on the reference split's seed-0 augmented set
        params_arm = {}
        for arm in run_arms:
            if a.tune == "none":
                params_arm[arm] = dict(RF_PARAMS)
            else:
                r0 = np.random.RandomState(0)
                mols0 = r0.choice(ref["pool_mols"], size=n_mols_ref, replace=False) if n_mols_ref else np.array([], dtype=ref["pool_mols"].dtype)
                uconf0 = np.concatenate([ref["m2i"][m] for m in mols0]) if len(mols0) else np.array([], int)
                add0 = uconf0 if arm == "unique" else dup_add(ref, len(uconf0), 0)
                if len(add0) == 0:
                    # pct=0 -> nothing added, so reuse the tuned mono+di params
                    # rather than re-tuning on nothing
                    params_arm[arm], sc = dict(RF_PARAMS), None
                else:
                    Xa = np.vstack([X_md, ref["Xp"][add0]]); ya = np.concatenate([y_md, ref["yp"][add0]])
                    ga = np.concatenate([g_md, ref["gp"][add0]])
                    params_arm[arm], sc = bo_tune(Xa, ya, ga, a.bo_trials, a.bo_subsample_molecules)
                tune_records.append({"pct": pct, "arm": arm, "search_macro_f1": sc, **params_arm[arm]})
                pd.DataFrame(tune_records).to_csv(os.path.join(OUT, f"frontier_{tag}_params.csv"), index=False)
                print(f"  [tune] pct {pct:3d} {arm:9s} -> {params_arm[arm]}", flush=True)

        for seed in seeds:
            sp = splits[seed]
            n_mols = round(pct / 100 * len(sp["pool_mols"]))
            rng = np.random.RandomState(seed)
            mols = rng.choice(sp["pool_mols"], size=n_mols, replace=False) if n_mols else np.array([], dtype=sp["pool_mols"].dtype)
            uconf = np.concatenate([sp["m2i"][m] for m in mols]) if len(mols) else np.array([], int)
            target = len(uconf)
            for arm in run_arms:
                add = uconf if arm == "unique" else dup_add(sp, target, seed)
                if len(add):
                    Xtr = np.vstack([X_md, sp["Xp"][add]]); ytr = np.concatenate([y_md, sp["yp"][add]])
                else:
                    Xtr, ytr = X_md, y_md
                rf = RandomForestClassifier(random_state=seed, n_jobs=-1, **params_arm[arm]).fit(Xtr, ytr)
                pred = rf.predict(sp["Xtest"])
                macro = metrics.macro_f1(sp["ytest"], pred)
                permetal = metrics.per_metal_f1(sp["ytest"], pred)
                row = {"pct": pct, "n_molecules_added": int(n_mols), "n_conformers_added": int(len(add)),
                       "arm": arm, "seed": int(seed), "test_seed": int(seed_test_seed[seed]),
                       "macro_f1": float(macro)}
                row.update({f"f1_{m}": float(v) for m, v in zip(METAL_ORDER, permetal)})
                rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)                # incremental write (crash-safe)
        cur = pd.DataFrame(rows); cur = cur[cur.pct == pct].groupby("arm")["macro_f1"].mean().round(3).to_dict()
        print(f"  pct {pct:3d}%: {cur}", flush=True)

    print(f"\nwrote {out_csv} (+ config)")


if __name__ == "__main__":
    main()
