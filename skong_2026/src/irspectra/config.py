"""Shared modeling constants for the metal-classification pipeline.

The modelling protocol and the RandomForest hyperparameters are defined once, here.
Not every script needs every constant -- the step3 PCA scripts import nothing from this
module, step4_confusion_perconformer takes no TEST_FRACTION, and the step6 scripts take
no N_FOLDS -- so add a constant here only when a second caller wants it.

The seeds are all hardcoded on purpose: every result in output/ is meant to be
byte-reproducible from a clean checkout.
"""
import os
import json
import math

from irspectra.paths import OUTPUT_DIR

# The four predicted classes, in the order every confusion matrix and per-metal F1 array uses.
# Lives here rather than in viz/ because it is a modelling fact -- metrics must not import viz.
METAL_ORDER = ["Co+2", "Ni+2", "Cu+2", "Zn+2"]

# 10 fixed seeds -> reproducible Monte-Carlo cross-validation
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# NOT a frozen split: callers pass random_state=seed, so the outer test is a fresh 20% of
# MOLECULES per seed. Only step6 pins its test set, and it does that via TEST_SEEDS below.
TEST_FRACTION = 0.20
N_FOLDS = 5              # grouped folds on the remaining 80%

# Hardcoded seeds for the held-out TRIMER test split (Step 6). Both step6 scripts read these,
# and they must stay identical or the two experiments stop being comparable.
# [0] = 20250722 is the ONE fixed holdout (step6_frontier uses TEST_SEEDS[0] for --test-mode
# fixed); the 2025070x block is consumed positionally, one per SEEDS entry. Do not reorder.
TEST_SEEDS = [20250722, 20250701, 20250702, 20250703, 20250704,
              20250705, 20250706, 20250707, 20250708, 20250709]

# Fallback only -- differs from the tuned params on 3 of 6 values, so a run that lands here
# is materially different from the committed results. RF_PARAMS_SOURCE records which won.
DEFAULT_RF_PARAMS = dict(
    n_estimators=300, max_depth=21, min_samples_leaf=1,
    min_samples_split=3, max_features=0.0752081675448156,
    class_weight="balanced_subsample",
)

BEST_PARAMS_PATH = os.path.join(OUTPUT_DIR, "step2", "best_params.json")


def _resolve_rf_params():
    """Read the tuned RandomForest params, falling back to the untuned defaults.

    Returns:
        tuple: ``(params, source)`` -- the RandomForestClassifier kwargs (dict) and a short
        human-readable string naming where they came from, for scripts to print.
    """
    try:
        with open(BEST_PARAMS_PATH) as f:
            return dict(json.load(f)["best_params"]), "tuned (output/step2/best_params.json)"
    except FileNotFoundError:
        return dict(DEFAULT_RF_PARAMS), "DEFAULT_RF_PARAMS (best_params.json not found)"
    except (ValueError, KeyError, OSError) as exc:
        return dict(DEFAULT_RF_PARAMS), f"DEFAULT_RF_PARAMS (best_params.json unreadable: {exc})"


def load_rf_params():
    """Re-read the RandomForest hyperparameters from disk.

    Scripts do NOT call this -- they import the module-level RF_PARAMS, resolved once at
    import. Use this only to pick up a best_params.json written after import (e.g. in a
    notebook that just ran the tuner).

    Returns:
        dict: Keyword arguments for ``sklearn.ensemble.RandomForestClassifier``
        (n_estimators, max_depth, min_samples_leaf, min_samples_split,
        max_features, class_weight). A fresh dict each call, safe to mutate.
    """
    return _resolve_rf_params()[0]


def ci95(values):
    """Half-width of the 95% confidence interval on the mean of ``values``.

    The +/- term the step scripts report alongside every mean: 1.96 standard errors,
    using the sample standard deviation (ddof=1).

    Args:
        values (array-like): The per-seed scores. Needs at least 2 entries.

    Returns:
        float: The CI half-width, or 0.0 when fewer than 2 values are given.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    # z=1.96, not the t_{.975,9}=2.262 a strict n=10 CI would use -> ~15% narrower bars. Every
    # +/- in output/ was computed this way; changing it re-labels every published number.
    return 1.96 * sd / math.sqrt(n)


RF_PARAMS, RF_PARAMS_SOURCE = _resolve_rf_params()
