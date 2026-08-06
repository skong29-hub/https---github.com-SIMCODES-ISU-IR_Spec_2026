"""Smoke tests for the committed model-result tables (the numbers the poster reports).

These read only from output/, so they pass without the raw data present.
"""

import numpy as np
from irspectra import results

POSTER_OUTPUTS = {
    "step2": ["best_params.json"],
    "step3": ["pca_coords.csv", "evr.json", "pca_loadings.csv"],
    "step4": ["cv_results_macrof1.csv", "confusion_perconformer_cap10.csv"],
    "step5": ["zeroshot_results_cap10.csv",
              "confusion_trimer_part_cap10.csv", "confusion_trimer_whole_cap10.csv"],
    "step6": ["frontier_fixed_resampledup_perseedtest_cap10.csv",
              "length_substitution_perconf_cap10_summary.csv"],
}


def test_all_poster_outputs_present():
    """Every result file the poster reports is committed under output/."""
    for step, files in POSTER_OUTPUTS.items():
        for name in files:
            assert results.exists(step, name), f"missing output/{step}/{name}"


def test_step4_macro_f1_in_range():
    """The Step-4 headline macro-F1 still sits in the poster's reported band."""
    s = results.load_json("step4", "summary_macrof1.json")
    assert 0.78 <= s["mean_macro_f1"] <= 0.84   # poster headline: 0.81


def test_zeroshot_shows_length_drop():
    """Zero-shot trimer transfer scores below the in-domain mono+di holdout."""
    s = results.load_json("step5", "zeroshot_summary_cap10.json")
    assert s["monodi"]["mean"] > s["trimer_whole"]["mean"]
    assert 0.58 <= s["trimer_whole"]["mean"] <= 0.68         # poster: ~0.63


def test_confusion_rows_are_recall():
    """The Step-4 confusion matrix is row-normalized over the four metals in order."""
    cm = results.load_confusion("step4", "confusion_perconformer_cap10.csv")
    assert list(cm.index) == ["Co+2", "Ni+2", "Cu+2", "Zn+2"]
    assert np.allclose(cm.to_numpy().sum(axis=1), 1.0, atol=1e-6)


def test_length_substitution_ceiling():
    """The per-conformer trimer ceiling still sits in the poster's reported band."""
    meta = results.load_json("step6", "length_substitution_perconf_cap10_meta.json")
    assert 0.68 <= meta["trimer_ceiling_macro_f1_perconf"] <= 0.76   # ~0.72
