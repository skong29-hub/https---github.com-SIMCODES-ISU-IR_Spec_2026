"""Metric wrappers that pin the label set.

sklearn's ``f1_score(..., average="macro")`` averages over whatever labels happen to appear in
``y_true | y_pred``. On a small or skewed test set that silently averages over 3 classes instead
of 4, and the number stops being comparable to the other points on the same curve. These
wrappers always pass ``labels=METAL_ORDER``, so a missing class scores 0 rather than vanishing.
"""
from sklearn.metrics import f1_score

from irspectra.config import METAL_ORDER


def macro_f1(y_true, y_pred, labels=METAL_ORDER):
    """Macro-averaged F1 over a fixed label set.

    Args:
        y_true (array-like): True metal labels.
        y_pred (array-like): Predicted metal labels.
        labels (list): Classes to average over. Defaults to METAL_ORDER.

    Returns:
        float: The macro-F1. A class absent from both arrays contributes 0, not nothing.
    """
    return float(f1_score(y_true, y_pred, labels=labels, average="macro"))


def per_metal_f1(y_true, y_pred, labels=METAL_ORDER):
    """Per-class F1, one entry per label, in ``labels`` order.

    Args:
        y_true (array-like): True metal labels.
        y_pred (array-like): Predicted metal labels.
        labels (list): Classes to score. Defaults to METAL_ORDER.

    Returns:
        numpy.ndarray: F1 per class, aligned with ``labels``.
    """
    return f1_score(y_true, y_pred, labels=labels, average=None)
