"""The metal palette -- colours and math-rendered labels, keyed by the `metal` column.

THE single source of truth for how a metal is drawn. Import these rather than re-declaring
them, or panels drift apart.

METAL_ORDER (the four predicted classes) lives in irspectra.config, not here: it is a
modelling fact, and putting it here would make metrics code import the viz layer.
ALL_ORDER adds the apo ("none") peptides for data views where apo rows are real data
rather than a class the model predicts.
"""
from irspectra.config import METAL_ORDER

# Okabe-Ito, colorblind-safe. Keys match the `metal` column ("Co+2", ...).
METAL_COLORS = {"Co+2": "#0072B2", "Ni+2": "#009E73", "Cu+2": "#E69F00", "Zn+2": "#CC79A7"}
METAL_LABELS = {"Co+2": "Co$^{2+}$", "Ni+2": "Ni$^{2+}$", "Cu+2": "Cu$^{2+}$", "Zn+2": "Zn$^{2+}$"}

# apo (metal-free) peptides -- a baseline in the data views, never a predicted class
APO_LABEL = "none"
ALL_COLORS = {APO_LABEL: "#898781", **METAL_COLORS}
ALL_LABELS = {APO_LABEL: "apo", **METAL_LABELS}
ALL_ORDER = [APO_LABEL] + METAL_ORDER


def pretty_metal(m):
    """"Cu+2" -> "Cu$^{2+}$" for math-rendered labels.

    Args:
        m (str): Metal label from the ``metal`` column, e.g. ``"Cu+2"``. Strings
            without a ``"+2"`` suffix (e.g. ``"none"``) pass through unchanged.

    Returns:
        str: The label with ``"+2"`` rewritten as a mathtext superscript.
    """
    return m.replace("+2", "$^{2+}$")
