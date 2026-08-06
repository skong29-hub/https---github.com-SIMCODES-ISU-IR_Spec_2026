"""The shared figure panels: the confusion-matrix heatmap and the IR-band annotation.

One implementation each, called by both the step scripts and the figure notebook, so the
same result never renders two different ways.
"""
import numpy as np

from irspectra.config import METAL_ORDER
from irspectra.data.spectra import BANDS
from irspectra.viz.palette import METAL_COLORS, METAL_LABELS

def plot_confusion(cm, title, cmap="Blues", metals=None, ax=None, figsize=(7.0, 6.0),
                   title_fontsize=20, label_fontsize=17, tick_labelsize=16, cell_fontsize=16):
    """Draw a row-normalized per-metal confusion matrix, metal-colored tick labels and all.

    THE one confusion-matrix renderer -- the step scripts and the figure notebook both call
    this, so the panels cannot drift apart. Callers do their own loading: scripts pass the
    array they just computed, the notebook passes results.load_confusion(...).

    Args:
        cm (array-like or pandas.DataFrame): Square matrix, already row-normalized so the
            diagonal is per-metal recall. A DataFrame is reindexed to ``metals`` first, so its
            row/column order does not have to match.
        title (str): Axes title, drawn over the matrix.
        cmap (str): Matplotlib colormap name. Defaults to ``"Blues"``.
        metals (list, optional): Class order for both axes. Defaults to METAL_ORDER.
        ax (matplotlib.axes.Axes, optional): Draw into these axes instead of a new figure.
        figsize (tuple): Figure size when ``ax`` is None. Defaults to (7.0, 6.0).
        title_fontsize (int): Title size. Defaults to 20.
        label_fontsize (int): "predicted" / "true" axis-label size. Defaults to 17.
        tick_labelsize (int): Metal tick-label size. Defaults to 16.
        cell_fontsize (int): Size of the number printed in each cell. Defaults to 16.

    Returns:
        tuple: ``(fig, ax)``.

    Raises:
        KeyError: If ``cm`` is a DataFrame missing one of ``metals`` -- most often from passing
            ALL_ORDER (which includes the apo "none") where METAL_ORDER is meant.
    """
    import matplotlib.pyplot as plt          # deferred: importing plotting must not pick a backend
                                             # (irspectra/__init__ imports this module eagerly)

    metals = list(METAL_ORDER if metals is None else metals)
    if hasattr(cm, "loc"):                                  # DataFrame -> honour the requested order
        cm = cm.loc[metals, metals].to_numpy(float)
    cm = np.asarray(cm, dtype=float)

    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=figsize)
    # scale pinned 0-1, not to the data range -> the same colour means the same recall in
    # every panel, so the Blues (step4) and Oranges (step5) matrices stay comparable
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    ticks = range(len(metals))
    labels = [METAL_LABELS.get(m, m) for m in metals]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=tick_labelsize)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=tick_labelsize)
    for tick, m in zip(list(ax.get_xticklabels()) + list(ax.get_yticklabels()), metals * 2):
        tick.set_color(METAL_COLORS.get(m, "black")); tick.set_fontweight("bold")
    ax.set_xlabel("predicted", fontsize=label_fontsize, labelpad=12)
    ax.set_ylabel("true", fontsize=label_fontsize, labelpad=12)
    ax.set_title(title, fontsize=title_fontsize, pad=14)
    for i in range(len(metals)):
        for j in range(len(metals)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=cell_fontsize,
                    color="white" if cm[i, j] > 0.5 else "black")
    fig.tight_layout()
    return fig, ax


def annotate_bands(ax, bands=BANDS, fontsize=10, alpha=0.10, color="0.45",
                   text_color="#555", levels=(0.98, 0.87, 0.76, 0.65)):
    """Shade the standard IR band regions and label them with the bond/group they
    report on, staggered vertically so the rotated labels do not overlap.

    Bands falling entirely outside the axes' current x-limits are skipped, and
    those partly outside are clipped to them, so the axes must already be scaled
    when this is called.

    Args:
        ax (matplotlib.axes.Axes): Axes to annotate in place.
        bands (list): ``(lo, hi, label)`` tuples in cm^-1. Defaults to
            ``data.spectra.BANDS``.
        fontsize (int): Font size of the rotated band labels. Defaults to 10.
        alpha (float): Opacity of the shaded band spans. Defaults to 0.10.
        color (str): Fill color of the shaded spans. Defaults to ``"0.45"``.
        text_color (str): Color of the band labels. Defaults to ``"#555"``.
        levels (tuple): Label heights as fractions of the y-range, cycled so
            neighbouring labels sit at different heights. Defaults to
            ``(0.98, 0.87, 0.76, 0.65)``.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    li = 0
    for lo, hi, label in bands:
        if hi < x0 or lo > x1:
            continue
        ax.axvspan(max(lo, x0), min(hi, x1), color=color, alpha=alpha, lw=0, zorder=0)
        xc = (max(lo, x0) + min(hi, x1)) / 2
        ax.text(xc, y0 + (y1 - y0) * levels[li % len(levels)], label, rotation=90,
                va="top", ha="center", fontsize=fontsize, color=text_color, zorder=5)
        li += 1
