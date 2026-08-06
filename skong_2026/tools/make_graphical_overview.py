#!/usr/bin/env python3
"""
make_graphical_overview.py -- build figures/graphical_overview.svg, the one-glance summary of the
whole project for the README's Graphical Overview section.

Three bands, read top to bottom:
  1. the method       peptide + M2+ -> xTB -> broaden -> RandomForest -> which metal?
  2. the result       train on short (0.81) -> zero-shot on long (0.63) -> add diverse trimers (0.72)
  3. the takeaway     one sentence, plus the per-metal recalls that explain where the errors are

Every number is read from output/ and data/processed/dataset_manifest.json at build time --
nothing is hardcoded, so the figure cannot drift away from the results the way a slide would.
The manifest matters here: the committed trimer table ships capped, so counting its rows would
under-report the dataset by 35k conformers.

    python tools/make_graphical_overview.py
      -> figures/graphical_overview.svg (+ .png preview)
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from irspectra import paths, results
from irspectra.data import processed
from irspectra.data.spectra import STEP, WN
from irspectra.viz.palette import METAL_COLORS, METAL_LABELS
from irspectra.config import METAL_ORDER

# The same three accents the existing panels already use, so this figure sits with the others:
# blue = in-domain (step4_cv), orange = zero-shot (step5), green = recovered (step6).
IN_DOMAIN = "#2A78D6"
ZERO_SHOT = "#EB6834"
RECOVERED = "#1BAF7A"
INK = "#1a1a18"
MUTED = "#6b6862"
RULE = "#c9c5bd"
WASH = "#f5f4ef"


def load_numbers():
    """Read every value the figure prints from the committed results and the dataset manifest.

    Returns:
        dict: the three macro-F1 scores (``monodi``, ``trimer``, ``ceiling``), the conformer
        counts (``monodi_conf``, ``trimer_sim``, ``sim_total``) and ``recall``, the per-metal
        out-of-fold recall keyed by metal label.

    Raises:
        FileNotFoundError: If the pipeline has not been run, or the dataset is not built.
    """
    zs = results.load_json("step5", "zeroshot_summary_cap10.json")
    meta6 = results.load_json("step6", "length_substitution_perconf_cap10_meta.json")
    cm = results.load_confusion("step4", "confusion_perconformer_cap10.csv")
    man = processed.load_manifest()      # counting rows would under-report: trimers ship capped

    sim = {int(k): v for k, v in man["conformers_simulated"].items()}
    return {
        "monodi": zs["monodi"]["mean"],
        "trimer": zs["trimer_whole"]["mean"],
        "ceiling": meta6["trimer_ceiling_macro_f1_perconf"],
        "monodi_conf": sim[1] + sim[2],
        "trimer_sim": sim[3],
        "sim_total": man["conformers_simulated_total"],
        "recall": {m: float(cm.loc[m, m]) for m in METAL_ORDER},
    }


def _arrow(ax, x0, x1, y, color=RULE, lw=1.4):
    """Draw one connector arrow between two boxes."""
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=lw, color=color, zorder=2))


def draw_pipeline(ax, n):
    """Band 1: the method, compressed to a single strip.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw into; cleared and scaled to 0-1.
        n (dict): The numbers from load_numbers().
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    steps = [("peptide + M$^{2+}$", "Co · Ni · Cu · Zn"),
             ("xTB (GFN2)", f"{n['sim_total']:,} spectra simulated"),
             ("broaden + normalize", f"{len(WN)} bins @ {STEP} cm$^{{-1}}$"),
             ("RandomForest", "grouped CV, 10 seeds"),
             ("which metal?", "4-way classification")]
    w, gap = 0.166, 0.043
    for i, (lab, sub) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.30), w, 0.66,
                                    boxstyle="round,pad=0.006,rounding_size=0.014",
                                    facecolor="none", edgecolor=RULE, linewidth=1.4, zorder=2))
        ax.text(x + w / 2, 0.74, lab, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=INK, zorder=3)
        ax.text(x + w / 2, 0.48, sub, ha="center", va="center", fontsize=8.5,
                color=MUTED, zorder=3)
        if i < len(steps) - 1:
            _arrow(ax, x + w + 0.006, x + w + gap - 0.006, 0.63)
    # the four class colours, under the last box. scatter, not Circle: a Circle in an axes this
    # wide renders as an ellipse.
    last = 4 * (w + gap)
    xs = last + w * np.linspace(0.20, 0.80, len(METAL_ORDER))
    ax.scatter(xs, [0.13] * len(METAL_ORDER), s=110,
               c=[METAL_COLORS[m] for m in METAL_ORDER], zorder=3, clip_on=False)


def _card(ax, x, w, colour, title, sub, value, caption):
    """Draw one result card: a title, a big coloured number, and a caption."""
    ax.add_patch(FancyBboxPatch((x, 0.10), w, 0.84,
                                boxstyle="round,pad=0.006,rounding_size=0.016",
                                facecolor="none", edgecolor=RULE, linewidth=1.4, zorder=2))
    ax.add_patch(FancyBboxPatch((x, 0.855), w, 0.085,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor=colour, edgecolor="none", alpha=0.9, zorder=3))
    cx = x + w / 2
    ax.text(cx, 0.895, title, ha="center", va="center", fontsize=10,
            fontweight="bold", color="white", zorder=4)
    ax.text(cx, 0.745, sub, ha="center", va="center", fontsize=8.5, color=MUTED, zorder=4)
    ax.text(cx, 0.47, f"{value:.2f}", ha="center", va="center", fontsize=40,
            fontweight="bold", color=colour, zorder=4)
    ax.text(cx, 0.20, caption, ha="center", va="center", fontsize=8.5, color=MUTED,
            zorder=4, linespacing=1.5)


def draw_transfer(ax, n):
    """Band 2: the result, as three cards joined by what happens between them.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw into; cleared and scaled to 0-1.
        n (dict): The numbers from load_numbers().
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    w, gap = 0.295, 0.058
    drop = n["monodi"] - n["trimer"]

    _card(ax, 0.0, w, IN_DOMAIN, "train: monomer + dimer",
          f"{n['monodi_conf']:,} conformers", n["monodi"],
          "macro-F1 on a held-out\nmono/di test set")
    _card(ax, w + gap, w, ZERO_SHOT, "test: unseen trimers",
          f"{n['trimer_sim']:,} conformers", n["trimer"],
          f"−{drop:.2f} length drop\nnever trained on a trimer")
    _card(ax, 2 * (w + gap), w, RECOVERED, "add diverse trimers",
          "duplicates plateau lower", n["ceiling"],
          "in-domain trimer ceiling\nreached with added variety")

    for i, (lab, colour) in enumerate([("zero-shot", ZERO_SHOT), ("+ diversity", RECOVERED)]):
        x0 = (i + 1) * w + i * gap + 0.008
        _arrow(ax, x0, x0 + gap - 0.016, 0.50, color=colour, lw=2.0)
        ax.text(x0 + gap / 2 - 0.004, 0.585, lab, ha="center", fontsize=8.5,
                fontweight="bold", color=colour)


def draw_takeaway(ax, n):
    """Band 3: the one-sentence takeaway, plus where the errors actually are.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw into; cleared and scaled to 0-1.
        n (dict): The numbers from load_numbers().
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.0, 0.40), 1.0, 0.60,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor=WASH, edgecolor="none", zorder=1))
    ax.text(0.018, 0.80, "Metal identity survives the jump to longer peptides — but not intact. "
                         "Diversity of the added long peptides, not their volume, recovers it.",
            ha="left", va="center", fontsize=10.5, color=INK, zorder=2)
    # the Co/Ni explanation lives here, not beside the recall row -- there it collided with the
    # rightmost metal label
    ax.text(0.018, 0.56, "Ni and Co are the hardest pair: adjacent in the first transition series "
                         "and both octahedral, so their metal–ligand bands overlap.",
            ha="left", va="center", fontsize=9, color=MUTED, zorder=2)

    ax.text(0.018, 0.15, "PER-METAL RECALL", ha="left", va="center", fontsize=8.5,
            fontweight="bold", color=MUTED)
    # ordered worst-first: the Co/Ni confusion is the story, so it should read first
    order = sorted(METAL_ORDER, key=lambda m: n["recall"][m])
    xs = np.linspace(0.30, 0.86, len(order))
    ax.scatter(xs, [0.15] * len(order), s=120, c=[METAL_COLORS[m] for m in order], zorder=3)
    for x, m in zip(xs, order):
        ax.text(x + 0.024, 0.15, f"{METAL_LABELS[m]} {n['recall'][m]:.2f}", ha="left",
                va="center", fontsize=9.5, color=INK)


def main():
    """Render the three-band overview and write it to figures/.

    Writes:
        figures/graphical_overview.svg: The figure, embedded by the README.
        figures/_v_overview.png: A white-background preview (git-ignored).

    Raises:
        FileNotFoundError: If output/ or data/processed/ has not been built.
    """
    n = load_numbers()
    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.95, 2.4, 1.15], hspace=0.20,
                          left=0.03, right=0.97, top=0.90, bottom=0.03)
    ax_pipe, ax_res, ax_take = (fig.add_subplot(gs[i]) for i in range(3))

    ax_pipe.set_title("SIMULATE  →  FEATURIZE  →  CLASSIFY", loc="left", fontsize=9.5,
                      fontweight="bold", color=MUTED, pad=7)
    ax_res.set_title("DOES THE METAL SIGNAL TRANSFER FROM SHORT PEPTIDES TO LONG?", loc="left",
                     fontsize=9.5, fontweight="bold", color=MUTED, pad=7)

    draw_pipeline(ax_pipe, n)
    draw_transfer(ax_res, n)
    draw_takeaway(ax_take, n)

    fig.suptitle("Long Story Short — metal identification that transfers from short peptides to long",
                 fontsize=14.5, fontweight="bold", color=INK, x=0.03, ha="left", y=0.975)

    out = os.path.join(paths.FIGURES_DIR, "graphical_overview.svg")
    fig.savefig(out, dpi=600, transparent=True, bbox_inches="tight")
    fig.savefig(os.path.join(paths.FIGURES_DIR, "_v_overview.png"),
                facecolor="white", dpi=110, bbox_inches="tight")
    print(f"in-domain {n['monodi']:.3f} | zero-shot {n['trimer']:.3f} | ceiling {n['ceiling']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
