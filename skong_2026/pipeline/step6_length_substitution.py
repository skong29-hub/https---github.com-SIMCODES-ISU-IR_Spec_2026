#!/usr/bin/env python3
"""
step6_length_substitution.py -- can short peptides substitute for long ones? A 2x2 design
(anchor length x added length), always tested on held-out trimers.

800-MOLECULE anchors, 10 hardcoded seeds, fixed molecule splits and anchor draws (ANCHOR_SEEDS /
TEST_SEEDS). Every molecule is represented by ALL its conformers (per-conformer samples), not one
averaged spectrum -- hence the `_perconf` tag on the artifacts this writes. Grouping stays by
molecule (a molecule's conformers never straddle a split). Metric = per-conformer macro-F1
(+ accuracy + per-metal F1).

    python pipeline/step6_length_substitution.py --cap 10
      -> output/step6/length_substitution_perconf_cap10.csv (+ summary, contrasts, meta, figure svg)
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["savefig.transparent"] = True
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

from irspectra import paths
from irspectra.modeling import protocol, metrics
from irspectra.config import METAL_ORDER, SEEDS, TEST_SEEDS, TEST_FRACTION, RF_PARAMS, RF_PARAMS_SOURCE, ci95

OUT = paths.output_dir("step6")
FIG = paths.FIGURES_DIR
# a THIRD seed stream: the anchor draw must vary independently of the test
# split (TEST_SEEDS) and the RF (SEEDS), else 'lucky anchor' and 'lucky test' are confounded
ANCHOR_SEEDS = [1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10010]   # anchor draw, this script only
ANCHOR_N = 800
GRID = [0, 100, 200, 350, 500, 700, 900, 1200, 1600, 2200, 3000, 4000]     # molecules added (+ arm cap)
ARMS = [("A1_short_add_trimer",  "short",  "trimer"),
        ("A2_trimer_add_trimer", "trimer", "trimer"),
        ("A3_trimer_add_short",  "trimer", "short"),
        ("A4_short_add_short",   "short",  "short")]
CONTRASTS = [("A2_trimer_add_trimer", "A3_trimer_add_short",  "headline: add trimer vs short, trimer anchor"),
             ("A1_short_add_trimer",  "A4_short_add_short",   "add trimer vs short, short anchor"),
             ("A2_trimer_add_trimer", "A1_short_add_trimer",  "trimer vs short anchor, add trimer"),
             ("A3_trimer_add_short",  "A4_short_add_short",   "trimer vs short anchor, add short")]


def strat_sample(mols, metal_of, n, rng):
    """Draw n molecules stratified evenly across the four metals.

    Takes ``n // len(METAL_ORDER)`` per metal, then tops the selection up from
    whatever is left over to reach n. Returns fewer than n only when the
    candidate pool is exhausted.

    Args:
        mols (sequence): Candidate molecule ids to sample from.
        metal_of (dict): Molecule id -> metal label.
        n (int): Target number of molecules.
        rng (numpy.random.RandomState): Seeded generator, for reproducibility.

    Returns:
        list: The chosen molecule ids, grouped by metal in METAL_ORDER order followed
        by any top-up draws.
    """
    per = n // len(METAL_ORDER); chosen = []
    for m in METAL_ORDER:
        pool = [x for x in mols if metal_of[x] == m]
        chosen += rng.choice(pool, size=min(per, len(pool)), replace=False).tolist()
    if len(chosen) < n:
        rest = [x for x in mols if x not in set(chosen)]
        chosen += rng.choice(rest, size=min(n - len(chosen), len(rest)), replace=False).tolist()
    return chosen


def balanced_order(mols, metal_of, rng):
    """Order molecules round-robin across metals so every prefix stays balanced.

    Each metal's molecules are permuted independently, then emitted one per
    metal per round; metals whose pool runs out simply drop out of later rounds.
    This is what lets the growth curve truncate the order at any grid point and
    still add a metal-balanced slice.

    Args:
        mols (sequence): Molecule ids to order.
        metal_of (dict): Molecule id -> metal label.
        rng (numpy.random.RandomState): Seeded generator, for reproducibility.

    Returns:
        list: Every molecule in ``mols`` whose metal is in METAL_ORDER, in round-robin
        order. Molecules of any other metal are dropped.
    """
    by = {m: rng.permutation([x for x in mols if metal_of[x] == m]).tolist() for m in METAL_ORDER}
    order, i = [], 0
    while any(len(by[m]) > i for m in METAL_ORDER):
        for m in METAL_ORDER:
            if len(by[m]) > i:
                order.append(by[m][i])
        i += 1
    return order


def mol_row_map(meta):
    """Map each molecule to the row indices of its conformers.

    Args:
        meta (pandas.DataFrame): Conformer table with a ``molecule`` column,
            aligned row-for-row with the matching spectra array.

    Returns:
        dict: ``{molecule_id: numpy.ndarray}`` of row indices, in the order they
        appear in ``meta``.
    """
    d = {}
    for i, m in enumerate(meta["molecule"].to_numpy()):
        d.setdefault(m, []).append(i)
    return {m: np.array(v) for m, v in d.items()}


def rows_for(rowmap, mols):
    """Gather the conformer row indices of several molecules into one array.

    Args:
        rowmap (dict): Molecule -> row indices, as built by mol_row_map().
        mols (sequence): Molecule ids to gather, in the order given.

    Returns:
        numpy.ndarray: The concatenated row indices, or an empty int array when
        ``mols`` is empty.

    Raises:
        KeyError: If a requested molecule is absent from ``rowmap``.
    """
    return np.concatenate([rowmap[m] for m in mols]) if len(mols) else np.array([], int)


def main():
    """Run the 2x2 anchor-length x added-length substitution study on held-out trimers.

    Parses ``--cap`` from the command line. Per seed: split trimer MOLECULES
    80/20 (grouped), draw a stratified 800-molecule anchor of each length, then
    for each of the four arms (short/trimer anchor x short/trimer added data)
    grow the training set along GRID and score per-conformer macro-F1, accuracy
    and per-metal F1 on every conformer of the held-out 20%. Finally it aggregates
    per-arm means with 95% CIs, runs the paired Wilcoxon contrasts (skipped if
    SciPy is unavailable) and draws the four-arm figure against the A2 trimer
    ceiling.

    Writes (with a ``_cap{K}`` suffix when capping):
        output/step6/length_substitution_perconf*.csv: One row per (seed, arm,
            n_added).
        output/step6/length_substitution_perconf*_summary.csv: Per-arm mean and
            95% CI per grid point.
        output/step6/length_substitution_perconf*_contrasts.csv: Paired
            arm-vs-arm differences with Wilcoxon p-values.
        output/step6/length_substitution_perconf*_meta.json: Design constants and
            the trimer ceiling.
        figures/step6_length_substitution_perconf*.svg: The four-arm figure.

    Raises:
        SystemExit: Propagated from argparse on an invalid command line.
        ValueError: Propagated from protocol.load_monodi_and_trimers() if the conformer
            CSVs are missing from data/processed/.
    """
    ap = argparse.ArgumentParser()
    protocol.add_cap_flag(ap)
    a = ap.parse_args()
    sfx = protocol.suffix_for(a)
    print("loading mono+di + trimers (per-conformer) ...", flush=True)
    meta_md, X_md, meta_tr, X_tr = protocol.load_monodi_and_trimers(a)
    md_rows = mol_row_map(meta_md); tr_rows = mol_row_map(meta_tr)
    md_metal = dict(zip(meta_md["molecule"], meta_md["metal"]))
    tr_metal = dict(zip(meta_tr["molecule"], meta_tr["metal"]))
    y_md = meta_md["metal"].to_numpy(); y_tr = meta_tr["metal"].to_numpy()
    md_mols = list(md_rows.keys())
    tr_mols = np.array(sorted(tr_rows.keys()))
    tr_mol_metal = np.array([tr_metal[m] for m in tr_mols])
    print(f"mono+di {len(md_mols)} mol / {len(y_md)} conf | trimers {len(tr_mols)} mol / {len(y_tr)} conf "
          f"| RF [{RF_PARAMS_SOURCE}] {RF_PARAMS}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    out_csv = os.path.join(OUT, f"length_substitution_perconf{sfx}.csv")
    rows = []
    for i, seed in enumerate(SEEDS):
        # split trimer MOLECULES 80/20 (grouped) -> partition is reproducible from TEST_SEEDS[i]
        pool_i, test_i = next(iter(GroupShuffleSplit(1, test_size=TEST_FRACTION,
                                   random_state=TEST_SEEDS[i]).split(tr_mols, tr_mol_metal, tr_mols)))
        test_mols = tr_mols[test_i]; pool_mols = list(tr_mols[pool_i])
        te_rows = rows_for(tr_rows, test_mols)
        Xte, yte = X_tr[te_rows], y_tr[te_rows]                    # test = every conformer of the 20% molecules

        arng = np.random.RandomState(ANCHOR_SEEDS[i])
        tri_anchor = strat_sample(pool_mols, tr_metal, ANCHOR_N, arng)   # shared A2/A3
        md_anchor = strat_sample(md_mols, md_metal, ANCHOR_N, arng)      # shared A1/A4
        tri_a_set, md_a_set = set(tri_anchor), set(md_anchor)
        tri_add = balanced_order([m for m in pool_mols if m not in tri_a_set], tr_metal, arng)   # A1 & A2
        md_add4 = balanced_order([m for m in md_mols if m not in md_a_set], md_metal, arng)      # A4
        # A3 is anchored on TRIMERS, so the 800 short anchor molecules are free to be ADDED
        # data here -> A3 runs out to every short molecule while A4 (anchored on them) stops short
        md_add3 = md_add4 + balanced_order(md_anchor, md_metal, arng)   # A3 (first ~876 == A4, then rest)

        anchors = {"short": md_anchor, "trimer": tri_anchor}
        add_orders = {"A1_short_add_trimer": tri_add, "A2_trimer_add_trimer": tri_add,
                      "A3_trimer_add_short": md_add3, "A4_short_add_short": md_add4}

        for arm, atype, dtype in ARMS:
            aX, arows, ay = (X_tr, tr_rows, y_tr) if atype == "trimer" else (X_md, md_rows, y_md)
            anch_rows = rows_for(arows, anchors[atype])
            Xa, ya = aX[anch_rows], ay[anch_rows]
            dX, drows, dy = (X_tr, tr_rows, y_tr) if dtype == "trimer" else (X_md, md_rows, y_md)
            order = add_orders[arm]
            grid = [g for g in GRID if g <= len(order)]
            if len(order) not in grid:
                grid.append(len(order))
            for x in grid:
                add_mols = order[:x]
                add_rows = rows_for(drows, add_mols)
                if len(add_rows):
                    Xtr_, ytr_ = np.vstack([Xa, dX[add_rows]]), np.concatenate([ya, dy[add_rows]])
                else:
                    Xtr_, ytr_ = Xa, ya
                rf = RandomForestClassifier(random_state=seed, n_jobs=-1, **RF_PARAMS).fit(Xtr_, ytr_)
                pred = rf.predict(Xte)
                permetal = metrics.per_metal_f1(yte, pred)
                row = {"seed": seed, "test_seed": TEST_SEEDS[i], "anchor_seed": ANCHOR_SEEDS[i],
                       "arm": arm, "anchor_type": atype, "added_type": dtype,
                       "n_added": int(x), "n_conformers_added": int(len(add_rows)),
                       "n_train_conformers": int(len(ytr_)),
                       "accuracy": float(accuracy_score(yte, pred)),
                       "macro_f1": float(metrics.macro_f1(yte, pred))}
                row.update({f"f1_{m}": float(v) for m, v in zip(METAL_ORDER, permetal)})
                rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)        # incremental (crash-safe)
        print(f"  seed {seed} done ({len(rows)} rows, {len(te_rows)} test conformers)", flush=True)

    df = pd.DataFrame(rows)
    summ = []
    for arm in df["arm"].unique():
        for x in sorted(df[df.arm == arm]["n_added"].unique()):
            d = df[(df.arm == arm) & (df.n_added == x)]
            for metric in ["macro_f1", "accuracy"]:
                v = d[metric].to_numpy()
                summ.append({"arm": arm, "n_added": int(x), "metric": metric,
                             "mean": float(v.mean()), "ci95": ci95(v)})
    pd.DataFrame(summ).to_csv(os.path.join(OUT, f"length_substitution_perconf{sfx}_summary.csv"), index=False)

    piv = df.pivot_table(index=["seed", "n_added"], columns="arm", values="macro_f1")
    crows = []
    for arm_a, arm_b, desc in CONTRASTS:        # NOT `a` -- that is the argparse namespace above
        for x in sorted(set(df[df.arm == arm_a]["n_added"]) & set(df[df.arm == arm_b]["n_added"])):
            sub = piv.xs(x, level="n_added")[[arm_a, arm_b]].dropna()
            diff = (sub[arm_a] - sub[arm_b]).to_numpy()
            p = None
            if wilcoxon is not None and len(diff) >= 5 and np.any(diff != 0):
                try:
                    p = float(wilcoxon(sub[arm_a].to_numpy(), sub[arm_b].to_numpy()).pvalue)
                except Exception:
                    p = None
            crows.append({"contrast": f"{arm_a} - {arm_b}", "desc": desc, "n_added": int(x),
                          "n_seeds": int(len(diff)), "mean_diff": float(diff.mean()),
                          "ci95": ci95(diff), "wilcoxon_p": p})
    pd.DataFrame(crows).to_csv(os.path.join(OUT, f"length_substitution_perconf{sfx}_contrasts.csv"), index=False)

    # per-conformer trimer ceiling = A2 (trimer anchor + all trimers) at its max n_added
    a2 = df[df.arm == "A2_trimer_add_trimer"]
    ceil = float(a2[a2.n_added == a2.n_added.max()]["macro_f1"].mean())
    json.dump({"design": "2x2 anchor x added length, PER-CONFORMER, tested on held-out trimers",
               "anchor_n_molecules": ANCHOR_N, "seeds": SEEDS, "test_seeds": TEST_SEEDS, "anchor_seeds": ANCHOR_SEEDS,
               "grid_molecules": GRID, "arms": [a[0] for a in ARMS], "trimer_ceiling_macro_f1_perconf": ceil,
               "metric": "per-conformer macro_f1", "grouping": "molecule", "rf_params": RF_PARAMS},
              open(os.path.join(OUT, f"length_substitution_perconf{sfx}_meta.json"), "w"), indent=2)

    # figure (ceiling = per-conformer A2 plateau)
    s = pd.read_csv(os.path.join(OUT, f"length_substitution_perconf{sfx}_summary.csv")); s = s[s.metric == "macro_f1"]
    arms_plot = [("A2_trimer_add_trimer", "#1BAF7A", "-", "o", "trimer anchor + trimers (in-domain)"),
                 ("A3_trimer_add_short",  "#1BAF7A", ":", "s", "trimer anchor + short data"),
                 ("A1_short_add_trimer",  "#EB6834", "-", "o", "short anchor + trimers"),
                 ("A4_short_add_short",   "#EB6834", ":", "s", "short anchor + short data")]
    fig, ax = plt.subplots(figsize=(11, 7.6))
    for arm, color, ls, mk, label in arms_plot:
        d = s[s.arm == arm].sort_values("n_added")
        ax.fill_between(d["n_added"], d["mean"] - d["ci95"], d["mean"] + d["ci95"],
                        color=color, alpha=0.13, zorder=1)                    # shaded 95% CI band
        ax.errorbar(d["n_added"], d["mean"], yerr=d["ci95"], marker=mk, color=color, ls=ls, lw=2.3,
                    markersize=6, capsize=4, capthick=1.5, elinewidth=1.5, label=label, zorder=3)
    ax.axhline(ceil, ls="--", color="0.4", lw=2, zorder=2, label=f"per-conformer trimer ceiling = {ceil:.3f}")
    ax.set_xlabel("molecules added to the 800-molecule anchor", fontsize=16, labelpad=12)
    ax.set_ylabel("macro-F1 on held-out trimer conformers", fontsize=16, labelpad=12)
    ax.set_title("How much trimer data do you actually need? (per-conformer)\n"
                 "2×2 anchor × added length  (10-seed 95% CI)", fontsize=15, pad=14)
    ax.set_xlim(-100, float(s["n_added"].max()))          # full-length arms (A1/A2) run to the right edge
    for sp in ax.spines.values():
        sp.set_linewidth(2.5)
    ax.tick_params(axis="both", which="major", length=8, width=2, labelsize=12)
    handles, labels = ax.get_legend_handles_labels()          # legend fills column-major; order for anchor-grouped rows
    order = ["trimer anchor + trimers (in-domain)", "short anchor + trimers",
             f"per-conformer trimer ceiling = {ceil:.3f}",
             "trimer anchor + short data", "short anchor + short data"]
    hd = dict(zip(labels, handles))
    ax.legend([hd[l] for l in order if l in hd], [l for l in order if l in hd],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False,
              fontsize=12, columnspacing=2.4, handlelength=2.8)
    fig.tight_layout()
    out = os.path.join(FIG, f"step6_length_substitution_perconf{sfx}.svg")
    fig.savefig(out, dpi=600, transparent=True, bbox_inches="tight")
    print(f"\nper-conformer trimer ceiling (A2 plateau) = {ceil:.3f}")
    print(f"wrote {out_csv} (+ summary, contrasts, meta) and {out}")


if __name__ == "__main__":
    main()
