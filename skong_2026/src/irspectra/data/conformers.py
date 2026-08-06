"""Per-conformer loading, and the two ways to collapse conformers back to one molecule.

This is the representation steps 3-6 actually train on: every converged geometry is its own
sample, and the `molecule` column is the group id that keeps a molecule's conformers on one
side of any train/test split.

Reads the committed CSVs in data/processed/, never the raw xTB tables -- see
tools/build_dataset.py, the only code in the repo that opens one.

Note the asymmetry, which is a size decision, not a scientific one:
  conformers_monodi.csv.gz      EVERY mono+dimer conformer (~25k), so the uncapped Step-4
                                cross-validation stays reproducible
  conformers_trimers_cap10.csv.gz  trimers capped at 10/molecule, because the full trimer set is
                                104k conformers and will not fit in a committed CSV
So `cap=None` on trimers still yields the pre-capped 10/molecule; `cap=K` for K < 10 caps
further, and reproduces the committed _cap10 outputs at K=10.
"""
import os

import numpy as np
import pandas as pd

from irspectra.paths import PROCESSED_DIR
from irspectra.data.spectra import SPEC_COLS, broaden_normalize

# (filename, lengths it covers) -- the mono+dimer file carries both in one `length` column.
# .csv.gz because the plain trimer table is 170 MB; pandas.read_csv handles .gz transparently.
CONFORMER_SOURCES = [
    ("conformers_monodi.csv.gz", (1, 2)),
    ("conformers_trimers_cap10.csv.gz", (3,)),
]


def load_conformers(data_dir=PROCESSED_DIR, lengths=(1, 2), fwhm=15.0, verbose=True,
                    cap=None, cap_seed=0):
    """Load conformers for the requested lengths, metal complexes only, broaden+normalize each.

    Apo rows (metal '0') are dropped, so this returns metal complexes only.

    Args:
        data_dir (str): Directory holding the conformer CSVs. Defaults to
            PROCESSED_DIR.
        lengths (tuple): Peptide lengths to include, from (1, 2, 3). Defaults to
            (1, 2).
        fwhm (float): Gaussian broadening FWHM in cm^-1. Defaults to 15.0.
        verbose (bool): Print per-file conformer counts and the final summary.
            Defaults to True.
        cap (int, optional): Keep at most this many conformers per molecule
            (reproducible RandomState(cap_seed), molecules iterated in sorted
            order) so per-metal conformer counts are balanced -- e.g. cap=10
            trims the octahedral Co/Ni 20->10 and keeps all Cu/Zn. Defaults to
            None, i.e. whatever the CSV holds: every mono+dimer conformer, and
            trimers already capped at 10 by tools/build_dataset.py.
        cap_seed (int): Seed for the capping subsample. Defaults to 0.

    Returns:
        tuple: ``(meta, X)`` aligned row-for-row. ``meta`` (pandas.DataFrame) has
        amino_acid, metal, length, [total_free_energy if present], and a
        ``molecule`` group id (= "amino_acid|metal") so every conformer of a
        molecule shares a group. ``X`` (numpy.ndarray) holds the broadened,
        area-normalized spectra, shape (n_conformers, 791).

    Raises:
        ValueError: If no conformer CSV covers ``lengths`` under ``data_dir``;
            build them with ``python tools/build_dataset.py``.
    """
    frames = []
    for fname, covers in CONFORMER_SOURCES:
        if not set(covers) & set(lengths):
            continue
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            if verbose:
                print(f"  (skip missing {fname})")
            continue
        df = pd.read_csv(path)
        df = df[df["length"].isin(lengths)]              # monodi covers 1 and 2 in one file
        frames.append(df)
        if verbose:
            print(f"  loaded {fname:30s} {len(df):>7d} conformers (len {sorted(set(covers) & set(lengths))})")
    if not frames:
        raise ValueError(f"no conformer CSV for lengths {lengths} under {data_dir}; "
                         "build them with `python tools/build_dataset.py`")
    raw = pd.concat(frames, ignore_index=True)
    raw["amino_acid"] = raw["amino_acid"].astype(str)
    raw["metal"] = raw["metal"].astype(str)
    raw = raw.dropna(subset=["amino_acid", "metal"])
    raw = raw[raw["metal"] != "0"].reset_index(drop=True)
    if cap is not None:                                            # cap BEFORE broadening -> bounds peak memory
        mol = (raw["amino_acid"].astype(str) + "|" + raw["metal"].astype(str)).to_numpy()
        rng = np.random.RandomState(cap_seed)
        idx = {}
        for i, mm in enumerate(mol):
            idx.setdefault(mm, []).append(i)
        keep = []
        for mm in sorted(idx):
            ii = np.array(idx[mm])
            keep.append(rng.choice(ii, size=cap, replace=False) if len(ii) > cap else ii)
        raw = raw.iloc[np.sort(np.concatenate(keep))].reset_index(drop=True)
    sticks = raw[SPEC_COLS].to_numpy(float)
    X = broaden_normalize(sticks, fwhm=fwhm)
    meta_cols = ["amino_acid", "metal", "length"] + [c for c in ("provenance", "total_free_energy")
                                                     if c in raw.columns]
    meta = raw[meta_cols].copy()
    meta["molecule"] = meta["amino_acid"] + "|" + meta["metal"]
    meta = meta.reset_index(drop=True)
    if verbose:
        captxt = f" (capped <= {cap}/molecule)" if cap is not None else ""
        print(f"  -> {len(meta)} conformers, {meta['molecule'].nunique()} molecules, "
              f"metals {sorted(meta['metal'].unique())}{captxt}")
    return meta, X


# k_B in Hartree/K (total_free_energy is in Hartree -> RT is in Hartree too)
_RT_HARTREE_PER_K = 3.166811563e-6


# NOTE: no callers anywhere in the repo -- steps 5 and 6 collapse with mean_average(), so
# nothing in output/ is Boltzmann-weighted. Delete unless the energy-weighted variant is
# still planned; leaving it here invites the assumption that the results use it.
def boltzmann_average(meta, X, temperature=298.15, verbose=True):
    """Collapse each molecule's conformers into ONE Boltzmann-weighted-average
    spectrum.

    Weights w_i = softmax(-(G_i - G_min)/RT) from total_free_energy (Hartree);
    each averaged spectrum is re-normalized to area 1. Conformers with a NaN
    energy get ~0 weight, and a molecule whose weights all underflow falls back
    to a uniform average.

    Args:
        meta (pandas.DataFrame): Conformer table from load_conformers(), needing
            a ``molecule`` group id and a ``total_free_energy`` column, aligned
            row-for-row with ``X`` and carrying a positional (reset) index.
        X (numpy.ndarray): Broadened spectra, shape (n_conformers, 791).
        temperature (float): Temperature in kelvin for the Boltzmann weights.
            Defaults to 298.15.
        verbose (bool): Print the conformer -> molecule reduction and RT.
            Defaults to True.

    Returns:
        tuple: ``(meta_mol, X_avg)`` -- one row per molecule. ``meta_mol``
        (pandas.DataFrame) has amino_acid, metal, length, molecule and
        n_conformers; ``X_avg`` (numpy.ndarray) holds the averaged, area-1
        spectra, shape (n_molecules, 791).

    Raises:
        KeyError: If ``meta`` lacks the ``total_free_energy`` column -- use
            mean_average() instead when the energies are unavailable.
    """
    rt = _RT_HARTREE_PER_K * temperature
    rows, avg_specs = [], []
    for mol, sub in meta.groupby("molecule", sort=False):
        pos = sub.index.to_numpy()                         # positions into X (meta is reset_index)
        e = sub["total_free_energy"].to_numpy(float)
        de = e - np.nanmin(e)
        de = np.where(np.isfinite(de), de, np.inf)          # NaN energy -> ~0 weight
        w = np.exp(-de / rt)
        tot = w.sum()
        w = (w / tot) if (np.isfinite(tot) and tot > 0) else np.full(len(e), 1.0 / len(e))
        avg = (w[:, None] * X[pos]).sum(axis=0)
        area = avg.sum()
        if area > 0:
            avg = avg / area
        avg_specs.append(avg)
        r = sub.iloc[0]
        rows.append({"amino_acid": r["amino_acid"], "metal": r["metal"],
                     "length": int(r["length"]), "molecule": mol, "n_conformers": len(sub)})
    meta_mol = pd.DataFrame(rows).reset_index(drop=True)
    X_avg = np.asarray(avg_specs, float)
    if verbose:
        print(f"  boltzmann-average: {len(meta)} conformers -> {len(meta_mol)} molecules "
              f"(T={temperature} K, RT={rt:.3e} Eh)")
    return meta_mol, X_avg


def mean_average(meta, X, verbose=True):
    """Collapse each molecule's conformers into ONE plain (uniform) mean spectrum,
    re-normalized to area 1.

    Works without an energy column, unlike boltzmann_average().

    Args:
        meta (pandas.DataFrame): Conformer table from load_conformers(), needing
            a ``molecule`` group id, aligned row-for-row with ``X`` and carrying
            a positional (reset) index.
        X (numpy.ndarray): Broadened spectra, shape (n_conformers, 791).
        verbose (bool): Print the conformer -> molecule reduction. Defaults to
            True.

    Returns:
        tuple: ``(meta_mol, X_avg)`` -- one row per molecule. ``meta_mol``
        (pandas.DataFrame) has amino_acid, metal, length, molecule and
        n_conformers; ``X_avg`` (numpy.ndarray) holds the mean, area-1 spectra,
        shape (n_molecules, 791).
    """
    rows, avg_specs = [], []
    for mol, sub in meta.groupby("molecule", sort=False):
        pos = sub.index.to_numpy()
        avg = X[pos].mean(axis=0)
        area = avg.sum()
        if area > 0:
            avg = avg / area
        avg_specs.append(avg)
        r = sub.iloc[0]
        rows.append({"amino_acid": r["amino_acid"], "metal": r["metal"],
                     "length": int(r["length"]), "molecule": mol, "n_conformers": len(sub)})
    meta_mol = pd.DataFrame(rows).reset_index(drop=True)
    X_avg = np.asarray(avg_specs, float)
    if verbose:
        print(f"  mean-average: {len(meta)} conformers -> {len(meta_mol)} molecules")
    return meta_mol, X_avg
