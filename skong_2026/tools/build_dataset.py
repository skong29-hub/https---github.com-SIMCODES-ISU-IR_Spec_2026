#!/usr/bin/env python3
"""
build_dataset.py -- turn the raw xTB energy tables into the committed CSV dataset.

Run ONCE, by whoever has the raw tables. Everything downstream reads only the CSVs this
writes, so no pipeline script ever opens an energy_all_*.csv. That is what makes a fresh
clone reproducible: the raw tables are hundreds of MB and are not in git, the CSVs here are.

Each output row is metadata followed by the 791 raw stick intensities (spec_50 .. spec_4000),
the same shape as the source tables. Broadening is NOT baked in -- the loaders apply it at
read time, so FWHM stays a knob rather than a property of the file.

    python tools/build_dataset.py --raw-dir path/to/raw
      -> data/processed/molecules.csv                     one row per molecule (lowest-energy conformer)
      -> data/processed/conformers_monodi.csv.gz          every mono+dimer metal conformer
      -> data/processed/conformers_trimers_cap10.csv.gz   trimer conformers, capped per molecule
"""
import os
import json
import argparse

import numpy as np
import pandas as pd

from irspectra import paths
from irspectra.data.spectra import SPEC_COLS

# 6 significant figures, matching the source tables exactly.
FLOAT_FORMAT = "%.6g"

# molecules.csv is written as a plain .csv (~22 MB) so it reads like the raw tables it replaces.
# The conformer tables are written .csv.gz: the trimer set is 69,520 rows and 170 MB in plain
# text, over GitHub's 100 MB per-file cap, and 83% of every stick spectrum is zeros so gzip is
# enormously effective. pandas.read_csv decompresses .gz transparently -- no loader code cares.
CONFORMER_EXT = ".csv.gz"

TRIMER_CAP = 10          # conformers per trimer molecule; matches the --cap 10 the poster runs use
CAP_SEED = 0             # RandomState seed for that subsample -- fixed so the file is reproducible

# (filename, length, is_apo) -- the molecule-level table is built from all five
MOLECULE_SOURCES = [
    ("energy_all_metal_onemers.csv", 1, False),
    ("energy_all_metal_dimers.csv",  2, False),
    ("energy_all_metal_trimers.csv", 3, False),
    ("energy_all_aa_singles.csv",    1, True),
    ("energy_all_aa_dimers.csv",     2, True),
]
# provenance = the quantum-chemistry method behind the spectrum. Constant "xtb" today, so it
# looks like dead weight -- but the project also has DFT tables (dft_*_spectra_*.csv), and xTB
# reaches trimers where DFT stops at dimers. If the two are ever merged this column is what stops
# a model learning "method" instead of "metal". Every table carries it, so the merge stays a
# concat rather than a schema fix.
PROVENANCE = "xtb"

META_COLS = ["amino_acid", "metal", "length", "provenance", "total_free_energy"]
CONFORMER_META_COLS = ["amino_acid", "metal", "length", "provenance", "total_free_energy"]


def read_raw(path, length, verbose=True):
    """Read one raw xTB table, keeping only the columns the pipeline uses.

    Args:
        path (str): Absolute path to an energy_all_*.csv.
        length (int): Peptide length to tag every row with (1, 2 or 3).
        verbose (bool): Print the row count. Defaults to True.

    Returns:
        pandas.DataFrame: amino_acid, metal, total_free_energy (if present), the spec_* sticks,
        and a ``length`` column.

    Raises:
        FileNotFoundError: If the raw table is missing.
    """
    cols = pd.read_csv(path, nrows=0).columns
    keep = [c for c in (["amino_acid", "metal", "total_free_energy"] + SPEC_COLS) if c in cols]
    df = pd.read_csv(path, usecols=keep)
    df = pd.concat([df, pd.DataFrame({"length": length}, index=df.index)], axis=1)
    if verbose:
        print(f"  read {os.path.basename(path):34s} {len(df):>7d} conformers (len {length})", flush=True)
    return df


def tag_provenance(df, method=PROVENANCE):
    """Stamp the simulation method onto every row.

    Concatenated once rather than assigned in place -- inserting a column into a 795-column
    frame is slow enough to raise a pandas fragmentation warning.

    Args:
        df (pandas.DataFrame): Rows to tag.
        method (str): Value for the provenance column. Defaults to PROVENANCE.

    Returns:
        pandas.DataFrame: A new frame with a ``provenance`` column appended.
    """
    return pd.concat([df, pd.DataFrame({"provenance": method}, index=df.index)], axis=1)


def cap_per_molecule(df, cap, seed=CAP_SEED):
    """Keep at most ``cap`` conformers per molecule, reproducibly.

    Molecules are iterated in sorted order against a fixed RandomState, so the selection is
    identical on every machine -- which is what lets the capped file be committed.

    Args:
        df (pandas.DataFrame): Conformer rows with amino_acid and metal columns.
        cap (int): Maximum conformers to keep per molecule.
        seed (int): RandomState seed. Defaults to CAP_SEED.

    Returns:
        pandas.DataFrame: The capped rows, original order preserved, index reset.
    """
    mol = (df["amino_acid"].astype(str) + "|" + df["metal"].astype(str)).to_numpy()
    rng = np.random.RandomState(seed)
    idx = {}
    for i, m in enumerate(mol):
        idx.setdefault(m, []).append(i)
    keep = []
    for m in sorted(idx):
        ii = np.array(idx[m])
        keep.append(rng.choice(ii, size=cap, replace=False) if len(ii) > cap else ii)
    return df.iloc[np.sort(np.concatenate(keep))].reset_index(drop=True)


def write_csv(df, path, verbose=True):
    """Write one dataset CSV and report its size.

    FLOAT_FORMAT would otherwise apply to EVERY float column, including total_free_energy --
    and 6 significant figures on a ~-49 Hartree energy is a 5e-4 Hartree error, larger than the
    conformer spacings any energy-weighted average depends on. So the energy column is written
    as a pre-formatted full-precision string and float_format reaches only the spec_* columns.

    Args:
        df (pandas.DataFrame): Rows to write.
        path (str): Destination path; a .gz suffix makes pandas compress it.
        verbose (bool): Print shape and file size. Defaults to True.
    """
    out = df
    if "total_free_energy" in out.columns:
        out = out.copy()
        out["total_free_energy"] = out["total_free_energy"].map(
            lambda v: "" if pd.isna(v) else repr(float(v)))
    out.to_csv(path, index=False, float_format=FLOAT_FORMAT)
    if verbose:
        print(f"  wrote {os.path.basename(path):34s} {df.shape[0]:>7d} rows x {df.shape[1]} cols"
              f"  ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


def build_molecules(raw_dir, out_dir, verbose=True):
    """Build the molecule-level table: the lowest-free-energy conformer of every molecule.

    Apo rows (metal '0') are mapped to 'none' and kept -- they are a baseline in the Step 1-2
    data views, though never a predicted class.

    Args:
        raw_dir (str): Directory holding the raw energy_all_*.csv tables.
        out_dir (str): Destination directory.
        verbose (bool): Print progress. Defaults to True.

    Returns:
        pandas.DataFrame: The molecule table that was written.

    Raises:
        ValueError: If none of MOLECULE_SOURCES exists under ``raw_dir``.
    """
    frames = []
    for fname, length, _ in MOLECULE_SOURCES:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            if verbose:
                print(f"  (skip missing {fname})")
            continue
        frames.append(read_raw(path, length, verbose))
    if not frames:
        raise ValueError(f"no raw tables found under {raw_dir}")

    raw = pd.concat(frames, ignore_index=True)
    raw["metal"] = raw["metal"].astype(object).map(lambda m: "none" if str(m) == "0" else m)
    raw = raw.dropna(subset=["amino_acid", "metal", "total_free_energy"]).reset_index(drop=True)
    raw["amino_acid"] = raw["amino_acid"].astype(str)

    # one row per molecule: the conformer you would actually compute at deployment
    idx = raw.groupby(["amino_acid", "metal"])["total_free_energy"].idxmin()
    single = raw.loc[idx].sort_values(["length", "amino_acid", "metal"]).reset_index(drop=True)
    # build metadata separately and concat once -- inserting into a 796-column frame is slow
    meta = tag_provenance(single[["amino_acid", "metal", "length", "total_free_energy"]])
    out = pd.concat([meta[META_COLS], single[SPEC_COLS]], axis=1)
    write_csv(out, os.path.join(out_dir, "molecules.csv"), verbose)
    if verbose:
        print("  class balance (rows = length, cols = metal):")
        tbl = out.groupby(["length", "metal"]).size().unstack(fill_value=0)
        print("    " + tbl.to_string().replace("\n", "\n    "))
    return out


def build_conformers(raw_dir, out_dir, verbose=True):
    """Build the two per-conformer tables: all mono+dimers, and capped trimers.

    Mono+dimers are kept in full (~25k rows) because the uncapped Step-4 cross-validation is the
    poster's headline number. Trimers are capped at TRIMER_CAP because the full set is 104k
    conformers, which no single CSV should carry.

    Args:
        raw_dir (str): Directory holding the raw energy_all_*.csv tables.
        out_dir (str): Destination directory.
        verbose (bool): Print progress. Defaults to True.

    Raises:
        ValueError: If the mono/dimer tables are missing.
    """
    frames = []
    for fname, length in [("energy_all_metal_onemers.csv", 1), ("energy_all_metal_dimers.csv", 2)]:
        path = os.path.join(raw_dir, fname)
        if os.path.exists(path):
            frames.append(read_raw(path, length, verbose))
        elif verbose:
            print(f"  (skip missing {fname})")
    if not frames:
        raise ValueError(f"no mono/dimer conformer tables found under {raw_dir}")
    md = pd.concat(frames, ignore_index=True)
    md = md.dropna(subset=["amino_acid", "metal"])
    md = md[md["metal"].astype(str) != "0"].reset_index(drop=True)      # metals only
    md = tag_provenance(md)
    write_csv(md[CONFORMER_META_COLS + SPEC_COLS],
              os.path.join(out_dir, f"conformers_monodi{CONFORMER_EXT}"), verbose)
    # mono+di ships uncapped, so simulated == committed for lengths 1 and 2
    stats = {"simulated": {}, "committed": {}}
    for L in (1, 2):
        stats["simulated"][L] = stats["committed"][L] = int((md["length"] == L).sum())

    path = os.path.join(raw_dir, "energy_all_metal_trimers.csv")
    if not os.path.exists(path):
        if verbose:
            print("  (skip missing energy_all_metal_trimers.csv -- no trimer table written)")
        return stats
    tr = read_raw(path, 3, verbose)
    tr = tr.dropna(subset=["amino_acid", "metal"])
    tr = tr[tr["metal"].astype(str) != "0"].reset_index(drop=True)
    # count BEFORE capping: the committed file is capped, so this is the only place the true
    # simulated trimer total exists. It goes into the manifest or it is lost.
    stats["simulated"][3] = len(tr)
    tr = cap_per_molecule(tr, TRIMER_CAP)
    stats["committed"][3] = len(tr)
    tr = tag_provenance(tr)
    write_csv(tr[CONFORMER_META_COLS + SPEC_COLS],
              os.path.join(out_dir, f"conformers_trimers_cap{TRIMER_CAP}{CONFORMER_EXT}"), verbose)
    return stats


def write_manifest(out_dir, molecules, conf_stats, verbose=True):
    """Record what was simulated versus what actually shipped.

    The committed trimer table is capped, so the number of trimer conformers xTB actually
    produced exists nowhere else in the repo once the build finishes -- every consumer would
    otherwise have to either hardcode it or silently under-report the dataset. Anything that
    quotes a dataset size (the README, the graphical overview) reads this file.

    Args:
        out_dir (str): Destination directory, alongside the CSVs it describes.
        molecules (pandas.DataFrame): The molecule table from build_molecules().
        conf_stats (dict): ``{"simulated": {length: n}, "committed": {length: n}}`` from
            build_conformers(); may be None if only the molecule table was built.
        verbose (bool): Print the totals. Defaults to True.

    Returns:
        dict: The manifest that was written.
    """
    metals_only = molecules[molecules["metal"] != "none"]
    sim = {str(k): v for k, v in sorted((conf_stats or {}).get("simulated", {}).items())}
    com = {str(k): v for k, v in sorted((conf_stats or {}).get("committed", {}).items())}
    manifest = {
        "note": "counts are metal complexes only; apo peptides are a baseline, never a class",
        "conformers_simulated": sim,          # post-QC, PRE-cap -- the true xTB output
        "conformers_committed": com,          # what the .csv.gz files actually hold
        "conformers_simulated_total": sum(sim.values()),
        "conformers_committed_total": sum(com.values()),
        "molecules": {str(k): int(v) for k, v in metals_only.groupby("length").size().items()},
        "molecules_table_rows": int(len(molecules)),     # includes apo
        "trimer_cap": TRIMER_CAP,
        "cap_seed": CAP_SEED,
        "spectral_bins": len(SPEC_COLS),
        "float_format": FLOAT_FORMAT,
        "provenance": PROVENANCE,
    }
    path = os.path.join(out_dir, "dataset_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(f"  wrote {os.path.basename(path):34s} {manifest['conformers_simulated_total']:,} "
              f"conformers simulated -> {manifest['conformers_committed_total']:,} committed")
    return manifest


def main():
    """Build every committed CSV from the raw tables, then record the manifest.

    Raises:
        SystemExit: Propagated from argparse on an invalid command line.
        ValueError: If ``--raw-dir`` holds none of the expected raw tables.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=paths.DATA_DIR,
                    help="directory holding the raw energy_all_*.csv tables (default: data/)")
    ap.add_argument("--out-dir", default=paths.PROCESSED_DIR, help="destination (default: data/processed/)")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    print(f"raw tables: {a.raw_dir}\nwriting to: {a.out_dir}\n", flush=True)
    molecules = build_molecules(a.raw_dir, a.out_dir)
    print()
    conf_stats = build_conformers(a.raw_dir, a.out_dir)
    print()
    write_manifest(a.out_dir, molecules, conf_stats)


if __name__ == "__main__":
    main()
