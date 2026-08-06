"""Molecule-level data: one lowest-free-energy spectrum per molecule.

Reads the committed CSVs and nothing else. The raw xTB tables are hundreds of MB and are not
in git; `tools/build_dataset.py` turns them into these CSVs, and that is the only code in the
repo that ever opens one. So every pipeline script and both notebooks run on a fresh clone.

The CSVs store RAW STICK intensities in `spec_50 .. spec_4000`, the same layout as the source
tables. Broadening happens here at read time, which keeps FWHM a knob rather than a property
baked into the file.
"""
import os
import json

import pandas as pd

from irspectra.paths import PROCESSED_DIR, TEACHING_DIR
from irspectra.data.spectra import SPEC_COLS, broaden_normalize

META_COLS = ["amino_acid", "metal", "length", "provenance", "total_free_energy"]


def load_spectra_csv(path, fwhm=15.0, meta_cols=None):
    """Split one dataset CSV into its metadata table and its broadened + raw spectra.

    The shared reader behind every loader here and in data/conformers.py.

    Args:
        path (str): Absolute path to a CSV with metadata columns then ``spec_*`` sticks.
        fwhm (float): Gaussian broadening FWHM in cm^-1. Defaults to 15.0.
        meta_cols (list, optional): Metadata columns to keep. Defaults to every non-``spec_*``
            column, in file order.

    Returns:
        tuple: ``(meta, X, sticks)`` aligned row-for-row -- the metadata (pandas.DataFrame), the
        broadened area-normalized spectra and the raw sticks (both numpy.ndarray).

    Raises:
        FileNotFoundError: If the CSV has not been built yet.
    """
    df = pd.read_csv(path)
    if meta_cols is None:
        meta_cols = [c for c in df.columns if not c.startswith("spec_")]
    sticks = df[SPEC_COLS].to_numpy(float)
    return df[meta_cols].copy(), broaden_normalize(sticks, fwhm=fwhm), sticks


def load_manifest(out_dir=PROCESSED_DIR):
    """Read data/processed/dataset_manifest.json -- what was simulated vs what shipped.

    The committed trimer table is capped, so the true simulated trimer count exists only here.
    Anything that quotes a dataset size should read this rather than counting rows, which would
    silently under-report by the capped conformers.

    Args:
        out_dir (str): Directory holding dataset_manifest.json. Defaults to PROCESSED_DIR.

    Returns:
        dict: ``conformers_simulated`` / ``conformers_committed`` keyed by peptide length (as
        strings), their totals, ``molecules`` per length, and the build settings that produced
        the CSVs (trimer_cap, cap_seed, spectral_bins, float_format, provenance).

    Raises:
        FileNotFoundError: If the dataset has not been built by tools/build_dataset.py.
    """
    with open(os.path.join(out_dir, "dataset_manifest.json")) as f:
        return json.load(f)


def load_processed(out_dir=PROCESSED_DIR, fwhm=15.0):
    """Load the molecule-level dataset: one spectrum per molecule.

    Args:
        out_dir (str): Directory holding molecules.csv. Defaults to PROCESSED_DIR.
        fwhm (float): Gaussian broadening FWHM in cm^-1. Defaults to 15.0.

    Returns:
        tuple: ``(meta, X, sticks)`` aligned row-for-row. ``meta`` carries amino_acid, metal,
        length, provenance and total_free_energy; ``X`` the broadened area-normalized spectra,
        shape (n_molecules, 791); ``sticks`` the raw stick intensities.

    Raises:
        FileNotFoundError: If data/processed/molecules.csv is absent; build it with
            ``python tools/build_dataset.py``.
    """
    return load_spectra_csv(os.path.join(out_dir, "molecules.csv"), fwhm, META_COLS)


def load_teaching(teaching_dir=TEACHING_DIR, fwhm=15.0):
    """Load the small committed subset used by notebooks/teaching_notebook.ipynb.

    Same columns and units as load_processed(), but only a few hundred molecules, stratified
    over (length, metal). Built by tools/make_teaching_subset.py.

    Args:
        teaching_dir (str): Directory holding molecules.csv. Defaults to TEACHING_DIR.
        fwhm (float): Gaussian broadening FWHM in cm^-1. Defaults to 15.0.

    Returns:
        tuple: ``(meta, X, sticks)`` aligned row-for-row.

    Raises:
        FileNotFoundError: If the subset has not been built; run
            ``python tools/make_teaching_subset.py``.
    """
    return load_spectra_csv(os.path.join(teaching_dir, "molecules.csv"), fwhm, META_COLS)
