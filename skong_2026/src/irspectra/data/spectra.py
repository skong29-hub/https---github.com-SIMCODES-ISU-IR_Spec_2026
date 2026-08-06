"""The spectral grid and the operations that act on a spectrum array.

Everything here is about the 791-point wavenumber axis itself -- the grid, the IR band
assignments that annotate it, and the stick -> broadened -> normalized transform. No file
I/O and no modelling; those live in data/processed.py and data/conformers.py.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

# ---- spectral grid ----
# Fixed by the raw CSV headers (spec_50 .. spec_4000). Both readers keep only the spec_*
# columns that exist, so editing STEP or the range silently drops bins instead of raising.
STEP = 5
WN = np.arange(50, 4000 + STEP, STEP)          # 791 wavenumbers, 50..4000 cm^-1
SPEC_COLS = [f"spec_{w}" for w in WN]

# metal colours/labels -> irspectra.viz.palette; the class order -> irspectra.config

# IR band regions (cm^-1) -> assignment, for annotating spectra
BANDS = [
    (50, 600, "metal-ligand (M-N, M-O)"),
    (1350, 1450, "carboxylate sym COO-"),
    (1500, 1580, "amide II (N-H + C-N)"),
    (1550, 1650, "carboxylate asym COO-"),
    (1600, 1700, "amide I (C=O)"),
    (2850, 3000, "C-H stretch"),
    (3200, 3600, "N-H / O-H stretch"),
]


def broaden_normalize(X, fwhm=15.0, step=STEP):
    """1C. Stick spectra -> Gaussian broadening (FWHM in cm^-1) -> area = 1.

    Args:
        X (array-like): Stick intensities, shape (n_molecules, 791).
        fwhm (float): Full width at half maximum of the Gaussian kernel, in
            cm^-1. Defaults to 15.0 -- the project-wide value: 3 grid bins, narrow
            enough to keep amide I (1600-1700) off amide II (1500-1580). Changing it
            invalidates data/processed/ and every number in output/.
        step (int): Spacing of the wavenumber grid in cm^-1. Defaults to STEP
            (5).

    Returns:
        numpy.ndarray: Broadened, area-normalized spectra, same shape as ``X``.
        All-zero rows are left as zeros instead of dividing by zero.
    """
    X = np.asarray(X, dtype=float)
    sigma = fwhm / 2.355 / step               # 2.355 = 2*sqrt(2 ln2); /step -> sigma in bins
    B = gaussian_filter1d(X, sigma=sigma, axis=1)
    areas = B.sum(axis=1, keepdims=True)
    # an all-zero row means the frequency calculation produced no modes -- kept as zeros
    # rather than raising, so one bad molecule cannot abort a 9,000-row rebuild
    areas[areas == 0] = 1.0
    return B / areas


def mean_by_length(meta, X, metal_filter=None):
    """Mean spectrum per peptide length (1/2/3), optionally restricted to one metal.

    Args:
        meta (pandas.DataFrame): Molecule table with ``length`` and ``metal``
            columns, aligned row-for-row with ``X``.
        X (numpy.ndarray): Spectra, shape (n_rows, n_wavenumbers).
        metal_filter (str, optional): Keep only rows with this metal label, e.g.
            ``"Cu+2"``. When None, all metals are averaged together. Defaults to
            None.

    Returns:
        dict: ``{length: 1-D numpy.ndarray or None}`` for lengths 1, 2 and 3;
        the value is None where no row matches.
    """
    out = {}
    for L in (1, 2, 3):
        m = (meta["length"].values == L)
        if metal_filter is not None:
            m = m & (meta["metal"].values == metal_filter)
        out[L] = X[m].mean(axis=0) if m.sum() else None
    return out
