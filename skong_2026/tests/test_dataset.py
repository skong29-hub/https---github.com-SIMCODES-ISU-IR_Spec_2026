"""Smoke tests for the committed dataset and its loaders.

Run: pytest -q   (needs data/processed/molecules.csv; rebuild it from the raw tables with
`python tools/build_dataset.py` if absent).
"""
import numpy as np
import pytest

from irspectra.data import conformers
from irspectra.data import processed
from irspectra.data import spectra

EXPECTED_COLS = ["amino_acid", "metal", "length", "provenance", "total_free_energy"]


def _load():
    """Load the processed dataset, skipping the calling test if it is absent.

    Returns:
        tuple: ``(meta, X, sticks)`` as returned by processed.load_processed().

    Raises:
        Skipped: Via ``pytest.skip`` when data/processed/molecules.csv has not been built.
    """
    try:
        return processed.load_processed()
    except FileNotFoundError:
        pytest.skip("data/processed/molecules.csv not present; rebuild with tools/build_dataset.py")


def test_processed_shapes_align():
    """The molecule table, broadened spectra and raw sticks line up row-for-row."""
    meta, X, sticks = _load()
    assert X.shape == (9060, 791)
    assert sticks.shape == X.shape
    assert len(meta) == len(X)
    assert list(meta.columns) == EXPECTED_COLS


def test_metals_and_lengths():
    """Only the five expected metal labels and all three peptide lengths appear."""
    meta, _, _ = _load()
    assert set(meta["metal"]) <= {"none", "Co+2", "Ni+2", "Cu+2", "Zn+2"}
    assert set(meta["length"]) == {1, 2, 3}


def test_spectral_grid():
    """The wavenumber grid is the expected 791 points spanning 50-4000 cm^-1."""
    assert len(spectra.WN) == 791
    assert spectra.WN.min() == 50
    assert spectra.WN.max() == 4000


def test_spectra_are_area_normalized():
    """Every stored broadened spectrum integrates to 1."""
    _, X, _ = _load()
    areas = X.sum(axis=1)
    assert np.allclose(areas, 1.0, atol=1e-5)


def test_teaching_subset_is_loadable_and_aligned():
    """The committed teaching subset loads, aligns row-for-row, and is normalized.

    This one runs on a fresh clone -- data/teaching/ is committed, unlike data/processed/.
    """
    meta, X, sticks = processed.load_teaching()
    assert len(meta) == len(X) == len(sticks)
    assert X.shape[1] == len(spectra.WN)
    assert list(meta.columns) == EXPECTED_COLS
    assert np.allclose(X.sum(axis=1), 1.0, atol=1e-4)      # 6-sig-fig CSV -> looser tolerance


def test_teaching_subset_covers_every_class_and_length():
    """The subset keeps all four metals plus apo, and all three peptide lengths.

    The teaching notebook's Step 5 needs mono+di to train on and trimers to transfer to, so a
    subset missing a length would break it silently.
    """
    meta, _, _ = processed.load_teaching()
    assert set(meta["metal"]) == {"none", "Co+2", "Ni+2", "Cu+2", "Zn+2"}
    assert set(meta["length"]) == {1, 2, 3}
    metals_only = meta[meta["metal"] != "none"]
    assert (metals_only["length"] < 3).sum() > 0           # something to train on
    assert (metals_only["length"] == 3).sum() > 0          # something to transfer to


def test_broaden_normalize_shape_preserved():
    """broaden_normalize() keeps the input shape and returns area-1 spectra."""
    _, _, sticks = _load()
    out = spectra.broaden_normalize(sticks[:5])
    assert out.shape == (5, 791)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_every_table_carries_provenance():
    """All four committed tables agree on the metadata schema, provenance included.

    The column is a constant "xtb" today and nothing branches on it, which makes it exactly the
    kind of thing a later edit drops from one table and not the others. It is the only thing that
    would keep xTB and DFT rows apart if they are ever merged, so a silent schema drift here is
    expensive to discover later.
    """
    for meta, _, _ in (processed.load_processed(), processed.load_teaching()):
        assert list(meta.columns) == EXPECTED_COLS
        assert set(meta["provenance"]) == {"xtb"}

    for lengths in [(1, 2), (3,)]:
        meta, _ = conformers.load_conformers(lengths=lengths, verbose=False)
        assert "provenance" in meta.columns, f"conformer table for {lengths} lost provenance"
        assert set(meta["provenance"]) == {"xtb"}


def test_manifest_matches_the_files_it_describes():
    """The manifest's committed counts equal the rows actually on disk.

    The manifest is the only record of the pre-cap trimer total, so nothing can cross-check it
    from the repo alone — but its *committed* half is checkable, and if that half has drifted the
    simulated half was almost certainly written by the same stale build.
    """
    man = processed.load_manifest()
    assert man["conformers_simulated_total"] >= man["conformers_committed_total"]

    for lengths in [(1, 2), (3,)]:
        meta, _ = conformers.load_conformers(lengths=lengths, verbose=False)
        expected = sum(man["conformers_committed"][str(L)] for L in lengths)
        assert len(meta) == expected, f"manifest says {expected} for {lengths}, file has {len(meta)}"

    meta, _, _ = processed.load_processed()
    assert len(meta) == man["molecules_table_rows"]
    metals_only = meta[meta["metal"] != "none"]
    assert metals_only.groupby("length").size().to_dict() == {
        int(k): v for k, v in man["molecules"].items()}

    # trimers are the only length that ships capped
    assert man["conformers_simulated"]["3"] > man["conformers_committed"]["3"]
    for L in ("1", "2"):
        assert man["conformers_simulated"][L] == man["conformers_committed"][L]
