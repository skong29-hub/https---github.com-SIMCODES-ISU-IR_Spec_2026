"""irspectra: predicting the bound metal of a peptide complex from its simulated IR spectrum.

Layered so the dependencies run one way -- viz and modeling may import data, never the reverse:

  paths          repo-root path resolution (data/, output/, figures/)
  config         the modelling protocol: METAL_ORDER, seeds, RandomForest params, ci95()
  data.spectra   the 791-point grid, the IR band table, broaden/normalize
  data.processed molecule-level loaders (one spectrum per molecule) over the committed CSVs
  data.conformers per-conformer loaders + the conformer -> molecule collapsers
  modeling.protocol shared --cap/--aggregate flags, filename suffixes, the mono+di/trimer loader
  modeling.metrics  macro_f1 / per_metal_f1 with the label set pinned
  viz.palette    metal colours and labels
  viz.panels     the shared confusion-matrix and IR-band panels
  results        load the model-result CSV/JSON tables written into output/
"""
from irspectra import paths, config, results
from irspectra.paths import ROOT, DATA_DIR, PROCESSED_DIR, TEACHING_DIR, OUTPUT_DIR, FIGURES_DIR

__all__ = [
    "paths", "config", "results",
    "ROOT", "DATA_DIR", "PROCESSED_DIR", "TEACHING_DIR", "OUTPUT_DIR", "FIGURES_DIR",
]
__version__ = "1.0.0"
