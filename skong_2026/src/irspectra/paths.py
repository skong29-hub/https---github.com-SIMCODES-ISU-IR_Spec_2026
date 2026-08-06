"""Project paths for the irspectra package.

The repo root is found once, robustly, so every module and notebook resolves
`data/`, `output/`, and `figures/` the same way no matter the working directory.

Resolution order:
  1. the IRSPECTRA_ROOT environment variable, if it points to a directory;
  2. walk up from this file until a folder holds both pyproject.toml and data/;
  3. fall back to two levels above src/irspectra/ (the repo root in this layout).
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root():
    """Resolve the repository root directory.

    Resolution order is documented once, in the module docstring.

    Returns:
        str: Absolute path to the repository root. Always resolves -- the final
        fallback cannot fail.
    """
    env = os.environ.get("IRSPECTRA_ROOT")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    p = _HERE
    while True:
        if os.path.exists(os.path.join(p, "pyproject.toml")) and os.path.isdir(os.path.join(p, "data")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return os.path.dirname(os.path.dirname(_HERE))   # src/irspectra/ -> src/ -> repo root


ROOT = _find_root()
DATA_DIR = os.path.join(ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TEACHING_DIR = os.path.join(DATA_DIR, "teaching")   # small committed subset for the teaching notebook
OUTPUT_DIR = os.path.join(ROOT, "output")
FIGURES_DIR = os.path.join(ROOT, "figures")


def output_dir(step=None):
    """Path to output/ or output/<step> (e.g. output_dir('step4')).

    Args:
        step (str, optional): Step subdirectory name, e.g. ``"step4"``. When
            None, the top-level ``output/`` directory is returned. Defaults to
            None.

    Returns:
        str: Absolute path to ``output/`` or ``output/<step>``. The directory is
        not created.
    """
    return os.path.join(OUTPUT_DIR, step) if step else OUTPUT_DIR
