"""Load the model-result tables written by the step scripts into output/.

This is the bridge between the compute layer (pipeline/) and the figure notebook:
the scripts write CSV/JSON metrics into output/step{2..6}/, and the notebook reads
them back with these helpers instead of recomputing anything.

    from irspectra import results
    cv   = results.load_csv("step4", "cv_results_macrof1.csv")
    cm   = results.load_confusion("step4", "confusion_perconformer_cap10.csv")
    summ = results.load_json("step5", "zeroshot_summary_cap10.json")

Every function here takes the same two arguments, documented once:
    step (str)   subdirectory under output/, e.g. "step4"
    name (str)   result filename, e.g. "cv_results_macrof1.csv"
and every loader raises FileNotFoundError if that step has not been run yet. Use
exists() to check first when a missing result should be a skip rather than an error.
"""
import os
import json

import pandas as pd

from irspectra.paths import OUTPUT_DIR


def path(step, name):
    """Absolute path to output/<step>/<name>. Neither the directory nor the file need exist.

    Returns:
        str: The absolute path.
    """
    return os.path.join(OUTPUT_DIR, step, name)


def exists(step, name):
    """Whether output/<step>/<name> is present on disk.

    Returns:
        bool: True if the result file exists.
    """
    return os.path.exists(path(step, name))


def load_csv(step, name, **kwargs):
    """Read a result CSV as a DataFrame.

    Args:
        **kwargs: Forwarded verbatim to ``pandas.read_csv``.

    Returns:
        pandas.DataFrame: The parsed result table.
    """
    return pd.read_csv(path(step, name), **kwargs)


def load_json(step, name):
    """Read a result JSON summary.

    Returns:
        dict: The decoded summary.

    Raises:
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path(step, name)) as f:
        return json.load(f)


def load_confusion(step, name):
    """Read a confusion-matrix CSV, metal labels in the first column.

    Returns:
        pandas.DataFrame: Square matrix indexed and columned by metal label; rows are
        row-normalized, so the diagonal is per-metal recall.
    """
    return pd.read_csv(path(step, name), index_col=0)


def list_step(step):
    """Every result filename present in output/<step>/.

    Returns:
        list[str]: Sorted filenames, or an empty list if the directory does not exist.
    """
    d = os.path.join(OUTPUT_DIR, step)
    return sorted(os.listdir(d)) if os.path.isdir(d) else []
