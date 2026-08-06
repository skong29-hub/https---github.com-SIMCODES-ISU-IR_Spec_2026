"""The CLI flags and data loading the step-5 and step-6 scripts share verbatim.

Five scripts accept ``--cap``, three accept ``--aggregate``, and five rebuilt the same output-filename
suffix under three different local variable names, while three repeated the same eight-line
"load mono+di and trimers, optionally collapse to one spectrum per molecule" block. All of that
lives here now, so a flag and the filename it produces cannot drift apart.

Deliberately NOT here: the per-seed split-fit-score loop. The four scripts that run one differ in
what they score and on which test set, and a shared version would need enough flags to be harder
to read than the duplication.

    ap = argparse.ArgumentParser()
    add_cap_flag(ap); add_aggregate_flag(ap)
    a = ap.parse_args()
    sfx = suffix_for(a)
    meta_md, X_md, meta_tr, X_tr = load_monodi_and_trimers(a)
"""
from irspectra.data import conformers

CAP_HELP = "cap conformers/molecule (e.g. 10) to balance per-metal counts; appends _cap{K}"
AGGREGATE_HELP = ("none = per-conformer (each conformer a sample); "
                  "mean = one mean-averaged spectrum per molecule")


def add_cap_flag(parser):
    """Add the shared ``--cap`` flag.

    Args:
        parser (argparse.ArgumentParser): Parser to add the flag to, modified in place.
    """
    parser.add_argument("--cap", type=int, default=None, help=CAP_HELP)


def add_aggregate_flag(parser):
    """Add the shared ``--aggregate`` flag.

    Args:
        parser (argparse.ArgumentParser): Parser to add the flag to, modified in place.
    """
    parser.add_argument("--aggregate", choices=["none", "mean"], default="none", help=AGGREGATE_HELP)


def suffix_for(args):
    """Output-filename suffix implied by the parsed flags.

    The committed artifacts in output/ all carry ``_cap10``; a run with no flags writes an
    unsuffixed file instead, which is why run_all.py passes ``--cap 10`` explicitly.

    Args:
        args (argparse.Namespace): Parsed flags. Missing ``cap`` / ``aggregate`` count as unset.

    Returns:
        str: ``""``, ``"_meanavg"``, ``"_cap10"``, or ``"_meanavg_cap10"``.
    """
    suffix = "" if getattr(args, "aggregate", "none") == "none" else "_meanavg"
    cap = getattr(args, "cap", None)
    return f"{suffix}_cap{cap}" if cap else suffix


def representation(args):
    """Human-readable name of the representation the flags select, for log lines.

    Args:
        args (argparse.Namespace): Parsed flags.

    Returns:
        str: ``"per-conformer"`` or ``"mean-averaged"``.
    """
    return "per-conformer" if getattr(args, "aggregate", "none") == "none" else "mean-averaged"


def load_monodi_and_trimers(args, verbose=True):
    """Load the mono+dimer training set and the trimer transfer set, honouring the flags.

    Args:
        args (argparse.Namespace): Parsed flags; ``cap`` and ``aggregate`` are read if present.
        verbose (bool): Forwarded to the loaders. Defaults to True.

    Returns:
        tuple: ``(meta_md, X_md, meta_tr, X_tr)`` -- mono+dimer then trimer, each a
        (pandas.DataFrame, numpy.ndarray) pair aligned row-for-row.

    Raises:
        ValueError: Propagated from load_conformers() if the conformer CSVs are missing.
    """
    cap = getattr(args, "cap", None)
    meta_md, X_md = conformers.load_conformers(lengths=(1, 2), cap=cap, verbose=verbose)
    meta_tr, X_tr = conformers.load_conformers(lengths=(3,), cap=cap, verbose=verbose)
    if getattr(args, "aggregate", "none") == "mean":
        meta_md, X_md = conformers.mean_average(meta_md, X_md, verbose=verbose)
        meta_tr, X_tr = conformers.mean_average(meta_tr, X_tr, verbose=verbose)
    return meta_md, X_md, meta_tr, X_tr
