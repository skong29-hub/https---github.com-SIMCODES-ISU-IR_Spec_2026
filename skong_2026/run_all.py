#!/usr/bin/env python3
"""
run_all.py -- the whole pipeline, in the right order, with the flags that reproduce output/.

This is the executable version of the README's regeneration recipe. It exists because the step
order is not obvious from the filenames (step3_pca_means.py only re-plots what step3_pca_coords.py
computes) and because several steps need `--cap 10` to write the artifacts the notebook and the
tests actually read -- run them bare and you get differently-named, uncapped results instead.

    python run_all.py --dry-run          # print the plan and exit, changes nothing
    python run_all.py                    # run everything (hours; needs the raw tables in data/)
    python run_all.py --from step4       # resume partway through
    python run_all.py --only step5       # just one stage
    python run_all.py --list             # one line per stage

Only the `data` stage needs the raw xTB tables (see data/README.md). Every other stage reads the
committed CSVs in data/processed/, so on a fresh clone you can start at `--from teaching`.
"""
import os
import sys
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

# (stage, description, argv). Stages run in this order; within a stage, commands run in order.
STAGES = [
    # the ONLY stage that reads raw xTB tables; skip it unless they changed, since the CSVs
    # it writes are committed and every later stage reads only those
    ("data", "Rebuild the committed CSV dataset from the raw xTB tables", [
        [sys.executable, "tools/build_dataset.py"],
    ]),
    ("teaching", "Refresh the committed teaching subset from data/processed/", [
        [sys.executable, "tools/make_teaching_subset.py"],
    ]),
    ("step2", "Tune the RandomForest (writes best_params.json that every later step reads)", [
        [sys.executable, "tools/tune_rf_bo.py"],
    ]),
    # only step3_pca_means re-plots the saved coords; step3_pca_loadings refits its own PCA
    ("step3", "PCA: fit the coordinates first, then the means view that re-plots them", [
        [sys.executable, "pipeline/step3_pca_coords.py"],
        [sys.executable, "pipeline/step3_pca_means.py"],
        [sys.executable, "pipeline/step3_pca_loadings.py"],
    ]),
    ("step4", "Cross-validation (uncapped, the 0.81 headline) and the K=10 confusion matrix", [
        [sys.executable, "pipeline/step4_cv_macrof1.py"],
        [sys.executable, "pipeline/step4_confusion_perconformer.py"],
    ]),
    ("step5", "Zero-shot length transfer, plus both trimer confusion matrices", [
        [sys.executable, "pipeline/step5_zeroshot.py", "--cap", "10"],
        [sys.executable, "pipeline/step5_confusion_part.py", "--cap", "10"],
        [sys.executable, "pipeline/step5_confusion_whole.py", "--cap", "10"],
    ]),
    ("step6", "Data-value frontier (Fig C settings) and the 2x2 length-substitution study", [
        [sys.executable, "pipeline/step6_frontier.py", "--cap", "10",
         "--dup-mode", "resample", "--test-mode", "per_seed"],
        [sys.executable, "pipeline/step6_length_substitution.py", "--cap", "10"],
    ]),
    ("tests", "Smoke tests over the committed tables", [
        [sys.executable, "-m", "pytest", "-q"],
    ]),
]
STAGE_NAMES = [s[0] for s in STAGES]


def show(cmd):
    """Render one argv list as a copy-pasteable command line.

    Args:
        cmd (list): The argv list, whose first element is the interpreter path.

    Returns:
        str: The command with the absolute interpreter path shortened to ``python``.
    """
    return " ".join(["python"] + [c for c in cmd[1:]])


def select(args):
    """Resolve --only / --from / --skip into the ordered list of stages to run.

    Args:
        args (argparse.Namespace): Parsed command line.

    Returns:
        list: The selected ``(stage, description, commands)`` tuples, in pipeline order.

    Raises:
        SystemExit: If a named stage does not exist.
    """
    for name in filter(None, [args.only, getattr(args, "from")] + args.skip):
        if name not in STAGE_NAMES:
            raise SystemExit(f"unknown stage {name!r}; choose from {', '.join(STAGE_NAMES)}")
    chosen = STAGES
    if args.only:
        chosen = [s for s in chosen if s[0] == args.only]
    elif getattr(args, "from"):
        chosen = chosen[STAGE_NAMES.index(getattr(args, "from")):]
    return [s for s in chosen if s[0] not in args.skip]


def main():
    """Run (or print) the pipeline stages in dependency order.

    Stops at the first failing command, so a broken step cannot silently leave a stale artifact
    in place while later steps read it.

    Raises:
        SystemExit: With the failing command's exit code, or 2 for an unknown stage name.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--list", action="store_true", help="list the stages and exit")
    ap.add_argument("--only", metavar="STAGE", help=f"run one stage ({', '.join(STAGE_NAMES)})")
    ap.add_argument("--from", metavar="STAGE", help="start at this stage and run the rest")
    ap.add_argument("--skip", metavar="STAGE", action="append", default=[], help="skip a stage (repeatable)")
    args = ap.parse_args()

    if args.list:
        for name, desc, cmds in STAGES:
            print(f"  {name:9s} {desc}  ({len(cmds)} command{'s' if len(cmds) > 1 else ''})")
        return

    stages = select(args)
    total = sum(len(c) for _, _, c in stages)
    print(f"{'DRY RUN: ' if args.dry_run else ''}{len(stages)} stage(s), {total} command(s)\n")

    n = 0
    for name, desc, cmds in stages:
        print(f"=== {name} - {desc}")          # ASCII only: Windows consoles default to cp1252
        for cmd in cmds:
            n += 1
            print(f"  [{n}/{total}] {show(cmd)}", flush=True)
            if args.dry_run:
                continue
            started = time.time()
            result = subprocess.run(cmd, cwd=HERE)
            if result.returncode != 0:
                print(f"\nFAILED ({result.returncode}) in stage {name}: {show(cmd)}")
                raise SystemExit(result.returncode)
            print(f"        done in {time.time() - started:.1f}s", flush=True)
        print()

    print("dry run complete - nothing was written" if args.dry_run else "pipeline complete")


if __name__ == "__main__":
    main()
