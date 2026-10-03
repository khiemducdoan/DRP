"""Run all six fusion configurations, as reported in Tables 2-3 of the paper.

    python run_all_fusions.py --dataset GDSC2 --test-mode LDO
    python run_all_fusions.py --dataset GDSC2 --test-mode LPO   # the paper's protocol

Produces one table with CC, WC, CA, WA, AC, AA side by side, plus paired
Wilcoxon tests between them.  The paper picks its winner by ordering the means;
with six configurations on the same folds that is a multiple-comparison problem,
so the ordering is reported *and* tested.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from model import FUSION_CONFIGS  # noqa: E402

LOGGER = logging.getLogger("fusions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="GDSC2")
    parser.add_argument("--test-mode", default="LDO")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--configs", nargs="+", default=FUSION_CONFIGS)
    parser.add_argument("--fix-attention", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)

    results_dir = HERE / "results"
    for code in args.configs:
        command = [
            sys.executable, str(HERE / "train.py"),
            "--drug-fusion", code[0], "--global-fusion", code[1],
            "--dataset", args.dataset, "--test-mode", args.test_mode,
        ]
        if args.folds:
            command += ["--folds", str(args.folds)]
        if args.fix_attention:
            command.append("--fix-attention")
        if args.quick:
            command.append("--quick")
        for override in args.overrides:
            command += ["--set", override]

        LOGGER.info("=" * 74)
        LOGGER.info("fusion %s  (DrugFusion=%s, GlobalFusion=%s)", code, code[0], code[1])
        LOGGER.info("=" * 74)
        outcome = subprocess.run(command, cwd=HERE)
        if outcome.returncode != 0:
            LOGGER.error("config %s failed (exit %d); continuing", code, outcome.returncode)

    # --- collect ---------------------------------------------------------- #
    frames: List[pd.DataFrame] = []
    for code in args.configs:
        path = results_dir / f"dafbrics_{code}_{args.dataset}_{args.test_mode}" / "per_fold_metrics.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        LOGGER.error("No results to summarise")
        return

    per_fold = pd.concat(frames, ignore_index=True)
    out = results_dir / f"all_fusions_{args.dataset}_{args.test_mode}"
    out.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(out / "per_fold_metrics.csv", index=False)

    from fragdrp.eval.stats import compare_against
    from fragdrp.train.metrics import primary_metric

    metric = primary_metric(args.test_mode)
    columns = ["rmse", "pearson", "r2", "partial_correlation_drug",
               "partial_correlation_cell_line"]
    summary = per_fold.groupby("fusion")[columns].agg(["mean", "std"]).round(4)
    summary.to_csv(out / "summary.csv")

    LOGGER.info("=" * 74)
    LOGGER.info("DAF-DRP (BRICS) — all fusion configurations | %s %s",
                args.dataset, args.test_mode)
    LOGGER.info("=" * 74)
    LOGGER.info("\n%s", summary.to_string())

    # Is the best configuration actually distinguishable from the others?
    best = per_fold.groupby("fusion")[metric].mean().idxmax()
    comparison = compare_against(per_fold, reference=best, metric=metric,
                                 model_column="fusion")
    if not comparison.empty:
        comparison.to_csv(out / f"significance_{metric}.csv", index=False)
        LOGGER.info("\nPaired comparison against the best configuration (%s) on %s:", best, metric)
        LOGGER.info("\n%s", comparison[
            ["model", "model_mean", "reference_mean", "improvement", "cohens_d",
             "p_wilcoxon", "p_adjusted", "significant"]
        ].round(4).to_string(index=False))
        n_sig = int(comparison["significant"].sum())
        LOGGER.info(
            "\n%d of %d configurations differ significantly from %s. %s",
            n_sig, len(comparison), best,
            "The choice of fusion operator matters." if n_sig else
            "No configuration is distinguishable from another, so an ordering of "
            "the means is not evidence that one fusion strategy is better.",
        )
    LOGGER.info("Wrote %s", out)


if __name__ == "__main__":
    main()
