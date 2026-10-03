"""E1 -- run the nf-core/drugresponseeval baselines on the same folds as FragDRP.

Includes the four naive predictors, which DAF-DRP omits.  They are not
decoration: under LDO the cell-line-mean predictor is a genuinely strong model,
because knowing a cell line is broadly drug-sensitive predicts a lot without
knowing anything about the molecule.  A drug encoder that does not beat it has
not demonstrated that it learned chemistry.

    python scripts/02_run_baselines.py --dataset GDSC2 --test-mode LDO
    python scripts/02_run_baselines.py --dataset GDSC2 --test-mode LPO
"""

from __future__ import annotations

import logging

import pandas as pd

from _common import CONFIG_ROOT, banner, base_parser, common_overrides, results_dir, setup_logging

LOGGER = logging.getLogger("baselines")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Baselines to run; defaults to fragdrp.baselines.DEFAULT_BASELINES",
    )
    args = parser.parse_args()

    out_dir = results_dir(args, "baselines")
    setup_logging(log_file=out_dir / f"{args.dataset}_{args.test_mode}.log")

    from fragdrp.baselines import DEFAULT_BASELINES, run_baselines
    from fragdrp.config import load_config
    from fragdrp.experiment import ExperimentContext

    cfg = load_config(CONFIG_ROOT / "base.yaml", common_overrides(args))
    banner(f"E1 baselines | {cfg.data.dataset_name} | {cfg.data.test_mode}")

    context = ExperimentContext(cfg)
    LOGGER.info(context.data.summary())
    context.fold_summary().to_csv(
        out_dir / f"folds_{cfg.data.dataset_name}_{cfg.data.test_mode}.csv", index=False
    )

    frame = run_baselines(
        context.data, context.folds, cfg, args.models or DEFAULT_BASELINES
    )
    if frame.empty:
        LOGGER.error("Every baseline failed; nothing written")
        return

    path = out_dir / f"per_fold_{cfg.data.dataset_name}_{cfg.data.test_mode}.csv"
    frame.to_csv(path, index=False)

    summary = (
        frame.groupby("model")[
            ["rmse", "pearson", "spearman", "r2",
             "partial_correlation_drug", "partial_correlation_cell_line"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    LOGGER.info("\n%s", summary.to_string())
    summary.to_csv(out_dir / f"summary_{cfg.data.dataset_name}_{cfg.data.test_mode}.csv")
    LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    main()
