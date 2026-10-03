"""E1b -- the same models across LPO, LCO and LDO.

The point of this experiment is not to find a winner.  It is to show how much a
headline number depends on the protocol, and therefore how little the protocol
DAF-DRP reports (random pairs) can say about a drug encoder.

Expected, and worth stating in advance so the result is a test rather than a
description:

* under **LPO** every model, including the additive mean-effects baseline,
  scores high, and the spread between drug encoders is small;
* under **LCO** the gap between models is driven by the omics encoder;
* under **LDO** the naive drug-mean predictor collapses to undefined (it has no
  training rows for a held-out drug), scores drop sharply for everyone, and the
  differences between drug encoders finally become visible.

    python scripts/05_protocol_comparison.py --dataset GDSC2
"""

from __future__ import annotations

import logging

import pandas as pd

from _common import CONFIG_ROOT, banner, base_parser, common_overrides, results_dir, setup_logging

LOGGER = logging.getLogger("protocols")

#: Only two models are needed: the proposal and the atom-level reference.
#: Adding more would make the figure busier without changing what it shows.
MODELS = ["a1_atom_only", "a5_fragdrp_full"]
PROTOCOLS = ["LPO", "LCO", "LDO"]


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--protocols", nargs="+", default=PROTOCOLS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--with-baselines", action="store_true",
                        help="Also run the naive drevalpy baselines in each protocol")
    args = parser.parse_args()

    out_dir = results_dir(args, "protocols")
    setup_logging(log_file=out_dir / f"{args.dataset}.log")

    from fragdrp.baselines import run_baselines
    from fragdrp.config import load_config
    from fragdrp.experiment import ExperimentContext, collect_summaries, run_experiment

    rows, fold_rows, baseline_rows = [], [], []
    for protocol in args.protocols:
        overrides = [o for o in common_overrides(args) if not o.startswith("data.test_mode=")]
        overrides.append(f"data.test_mode={protocol}")

        banner(f"E1b protocol {protocol} | {args.dataset}")
        first = load_config(CONFIG_ROOT / "model" / f"{args.models[0]}.yaml", overrides)
        context = ExperimentContext(first)
        LOGGER.info(context.data.summary())

        results = []
        for name in args.models:
            cfg = load_config(CONFIG_ROOT / "model" / f"{name}.yaml", overrides)
            # Keep the run name unique per protocol so results do not overwrite.
            cfg.name = f"{cfg.name}_{protocol}"
            result = run_experiment(cfg, context, save=True, verbose=True)
            results.append(result)
            fold_rows.append(result["per_fold"].assign(protocol=protocol))
        rows.append(collect_summaries(results).assign(protocol=protocol))

        if args.with_baselines:
            frame = run_baselines(
                context.data, context.folds, first,
                ["NaivePredictor", "NaiveDrugMeanPredictor",
                 "NaiveCellLineMeanPredictor", "NaiveMeanEffectsPredictor"],
            )
            if not frame.empty:
                baseline_rows.append(frame.assign(protocol=protocol))

    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(out_dir / f"summary_{args.dataset}.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(
        out_dir / f"per_fold_{args.dataset}.csv", index=False
    )
    if baseline_rows:
        pd.concat(baseline_rows, ignore_index=True).to_csv(
            out_dir / f"baselines_{args.dataset}.csv", index=False
        )

    banner("Protocol comparison")
    columns = ["run", "test_mode", "pearson_mean", "rmse_mean",
               "partial_correlation_drug_mean", "partial_correlation_cell_line_mean"]
    LOGGER.info("\n%s", summary[[c for c in columns if c in summary]].to_string(index=False))
    LOGGER.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
