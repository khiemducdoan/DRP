"""E2 -- the ablation that tests the hypothesis.

The claim under test is: *representing a drug at the level of chemical
fragments and functional groups improves drug response prediction, beyond what
an atom-level graph encoder achieves.*

The ladder isolates one factor per rung, and every rung shares the data,
splits, cell-line encoder, optimiser and schedule, so a difference between two
rungs is that rung's factor and nothing else:

===== ============================================================ ====================
Rung  Model                                                        Adds over previous
===== ============================================================ ====================
A0    cell line only                                               --  (the floor)
A1    + atom-level GIN                                             atom-level structure
A2    fragment view only, no atom message passing                  (side branch)
A3    + fragment graph composed from atoms, no chemical priors      hierarchy
A4    + fragment descriptors and functional-group indicators        chemical priors
A5    + cell-line-conditioned fragment attention                    conditioning
A6    DAF-DRP-style learned fragment vocabulary                     (side branch)
===== ============================================================ ====================

The hypothesis is supported only if **A4 and A5 beat A1 under LDO** with a
significant paired test.  A3 > A1 without A4 > A3 would mean the benefit is
extra depth rather than chemistry; A6 <= A1 would show a lookup-table fragment
representation does not transfer.  All of these outcomes are reported as found.

    python scripts/03_ablation.py --dataset GDSC2 --test-mode LDO --folds 5
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from _common import CONFIG_ROOT, banner, base_parser, common_overrides, results_dir, setup_logging

LOGGER = logging.getLogger("ablation")

#: In ladder order; the report reads this list to lay out its table.
ABLATION_CONFIGS = [
    "a0_cell_only",
    "a1_atom_only",
    "a2_frag_only",
    "a3_atom_frag_bare",
    "a4_atom_frag_priors",
    "a5_fragdrp_full",
    "a6_daf_style_vocab",
]

#: The rung everything is compared against: the atom-level encoder that stands
#: in for the GraphDRP / DeepCDR family.
REFERENCE_RUN = "abl_a1_atom_only"


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--configs", nargs="+", default=ABLATION_CONFIGS)
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse any variant whose per-fold metrics are already on disk. "
             "A sweep is a long sequence of independent trainings; one failure "
             "part-way through should not cost the runs that already succeeded.",
    )
    args = parser.parse_args()

    out_dir = results_dir(args, "ablation")
    setup_logging(log_file=out_dir / f"{args.dataset}_{args.test_mode}.log")

    from fragdrp.config import load_config
    from fragdrp.eval.stats import compare_against, summarise_significance
    from fragdrp.eval.vocabulary import summarise_coverage, vocabulary_coverage
    from fragdrp.experiment import ExperimentContext, collect_summaries, run_experiment
    from fragdrp.train.metrics import primary_metric

    overrides = common_overrides(args)
    banner(f"E2 ablation | {args.dataset} | {args.test_mode} | {len(args.configs)} variants")

    # One context for every variant: identical data, identical folds.
    first = load_config(CONFIG_ROOT / "model" / f"{args.configs[0]}.yaml", overrides)
    context = ExperimentContext(first)
    LOGGER.info(context.data.summary())

    # Quantify the claim behind A6 before training anything: how much of a
    # held-out drug can a training-only fragment vocabulary even represent?
    if args.test_mode == "LDO":
        coverage = vocabulary_coverage(
            context.data, context.folds, first.fragmenter, first.featurizer.vocab_size
        )
        coverage.to_csv(
            out_dir / f"vocabulary_coverage_{args.dataset}_{args.test_mode}.csv", index=False
        )
        banner("Fragment-vocabulary coverage of held-out drugs")
        LOGGER.info("\n%s", coverage.to_string(index=False))
        LOGGER.info(summarise_coverage(coverage))

    results, per_fold = [], []
    for name in args.configs:
        cfg = load_config(CONFIG_ROOT / "model" / f"{name}.yaml", overrides)
        cached = cfg.run_dir / "per_fold_metrics.csv"

        if args.resume and cached.exists():
            frame = pd.read_csv(cached)
            banner(f"{cfg.name}  (reusing {len(frame)} folds from {cached})")
            from fragdrp.experiment import aggregate_folds

            results.append({"config": cfg, "per_fold": frame,
                            "summary": aggregate_folds(frame, cfg)})
            per_fold.append(frame)
            continue

        banner(f"{cfg.name}  ({cfg.model.name})")
        try:
            result = run_experiment(cfg, context, save=True, verbose=True)
        except Exception as error:  # noqa: BLE001
            # One variant failing must not destroy the comparison table built
            # from the variants that worked; the report marks it as missing.
            LOGGER.exception("%s failed and will be missing from the table: %s",
                             cfg.name, error)
            continue
        results.append(result)
        per_fold.append(result["per_fold"])

    if not per_fold:
        LOGGER.error("Every variant failed; nothing to summarise")
        return

    fold_frame = pd.concat(per_fold, ignore_index=True)
    fold_frame.to_csv(out_dir / f"per_fold_{args.dataset}_{args.test_mode}.csv", index=False)

    summary = collect_summaries(results)
    summary.to_csv(out_dir / f"summary_{args.dataset}_{args.test_mode}.csv", index=False)

    metric = primary_metric(args.test_mode)
    LOGGER.info("\n%s", summary[["run", "model", f"{metric}_mean", f"{metric}_std",
                                 "rmse_mean", "pearson_mean"]].to_string(index=False))

    # The decisive table: every rung against the atom-only reference.
    for test_metric in (metric, "rmse", "pearson"):
        comparison = compare_against(
            fold_frame, reference=REFERENCE_RUN, metric=test_metric, model_column="run"
        )
        if comparison.empty:
            continue
        comparison.to_csv(
            out_dir / f"significance_{test_metric}_{args.dataset}_{args.test_mode}.csv",
            index=False,
        )
        banner(f"Paired comparison vs {REFERENCE_RUN} on {test_metric}")
        LOGGER.info("\n%s", comparison.to_string(index=False))
        LOGGER.info("\n%s", summarise_significance(comparison))

    LOGGER.info("Wrote ablation results to %s", out_dir)


if __name__ == "__main__":
    main()
