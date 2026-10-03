"""E4 -- does the fragment prior help most when training drugs are scarce?

This is the experiment that turns the hypothesis into a *prediction* rather
than a post-hoc explanation.  If fragment and functional-group priors work
because they let the model reuse chemistry it has seen in other molecules, then
the advantage over an atom-only encoder should be largest when few training
drugs are available, and should shrink as drug coverage grows -- because with
enough drugs, an atom-level encoder can learn the same regularities from data.

The prediction is therefore a *widening gap as the drug budget shrinks*.  A flat
or inverted gap would be evidence against the mechanism even if the fragment
model wins overall, and the report says so if that is what comes out.

Only the number of training drugs is varied.  The test set is held fixed for a
given fold, so every budget is scored on exactly the same held-out drugs.

    python scripts/06_drug_budget.py --dataset GDSC2 --budgets 0.25 0.5 1.0
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from _common import CONFIG_ROOT, banner, base_parser, common_overrides, results_dir, setup_logging

LOGGER = logging.getLogger("budget")

MODELS = ["a1_atom_only", "a5_fragdrp_full"]
BUDGETS = [0.2, 0.4, 0.7, 1.0]


def subsample_training_drugs(context, fold, fraction: float, seed: int):
    """Keep only ``fraction`` of the fold's training drugs.

    Drugs are dropped rather than rows, because the resource the experiment
    varies is chemical diversity, not the number of measurements.  The held-out
    test drugs are untouched, so budgets remain directly comparable.
    """
    from fragdrp.data.splits import Fold

    if fraction >= 1.0:
        return fold

    data = context.data
    train_drugs = np.unique(data.drug_ids[fold.train])
    rng = np.random.default_rng(seed)
    n_keep = max(2, int(round(fraction * len(train_drugs))))
    keep = set(rng.choice(train_drugs, size=n_keep, replace=False).tolist())

    train_mask = np.array([d in keep for d in data.drug_ids[fold.train]])
    val_mask = np.array([d in keep for d in data.drug_ids[fold.validation]])
    return Fold(
        index=fold.index,
        train=fold.train[train_mask],
        # Validation is restricted too: early stopping must not peek at drugs
        # the budget says are unavailable.
        validation=fold.validation[val_mask] if val_mask.any() else fold.validation,
        test=fold.test,
    )


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--budgets", nargs="+", type=float, default=BUDGETS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--budget-folds", type=int, default=4,
                        help="Folds per budget point; fewer than the main experiments "
                             "because this sweep is a trend, not a significance test")
    args = parser.parse_args()

    out_dir = results_dir(args, "budget")
    setup_logging(log_file=out_dir / f"{args.dataset}_{args.test_mode}.log")

    from fragdrp.config import load_config
    from fragdrp.experiment import ExperimentContext, run_fold
    from fragdrp.train.metrics import primary_metric

    overrides = common_overrides(args)
    banner(f"E4 drug-budget curve | {args.dataset} | {args.test_mode}")

    first = load_config(CONFIG_ROOT / "model" / f"{args.models[0]}.yaml", overrides)
    context = ExperimentContext(first)
    LOGGER.info(context.data.summary())

    rows = []
    for budget in args.budgets:
        for name in args.models:
            cfg = load_config(CONFIG_ROOT / "model" / f"{name}.yaml", overrides)
            cfg.name = f"{cfg.name}_budget{int(budget * 100)}"
            for fold in context.folds[: args.budget_folds]:
                reduced = subsample_training_drugs(context, fold, budget, cfg.train.seed + fold.index)
                n_drugs = int(len(np.unique(context.data.drug_ids[reduced.train])))
                LOGGER.info(
                    "%s | budget %.0f%% (%d drugs, %d rows) | fold %d",
                    cfg.model.name, 100 * budget, n_drugs, len(reduced.train), fold.index,
                )
                outcome = run_fold(cfg, context, reduced, verbose=False)
                rows.append(
                    {
                        "run": cfg.name,
                        "model": cfg.model.name,
                        "budget": budget,
                        "n_train_drugs": n_drugs,
                        "n_train_rows": len(reduced.train),
                        "fold": fold.index,
                        **outcome["metrics"],
                    }
                )
                pd.DataFrame(rows).to_csv(
                    out_dir / f"per_fold_{args.dataset}_{args.test_mode}.csv", index=False
                )

    frame = pd.DataFrame(rows)
    metric = primary_metric(args.test_mode)
    summary = (
        frame.groupby(["model", "budget"])
        .agg(
            n_train_drugs=("n_train_drugs", "mean"),
            **{f"{metric}_mean": (metric, "mean"), f"{metric}_std": (metric, "std"),
               "rmse_mean": ("rmse", "mean"), "pearson_mean": ("pearson", "mean")},
        )
        .reset_index()
    )
    summary.to_csv(out_dir / f"summary_{args.dataset}_{args.test_mode}.csv", index=False)

    banner("Drug-budget curve")
    LOGGER.info("\n%s", summary.to_string(index=False))

    # The prediction under test: the gap should shrink as the budget grows.
    gaps = summary.pivot_table(index="budget", columns="model", values=f"{metric}_mean")
    if gaps.shape[1] == 2:
        left, right = gaps.columns
        gaps["gap"] = gaps[right] - gaps[left]
        LOGGER.info("\nGap (%s minus %s) by budget:\n%s", right, left,
                    gaps["gap"].to_string())
        trend = np.polyfit(gaps.index.to_numpy(dtype=float), gaps["gap"].to_numpy(), 1)[0]
        LOGGER.info(
            "Slope of the gap against budget: %+.4f -- %s",
            trend,
            "consistent with the prior substituting for training drugs (gap narrows "
            "as drugs are added)" if trend < 0 else
            "NOT consistent with that mechanism (the gap does not narrow with more drugs)",
        )
    LOGGER.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
