"""E5 -- what does the model actually attend to, and does it matter?

Interpretability claims are cheap.  This script tries to make one that can
fail, in three steps of increasing strength:

1. **Description.** Which functional groups receive more attention than uniform?
2. **Conditioning.** Does a drug's attention pattern change across cell lines,
   or has it collapsed into a cell-line-independent saliency map?  Both are
   reportable; they support different claims.
3. **Intervention (the falsifiable part).** Mask the most-attended fragment and
   measure how much the prediction moves, against masking a random fragment of
   the same molecule.  If top-masking does not move predictions more than
   random-masking, the attention is decorative and the attribution story does
   not hold, whatever the enrichment table looks like.

    python scripts/07_attribution.py --dataset GDSC2 --test-mode LDO
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from _common import CONFIG_ROOT, banner, base_parser, common_overrides, results_dir, setup_logging

LOGGER = logging.getLogger("attribution")


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--config", default="a5_fragdrp_full",
                        help="Model config; must have cell-conditioned attention on")
    parser.add_argument("--attribution-folds", type=int, default=2,
                        help="Folds to attribute over; attention is stable, so a "
                             "couple of folds is enough and keeps this cheap")
    parser.add_argument("--max-batches", type=int, default=40)
    args = parser.parse_args()

    out_dir = results_dir(args, "attribution")
    setup_logging(log_file=out_dir / f"{args.dataset}_{args.test_mode}.log")

    from fragdrp.config import load_config
    from fragdrp.data.dataset import DrugResponseTorchDataset, fit_scalers
    from fragdrp.data.loader import drug_names
    from fragdrp.eval.attribution import (
        attention_ablation_curve,
        attention_specificity,
        fragment_attention_table,
        functional_group_importance,
        summarise_ablation,
        top_fragments_per_drug,
    )
    from fragdrp.experiment import ExperimentContext, run_fold
    from fragdrp.train.trainer import resolve_device

    cfg = load_config(CONFIG_ROOT / "model" / f"{args.config}.yaml", common_overrides(args))
    if not cfg.model.use_cell_conditioned_attention:
        raise SystemExit(f"{args.config} has no cell-conditioned attention to attribute")

    banner(f"E5 attribution | {args.dataset} | {args.test_mode}")
    context = ExperimentContext(cfg)
    names = drug_names(cfg.data.path_data, cfg.data.dataset_name)
    device = resolve_device(cfg.train.device)

    attention_frames, ablation_frames = [], []
    for fold in context.folds[: args.attribution_folds]:
        LOGGER.info("Training fold %d for attribution", fold.index)
        outcome = run_fold(cfg, context, fold, verbose=False)
        LOGGER.info("fold %d test metrics: %s", fold.index, outcome["metrics"])

        graphs = context.drug_graphs(cfg)
        test_set = DrugResponseTorchDataset(
            context.data, fold.test, graphs,
            outcome["cell_scaler"], outcome["response_scaler"],
        )

        frame = fragment_attention_table(
            outcome["model"], test_set, device=device, max_batches=args.max_batches
        )
        if frame.empty:
            LOGGER.warning("No attention recovered on fold %d", fold.index)
            continue
        attention_frames.append(frame.assign(fold=fold.index))

        ablation_frames.append(
            attention_ablation_curve(
                outcome["model"], test_set, device=device, seed=cfg.train.seed + fold.index
            ).assign(fold=fold.index)
        )

    if not attention_frames:
        raise SystemExit("Nothing to attribute")

    attention = pd.concat(attention_frames, ignore_index=True)
    attention["drug_name"] = attention["drug_id"].map(names).fillna(attention["drug_id"])
    attention.to_csv(out_dir / f"fragment_attention_{args.dataset}.csv", index=False)

    # --- 1. description ---------------------------------------------------- #
    enrichment = functional_group_importance(attention)
    enrichment.to_csv(out_dir / f"functional_group_enrichment_{args.dataset}.csv", index=False)
    banner("Functional groups by attention enrichment (1.0 = uniform)")
    LOGGER.info("\n%s", enrichment.head(20).to_string(index=False))

    top = top_fragments_per_drug(attention, top_k=1)
    top["drug_name"] = top["drug_id"].map(names).fillna(top["drug_id"])
    top.to_csv(out_dir / f"top_fragment_per_drug_{args.dataset}.csv", index=False)
    banner("Most-attended fragment per held-out drug (first 25)")
    LOGGER.info("\n%s", top[["drug_name", "fragment_smiles", "functional_groups",
                             "mean_attention"]].head(25).to_string(index=False))

    # --- 2. conditioning --------------------------------------------------- #
    specificity = attention_specificity(attention)
    specificity.to_csv(out_dir / f"attention_specificity_{args.dataset}.csv", index=False)
    if not specificity.empty:
        ratio = float(specificity["conditioning_ratio"].median())
        banner("Is the attention actually conditioned on the cell line?")
        LOGGER.info(
            "Median ratio of between-cell-line spread to between-fragment spread: %.3f", ratio
        )
        LOGGER.info(
            "Reading: %s",
            "the pattern shifts meaningfully with the cell line"
            if ratio > 0.25 else
            "the pattern is close to cell-line-independent -- a drug saliency map, "
            "not a drug x cell-line interaction",
        )

    # --- 3. intervention --------------------------------------------------- #
    summary = {}
    if ablation_frames:
        curve = pd.concat(ablation_frames, ignore_index=True)
        curve.to_csv(out_dir / f"attention_masking_{args.dataset}.csv", index=False)
        summary = summarise_ablation(curve)
        banner("Masking test: top-attended fragment vs a random fragment")
        LOGGER.info("\n%s", json.dumps(summary, indent=2))
        LOGGER.info(
            "Reading: %s",
            "masking the attended fragment moves predictions more than masking a "
            "random one -- the attention marks fragments the prediction depends on."
            if summary.get("mean_difference", 0) > 0 and summary.get("p_wilcoxon", 1) < 0.05
            else "no reliable difference -- the attention weights do not identify "
                 "fragments the prediction depends on.",
        )
        (out_dir / f"masking_summary_{args.dataset}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    LOGGER.info("Wrote attribution results to %s", out_dir)


if __name__ == "__main__":
    main()
