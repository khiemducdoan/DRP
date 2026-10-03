"""E3 -- does the *chemistry* of the cut matter, or just the coarse-graining?

BRICS is the default fragmenter, and the hypothesis says it helps because its
cuts fall on bonds that separate functional groups.  That explanation has a
competitor: any coarse-graining adds a second, smaller graph and a few hundred
thousand parameters, and maybe that is all the fragment branch is worth.

The controls separate the two:

``random_cut``
    The decisive one.  Cuts are placed uniformly at random on acyclic single
    bonds, with the *number* of cuts tuned per molecule so the resulting
    fragment count matches what BRICS produces on that same molecule.  Fragment
    count and size distribution therefore match; only the chemical meaning of
    the cut is destroyed.  If random_cut matches BRICS, the functional-group
    explanation is wrong and the benefit is architectural.
``whole``
    One fragment per molecule: the fragment branch degenerates to a second
    readout.  The lower bound.
``atom``
    One fragment per atom: the fragment graph is the molecular graph again, so
    the branch adds depth but no abstraction.

``recap``, ``murcko`` and ``rotatable_bond`` ask a narrower question: is it
BRICS specifically, or does any chemically-motivated decomposition work?

The script also writes the fragmentation statistics per method, which is the
confound check -- the comparison only means something if the controls really
are matched on fragment count and size.

    python scripts/04_fragmenter_comparison.py --dataset GDSC2 --test-mode LDO
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from _common import CONFIG_ROOT, banner, base_parser, common_overrides, results_dir, setup_logging

LOGGER = logging.getLogger("fragmenters")

METHODS = ["brics", "recap", "murcko", "rotatable_bond", "random_cut", "atom", "whole"]
REFERENCE_METHOD = "frag_brics"


def fragmentation_statistics(smiles: dict, methods) -> pd.DataFrame:
    """Fragment count and size per method -- the confound check for E3."""
    from fragdrp.chem.fragmenters import fragment_smiles
    from fragdrp.config import FragmenterConfig

    rows = []
    for method in methods:
        cfg = FragmenterConfig(method=method)
        if method in ("atom", "whole"):
            cfg.min_fragment_atoms = 1
        if method == "atom":
            cfg.max_fragments = 64
        counts, sizes, singletons = [], [], 0
        for smi in smiles.values():
            frag = fragment_smiles(smi, cfg)
            if frag is None:
                continue
            counts.append(frag.n_fragments)
            sizes.extend(len(f) for f in frag.fragments)
            singletons += sum(1 for f in frag.fragments if len(f) == 1)
        rows.append(
            {
                "method": method,
                "molecules": len(counts),
                "fragments_per_molecule": round(float(np.mean(counts)), 2),
                "fragments_sd": round(float(np.std(counts)), 2),
                "atoms_per_fragment": round(float(np.mean(sizes)), 2),
                "max_fragments": int(np.max(counts)) if counts else 0,
                "singleton_fragments_pct": round(100 * singletons / max(len(sizes), 1), 1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    args = parser.parse_args()

    out_dir = results_dir(args, "fragmenters")
    setup_logging(log_file=out_dir / f"{args.dataset}_{args.test_mode}.log")

    from fragdrp.config import load_config
    from fragdrp.eval.stats import compare_against, summarise_significance
    from fragdrp.experiment import ExperimentContext, collect_summaries, run_experiment
    from fragdrp.train.metrics import primary_metric

    overrides = common_overrides(args)
    banner(f"E3 fragmenter comparison | {args.dataset} | {args.test_mode}")

    first = load_config(CONFIG_ROOT / "fragmenter" / f"{args.methods[0]}.yaml", overrides)
    context = ExperimentContext(first)
    LOGGER.info(context.data.summary())

    stats = fragmentation_statistics(context.data.smiles, args.methods)
    stats.to_csv(out_dir / f"fragment_stats_{args.dataset}.csv", index=False)
    banner("Fragmentation statistics (the confound check)")
    LOGGER.info("\n%s", stats.to_string(index=False))

    results, per_fold = [], []
    for method in args.methods:
        cfg = load_config(CONFIG_ROOT / "fragmenter" / f"{method}.yaml", overrides)
        banner(f"{cfg.name}  (fragmenter={cfg.fragmenter.method})")
        result = run_experiment(cfg, context, save=True, verbose=True)
        results.append(result)
        per_fold.append(result["per_fold"])

    fold_frame = pd.concat(per_fold, ignore_index=True)
    fold_frame.to_csv(out_dir / f"per_fold_{args.dataset}_{args.test_mode}.csv", index=False)

    summary = collect_summaries(results)
    summary.to_csv(out_dir / f"summary_{args.dataset}_{args.test_mode}.csv", index=False)

    metric = primary_metric(args.test_mode)
    LOGGER.info("\n%s", summary[["run", "fragmenter", f"{metric}_mean", f"{metric}_std",
                                 "rmse_mean"]].to_string(index=False))

    comparison = compare_against(
        fold_frame, reference=REFERENCE_METHOD, metric=metric, model_column="run"
    )
    if not comparison.empty:
        comparison.to_csv(
            out_dir / f"significance_{args.dataset}_{args.test_mode}.csv", index=False
        )
        banner(f"Paired comparison vs BRICS on {metric}")
        LOGGER.info("\n%s", comparison.to_string(index=False))
        LOGGER.info("\n%s", summarise_significance(comparison))

        control = comparison[comparison["model"] == "frag_random_cut"]
        if not control.empty:
            row = control.iloc[0]
            banner("THE FALSIFICATION TEST: BRICS vs matched random cuts")
            LOGGER.info(
                "BRICS %s = %.4f, random-cut = %.4f, gap = %+.4f (d=%.2f, p_adj=%.4f)",
                metric, row["reference_mean"], row["model_mean"],
                -row["improvement"], -row["cohens_d"], row["p_adjusted"],
            )
            LOGGER.info(
                "Reading: %s",
                "BRICS is significantly better -- the chemistry of the cut matters."
                if row["significant"] and row["improvement"] < 0
                else "no significant gap -- the benefit is coarse-graining, "
                     "NOT functional-group semantics.",
            )

    LOGGER.info("Wrote fragmenter results to %s", out_dir)


if __name__ == "__main__":
    main()
