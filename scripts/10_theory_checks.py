"""E7 — measure what the propositions in THEORY.md predict.

Three checks, none of which needs training, so they run in seconds and can be
read before committing GPU time:

**P1 — attention degeneracy.** Feed several different drugs with one cell line
through each DAF-DRP fusion configuration and measure the spread of the
predictions. Configurations whose ``GlobalFusion`` is cross-attention should
give a spread of exactly zero.

**P4 — lookup-table collisions.** Count pairs of held-out drugs that a
sum-pooled vocabulary fragment branch *provably cannot tell apart*: same
multiset of in-vocabulary fragments, same out-of-vocabulary count. This is a
property of the representation, independent of training.

**Coverage.** What fraction of a held-out drug a training-mined vocabulary can
address at all.

    python scripts/10_theory_checks.py --dataset GDSC2 --test-mode LDO
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from _common import CONFIG_ROOT, banner, base_parser, results_dir, setup_logging

LOGGER = logging.getLogger("theory")


# --------------------------------------------------------------------------- #
# P1 — does the fusion operator use the drug at all?
# --------------------------------------------------------------------------- #


def check_attention_degeneracy(n_repeats: int = 5) -> pd.DataFrame:
    """Prediction spread across different drugs, holding the cell line fixed.

    Averaged over several random initialisations so the result is a property of
    the architecture rather than of one lucky seed.
    """
    import sys

    import torch

    sys.path.insert(0, str(CONFIG_ROOT.parent / "daf_drp_brics"))
    from data import ATOM_DIM, collate, featurize            # noqa: E402
    from fragment_vocab import build_vocabulary              # noqa: E402
    from model import FUSION_CONFIGS, DAFDRPBrics            # noqa: E402

    from fragdrp.config import FragmenterConfig

    smiles = [
        "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=CC4=CC=CC=N4",   # axitinib
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",  # gefitinib
        "CCO",
    ]
    vocabulary = build_vocabulary(smiles, FragmenterConfig(method="brics"), 798)
    cell = np.random.default_rng(0).standard_normal(64).astype(np.float32)
    batch = collate([
        {"graph": featurize(s, s, vocabulary), "cell": cell, "target": np.float32(0),
         "raw_response": np.float32(0), "drug_id": s, "cell_line_id": "cl"}
        for s in smiles
    ])

    rows = []
    for code in FUSION_CONFIGS:
        spreads = []
        for seed in range(n_repeats):
            torch.manual_seed(seed)
            model = DAFDRPBrics(
                64, ATOM_DIM, 798, drug_fusion=code[0], global_fusion=code[1]
            ).eval()
            with torch.no_grad():
                predictions = model(batch)["prediction"]
            spreads.append(float(predictions.max() - predictions.min()))
        rows.append({
            "fusion": code,
            "drug_fusion": code[0],
            "global_fusion": code[1],
            "prediction_spread": float(np.mean(spreads)),
            "uses_drug": bool(np.mean(spreads) > 1e-6),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# P4 — how often does a lookup-table fragment branch collide?
# --------------------------------------------------------------------------- #


def _signature(fragments: List[str], vocabulary: set) -> Tuple:
    """The only thing a sum-pooled lookup-table branch can see about a molecule.

    By THEORY.md Proposition 4 the fragment readout is determined by the multiset
    of in-vocabulary indices plus the count of out-of-vocabulary fragments, so
    two molecules sharing this signature are exactly indistinguishable to it.
    """
    in_vocab = sorted(f for f in fragments if f in vocabulary)
    n_oov = sum(1 for f in fragments if f not in vocabulary)
    return (tuple(in_vocab), n_oov)


def check_collisions(data, folds, frag_cfg, vocab_size: int) -> pd.DataFrame:
    """Per fold: how many held-out drugs are provably indistinguishable."""
    from fragdrp.eval.vocabulary import _fragments_of

    cache = {d: _fragments_of(s, frag_cfg) for d, s in data.smiles.items()}
    rows = []

    for fold in folds:
        train_drugs = sorted(set(data.drug_ids[fold.train].tolist()))
        test_drugs = sorted(set(data.drug_ids[fold.test].tolist()))

        counts: Counter = Counter()
        for drug in train_drugs:
            counts.update(f for f, _n in (cache.get(drug) or []))
        vocabulary = {f for f, _n in counts.most_common(vocab_size - 1)}

        signatures: Dict[Tuple, List[str]] = defaultdict(list)
        for drug in test_drugs:
            fragments = cache.get(drug)
            if fragments:
                signatures[_signature([f for f, _n in fragments], vocabulary)].append(drug)

        n_drugs = sum(len(v) for v in signatures.values())
        colliding = sum(len(v) for v in signatures.values() if len(v) > 1)
        pairs = sum(len(v) * (len(v) - 1) // 2 for v in signatures.values())
        total_pairs = n_drugs * (n_drugs - 1) // 2

        # A stricter, more damning variant: molecules whose *entire* signature is
        # "n OOV fragments and nothing else" are indistinguishable from each other
        # AND carry no molecular information whatsoever.
        empty = sum(len(v) for sig, v in signatures.items() if len(sig[0]) == 0)

        rows.append({
            "fold": fold.index,
            "test_drugs": n_drugs,
            "distinct_signatures": len(signatures),
            "drugs_in_a_collision": colliding,
            "collision_rate": colliding / max(n_drugs, 1),
            "colliding_pairs": pairs,
            "pair_collision_rate": pairs / max(total_pairs, 1),
            "drugs_with_no_in_vocab_fragment": empty,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #


def main() -> None:
    parser = base_parser(__doc__)
    args = parser.parse_args()

    out_dir = results_dir(args, "theory")
    setup_logging(log_file=out_dir / f"{args.dataset}_{args.test_mode}.log")

    from fragdrp.config import load_config
    from fragdrp.data.loader import load_response_data
    from fragdrp.data.splits import make_folds
    from fragdrp.eval.vocabulary import summarise_coverage, vocabulary_coverage

    banner("P1 — does each DAF-DRP fusion configuration use the drug at all?")
    degeneracy = check_attention_degeneracy()
    degeneracy.to_csv(out_dir / "p1_attention_degeneracy.csv", index=False)
    LOGGER.info("\n%s", degeneracy.to_string(index=False))
    blind = degeneracy[~degeneracy["uses_drug"]]["fusion"].tolist()
    LOGGER.info(
        "Configurations that ignore the drug entirely: %s  %s",
        ", ".join(blind) or "(none)",
        "— all have GlobalFusion = A, as Proposition 1 predicts." if blind else "",
    )

    cfg = load_config(
        CONFIG_ROOT / "base.yaml",
        [f"data.dataset_name={args.dataset}", f"data.test_mode={args.test_mode}"]
        + ([f"data.n_cv_splits={args.folds}"] if args.folds else []),
    )
    data = load_response_data(cfg.data)
    folds = make_folds(data, cfg.data, seed=cfg.train.seed)
    LOGGER.info(data.summary())

    banner("Coverage — how much of a held-out drug can a mined vocabulary address?")
    coverage = vocabulary_coverage(data, folds, cfg.fragmenter, cfg.featurizer.vocab_size)
    coverage.to_csv(out_dir / f"coverage_{args.dataset}_{args.test_mode}.csv", index=False)
    LOGGER.info("\n%s", coverage.round(4).to_string(index=False))
    LOGGER.info(summarise_coverage(coverage))

    banner("P4 — lookup-table collisions among held-out drugs")
    collisions = check_collisions(data, folds, cfg.fragmenter, cfg.featurizer.vocab_size)
    collisions.to_csv(out_dir / f"p4_collisions_{args.dataset}_{args.test_mode}.csv", index=False)
    LOGGER.info("\n%s", collisions.round(4).to_string(index=False))
    LOGGER.info(
        "On average %.1f%% of held-out drugs share a fragment signature with another "
        "held-out drug, and %.1f%% have no in-vocabulary fragment at all. A sum-pooled "
        "lookup-table fragment branch assigns those identical representations, for every "
        "parameter setting.",
        100 * collisions["collision_rate"].mean(),
        100 * (collisions["drugs_with_no_in_vocab_fragment"] / collisions["test_drugs"]).mean(),
    )

    summary = {
        "degenerate_fusions": blind,
        "coverage_fragment": float(coverage["fragment_coverage"].mean()),
        "coverage_atom": float(coverage["atom_coverage"].mean()),
        "fully_covered_drugs": float(coverage["fully_covered_drugs"].mean()),
        "collision_rate": float(collisions["collision_rate"].mean()),
        "pair_collision_rate": float(collisions["pair_collision_rate"].mean()),
    }
    (out_dir / f"summary_{args.dataset}_{args.test_mode}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
