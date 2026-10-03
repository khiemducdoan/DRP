"""How often does a learned fragment vocabulary fail on a held-out drug?

DAF-DRP represents a fragment by an index into a mined vocabulary of frequent
substructures.  The argument in `MODEL.md` §2.3 is that this cannot work under
leave-drug-out, because a drug the model has never seen routinely contains
fragments that were never mined.  That is an empirical claim, so this module
measures it instead of asserting it:

* **fragment coverage** -- the share of a held-out drug's fragments that appear
  in a vocabulary mined from the training drugs only;
* **drug coverage** -- the share of held-out drugs whose fragments are *all* in
  the vocabulary;
* **atom coverage** -- the share of a held-out drug's atoms that sit inside a
  covered fragment, which is the quantity that matters, since an uncovered
  fragment contributes nothing.

The same numbers also show why the vocabulary ablation (A6) is run with a
vocabulary mined over **all** drugs rather than the training drugs only: that is
deliberately generous to A6, so its result is an upper bound on what a
vocabulary representation can do here rather than a straw man.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem

from ..chem.featurizer import strip_salts
from ..chem.fragmenters import fragment_molecule
from ..config import FragmenterConfig
from ..data.loader import ResponseData
from ..data.splits import Fold

LOGGER = logging.getLogger(__name__)


def _fragments_of(smiles: str, cfg: FragmenterConfig) -> Optional[List[tuple]]:
    """``[(fragment_smiles, n_atoms), ...]`` for one molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = strip_salts(mol)
    result = fragment_molecule(mol, cfg, mol_key=smiles)
    return [
        (Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), canonical=True), len(atoms))
        for atoms in result.fragments
    ]


def vocabulary_coverage(
    data: ResponseData,
    folds: Sequence[Fold],
    frag_cfg: FragmenterConfig,
    vocab_size: int = 798,
) -> pd.DataFrame:
    """Per-fold coverage of held-out drugs by a training-only fragment vocabulary."""
    cache: Dict[str, Optional[List[tuple]]] = {
        drug_id: _fragments_of(smiles, frag_cfg) for drug_id, smiles in data.smiles.items()
    }

    rows: List[Dict[str, object]] = []
    for fold in folds:
        train_drugs = sorted(set(data.drug_ids[fold.train].tolist()))
        test_drugs = sorted(set(data.drug_ids[fold.test].tolist()))

        counter: Counter = Counter()
        for drug_id in train_drugs:
            for fragment, _n in cache.get(drug_id) or []:
                counter[fragment] += 1
        vocabulary = {s for s, _n in counter.most_common(vocab_size - 1)}

        covered_fragments = total_fragments = 0
        covered_atoms = total_atoms = 0
        fully_covered_drugs = 0
        scored_drugs = 0

        for drug_id in test_drugs:
            fragments = cache.get(drug_id)
            if not fragments:
                continue
            scored_drugs += 1
            hits = [f in vocabulary for f, _n in fragments]
            covered_fragments += sum(hits)
            total_fragments += len(hits)
            covered_atoms += sum(n for (_f, n), hit in zip(fragments, hits) if hit)
            total_atoms += sum(n for _f, n in fragments)
            fully_covered_drugs += int(all(hits))

        rows.append(
            {
                "fold": fold.index,
                "fragmenter": frag_cfg.method,
                "vocab_size": min(vocab_size - 1, len(counter)),
                "train_drugs": len(train_drugs),
                "test_drugs": scored_drugs,
                "fragment_coverage": covered_fragments / max(total_fragments, 1),
                "atom_coverage": covered_atoms / max(total_atoms, 1),
                "fully_covered_drugs": fully_covered_drugs / max(scored_drugs, 1),
            }
        )
    return pd.DataFrame(rows)


def summarise_coverage(frame: pd.DataFrame) -> str:
    """A sentence the report can use directly."""
    if frame.empty:
        return "Vocabulary coverage could not be computed."
    fragment = 100 * frame["fragment_coverage"].mean()
    atom = 100 * frame["atom_coverage"].mean()
    whole = 100 * frame["fully_covered_drugs"].mean()
    return (
        f"With a vocabulary mined from the training drugs only, "
        f"{fragment:.1f}% of a held-out drug's fragments and {atom:.1f}% of its atoms "
        f"are covered on average, and only {whole:.1f}% of held-out drugs are covered "
        f"completely. The rest of each molecule has no learned representation at all "
        f"under a lookup-table scheme."
    )
