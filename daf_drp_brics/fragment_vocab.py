"""BRICS-derived fragment vocabulary — the one substitution in this folder.

DAF-DRP decomposes a molecule against a vocabulary ``S = {S_1 … S_K}`` of
|S| = 798 frequent substructures obtained by Principal Subgraph Mining (Kong et
al. 2022, as used by Luong & Singh 2023), then represents fragment *k* by a
learnable embedding indexed by that fragment's identity in ``S``.

Here the decomposition comes from **BRICS** (Degen et al. 2008) instead.
Everything downstream is unchanged: fragments are still identified by a
vocabulary index and still embedded through a learnable lookup table of shape
``|S| x 128``.

Why the swap is worth making
----------------------------
Principal Subgraph Mining needs a mining pass over a large corpus and produces a
vocabulary specific to that corpus; BRICS is a fixed set of 16 retrosynthetic
rules that runs on any molecule with no corpus at all.  If the fragment view is
carrying chemistry rather than corpus statistics, the two should behave alike.

What the swap does *not* fix
----------------------------
Both schemes address a fragment by an index, so a fragment absent from the
vocabulary has no representation and falls back to the out-of-vocabulary slot.
On GDSC2, BRICS yields 422 distinct fragments across 241 drugs, of which **302
occur exactly once** — so most of the vocabulary is learned from a single
molecule, and a held-out drug routinely contains pieces the table has never had
a gradient for.  That limitation is inherent to a lookup-table representation
and survives the change of fragmenter.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger

from fragdrp.chem.featurizer import strip_salts
from fragdrp.chem.fragmenters import Fragmentation, fragment_molecule
from fragdrp.config import FragmenterConfig

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger(__name__)

#: Index 0 is reserved for fragments outside the vocabulary.
OOV_INDEX = 0


@dataclass
class FragmentVocabulary:
    """Maps a canonical fragment SMILES to an embedding index.

    Attributes
    ----------
    index_of:
        ``fragment SMILES -> index`` (indices start at 1; 0 means OOV).
    counts:
        Training frequency of each fragment, kept for the coverage report.
    size:
        Number of embedding rows, i.e. ``max(index) + 1``.
    """

    index_of: Dict[str, int]
    counts: Counter
    size: int
    method: str = "brics"

    def encode(self, fragments: Sequence[str]) -> List[int]:
        return [self.index_of.get(s, OOV_INDEX) for s in fragments]

    def coverage(self, fragments: Iterable[str]) -> float:
        fragments = list(fragments)
        if not fragments:
            return 1.0
        return sum(s in self.index_of for s in fragments) / len(fragments)


def fragment_smiles_of(mol: Chem.Mol, fragmentation: Fragmentation) -> List[str]:
    """Canonical SMILES of every fragment, in fragment-index order."""
    return [
        Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), canonical=True)
        for atoms in fragmentation.fragments
    ]


def decompose(smiles: str, cfg: FragmenterConfig) -> Optional[Tuple[Chem.Mol, Fragmentation]]:
    """Parse, strip the counter-ion, and fragment.  ``None`` if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = strip_salts(mol)
    if mol.GetNumAtoms() == 0:
        return None
    return mol, fragment_molecule(mol, cfg, mol_key=smiles)


def build_vocabulary(
    smiles: Sequence[str],
    cfg: Optional[FragmenterConfig] = None,
    max_size: int = 798,
    min_count: int = 1,
) -> FragmentVocabulary:
    """Mine a vocabulary of the most frequent BRICS fragments.

    ``max_size`` matches DAF-DRP's |S| = 798.  On GDSC2, BRICS produces fewer
    distinct fragments than that, so the cap never binds and the vocabulary is
    strictly more generous than the paper's.

    ``min_count`` is left at 1 for faithfulness: DAF-DRP keeps the full mined
    vocabulary.  Raising it to 2 would drop the 302 fragments that occur once,
    which is worth trying, because an embedding row trained on a single molecule
    is mostly noise.
    """
    cfg = cfg or FragmenterConfig(method="brics")
    counts: Counter = Counter()
    failed = 0

    for smi in smiles:
        decomposed = decompose(smi, cfg)
        if decomposed is None:
            failed += 1
            continue
        mol, fragmentation = decomposed
        counts.update(fragment_smiles_of(mol, fragmentation))

    kept = [(s, n) for s, n in counts.most_common() if n >= min_count][: max_size - 1]
    index_of = {s: i + 1 for i, (s, _n) in enumerate(kept)}

    LOGGER.info(
        "BRICS vocabulary: %d distinct fragments from %d molecules (%d unparseable); "
        "kept %d (cap %d, min_count %d); %d occur exactly once",
        len(counts), len(smiles), failed, len(index_of), max_size, min_count,
        sum(1 for _s, n in counts.items() if n == 1),
    )
    return FragmentVocabulary(
        index_of=index_of, counts=counts, size=max_size, method=cfg.method
    )


def coverage_report(
    vocabulary: FragmentVocabulary,
    smiles_by_drug: Dict[str, str],
    cfg: Optional[FragmenterConfig] = None,
) -> Dict[str, float]:
    """How much of a set of drugs the vocabulary can actually address.

    Reported before training, because it bounds what the fragment branch can
    contribute: an out-of-vocabulary fragment contributes only the OOV
    embedding, which is the same vector for every unrecognised piece of
    chemistry.
    """
    cfg = cfg or FragmenterConfig(method="brics")
    covered = total = 0
    covered_atoms = total_atoms = 0
    fully = scored = 0

    for smi in smiles_by_drug.values():
        decomposed = decompose(smi, cfg)
        if decomposed is None:
            continue
        mol, fragmentation = decomposed
        names = fragment_smiles_of(mol, fragmentation)
        hits = [s in vocabulary.index_of for s in names]
        covered += sum(hits)
        total += len(hits)
        for atoms, hit in zip(fragmentation.fragments, hits):
            total_atoms += len(atoms)
            if hit:
                covered_atoms += len(atoms)
        fully += int(all(hits))
        scored += 1

    return {
        "drugs": scored,
        "fragment_coverage": covered / max(total, 1),
        "atom_coverage": covered_atoms / max(total_atoms, 1),
        "fully_covered_drugs": fully / max(scored, 1),
    }
