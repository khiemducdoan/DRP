"""A curated pharmacophore / functional-group SMARTS vocabulary.

This list is the explicit chemical prior at the centre of the hypothesis.  Each
entry is a SMARTS pattern for a group that medicinal chemists treat as a unit
of activity -- a hinge-binding heterocycle, a solubilising amine, a metabolic
liability, a covalent warhead.

Two properties matter for the experiments:

1. **It is fixed and vocabulary-free at inference.**  Unlike a learned lookup
   table over mined fragments, a SMARTS match works on a molecule the model has
   never seen.  This is what lets the fragment view generalise in the
   leave-drug-out setting.
2. **It is interpretable.**  When the model attends to a fragment, we can name
   the groups inside it, which turns an attention weight into a chemical claim
   that a domain expert can check (experiment E5).

Patterns are applied to *fragments*, not to whole molecules, so the resulting
indicator vector says "this piece of the molecule is a sulfonamide", not "this
drug contains a sulfonamide somewhere".
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from rdkit import Chem

# (name, SMARTS).  Order is fixed: it defines the feature vector layout.
FUNCTIONAL_GROUPS: Sequence[Tuple[str, str]] = (
    # --- acids / bases (ionisation, solubility, binding) ------------------ #
    ("carboxylic_acid", "[CX3](=O)[OX2H1]"),
    ("carboxylate", "[CX3](=O)[OX1-]"),
    ("primary_amine", "[NX3;H2;!$(NC=[O,N,S]);!$(N[a])]"),
    ("secondary_amine", "[NX3;H1;!$(NC=[O,N,S]);!$(N[a])]"),
    ("tertiary_amine", "[NX3;H0;!$(NC=[O,N,S]);!$(N[a]);!$([N+])]"),
    ("quaternary_n", "[NX4+]"),
    ("guanidine", "[NX3][CX3](=[NX2])[NX3]"),
    ("amidine", "[NX3][CX3]=[NX2]"),
    # --- carbonyl family --------------------------------------------------- #
    ("amide", "[NX3][CX3](=[OX1])"),
    ("sulfonamide", "[NX3][SX4](=[OX1])(=[OX1])"),
    ("sulfone", "[#6][SX4](=[OX1])(=[OX1])[#6]"),
    ("sulfoxide", "[#6][SX3](=[OX1])[#6]"),
    ("ester", "[CX3](=[OX1])[OX2][#6]"),
    ("ketone", "[#6][CX3](=[OX1])[#6]"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("urea", "[NX3][CX3](=[OX1])[NX3]"),
    ("carbamate", "[NX3][CX3](=[OX1])[OX2]"),
    ("lactam", "[NX3;R][CX3;R](=[OX1])"),
    # --- polar neutral ----------------------------------------------------- #
    ("hydroxyl", "[OX2H][CX4]"),
    ("phenol", "[OX2H][c]"),
    ("ether", "[OD2]([#6])[#6]"),
    ("nitrile", "[NX1]#[CX2]"),
    ("nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("thiol", "[SX2H]"),
    ("thioether", "[SX2]([#6])[#6]"),
    # --- halogens / metabolic handles -------------------------------------- #
    ("fluorine", "[F]"),
    ("chlorine", "[Cl]"),
    ("bromo_iodo", "[Br,I]"),
    ("trifluoromethyl", "[CX4](F)(F)F"),
    # --- aromatic carbocycles ---------------------------------------------- #
    ("benzene", "c1ccccc1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    # --- N-heteroaromatics (the classic hinge binders) --------------------- #
    ("pyridine", "n1ccccc1"),
    ("pyrimidine", "c1cncnc1"),
    ("pyrazine_pyridazine", "[$(c1cnccn1),$(c1ccnnc1)]"),
    ("quinazoline", "c1ccc2ncncc2c1"),
    ("quinoline", "c1ccc2ncccc2c1"),
    ("indole", "c1ccc2[nH]ccc2c1"),
    ("indazole", "c1ccc2[nH]ncc2c1"),
    ("benzimidazole", "c1ccc2[nH]cnc2c1"),
    ("purine_like", "c1ncc2ncnc2n1"),
    ("imidazole", "c1c[nH]cn1"),
    ("pyrazole", "c1cc[nH]n1"),
    ("triazole", "c1nc[nH]n1"),
    ("tetrazole", "c1nnn[nH]1"),
    ("oxadiazole_thiadiazole", "[$(c1nnco1),$(c1nncs1)]"),
    ("thiophene", "c1ccsc1"),
    ("furan", "c1ccoc1"),
    ("thiazole", "c1cscn1"),
    ("oxazole_isoxazole", "[$(c1cocn1),$(c1ccno1)]"),
    # --- saturated N-heterocycles (solubilising tails) --------------------- #
    ("piperazine", "C1CNCCN1"),
    ("piperidine", "C1CCNCC1"),
    ("morpholine", "C1COCCN1"),
    ("pyrrolidine", "C1CCNC1"),
    # --- covalent warheads / reactive ------------------------------------- #
    ("michael_acceptor", "[CX3]=[CX3][CX3]=[OX1]"),
    ("epoxide", "C1OC1"),
    ("boronic_acid", "[BX3]([OX2H])[OX2H]"),
    # --- misc pharmacophores ----------------------------------------------- #
    ("phosphate_phosphonate", "[PX4](=O)([OX2,OX1-])[OX2,OX1-]"),
    ("hydrazine_hydrazone", "[NX3][NX3,NX2]"),
    ("biaryl", "[c]-[c]"),
    ("aniline", "[NX3;!$(NC=O)][c]"),
    ("benzylic", "[CX4][c]"),
)

GROUP_NAMES: List[str] = [name for name, _ in FUNCTIONAL_GROUPS]
N_FUNCTIONAL_GROUPS: int = len(FUNCTIONAL_GROUPS)

# Compiled once at import; SMARTS compilation is the expensive part.
_COMPILED: List[Tuple[str, Chem.Mol]] = [
    (name, patt)
    for name, smarts in FUNCTIONAL_GROUPS
    if (patt := Chem.MolFromSmarts(smarts)) is not None
]

if len(_COMPILED) != N_FUNCTIONAL_GROUPS:  # pragma: no cover - guards typos
    bad = set(GROUP_NAMES) - {n for n, _ in _COMPILED}
    raise RuntimeError(f"Invalid SMARTS for functional groups: {sorted(bad)}")


def match_counts(mol: Chem.Mol) -> Dict[str, int]:
    """Number of matches of each functional group in ``mol``."""
    return {name: len(mol.GetSubstructMatches(patt)) for name, patt in _COMPILED}


def indicator_vector(mol: Chem.Mol, binary: bool = True) -> np.ndarray:
    """Functional-group feature vector for one molecule or fragment.

    ``binary`` gives presence/absence; otherwise counts are log1p-compressed so
    that a fragment with eight fluorines does not swamp the rest of the vector.
    """
    counts = np.array(
        [len(mol.GetSubstructMatches(patt)) for _name, patt in _COMPILED],
        dtype=np.float32,
    )
    return (counts > 0).astype(np.float32) if binary else np.log1p(counts)


def describe(mol: Chem.Mol) -> List[str]:
    """Names of the functional groups present -- used for readable attributions."""
    return [name for name, patt in _COMPILED if mol.HasSubstructMatch(patt)]


# --------------------------------------------------------------------------- #
# Topological split (experiment E6, THEORY.md Proposition 3)
# --------------------------------------------------------------------------- #

#: Groups **defined by a cycle**: recognising one means recognising a ring
#: system, not just a local neighbourhood.  Curated by hand rather than derived
#: from the SMARTS string, because several patterns use recursive ``$(...)``
#: syntax whose outer query is a single atom even though the group is a ring.
RING_GROUPS: frozenset = frozenset({
    "benzene", "naphthalene", "pyridine", "pyrimidine", "pyrazine_pyridazine",
    "quinazoline", "quinoline", "indole", "indazole", "benzimidazole",
    "purine_like", "imidazole", "pyrazole", "triazole", "tetrazole",
    "oxadiazole_thiadiazole", "thiophene", "furan", "thiazole",
    "oxazole_isoxazole", "piperazine", "piperidine", "morpholine",
    "pyrrolidine", "epoxide", "lactam",
})

#: Groups that are a **tree around a central atom** -- the star-shaped patterns
#: that message passing can subgraph-count (Chen et al. 2020).
STAR_GROUPS: frozenset = frozenset(GROUP_NAMES) - RING_GROUPS

#: ``biaryl``, ``aniline``, ``phenol`` and ``benzylic`` are deliberately in
#: STAR_GROUPS: each is a *substituent relationship to* an aromatic atom, and
#: aromaticity is already supplied to the atom encoder as a node feature, so the
#: group adds a local pattern rather than a ring-identification problem.

GROUP_SUBSETS = {
    "all": frozenset(GROUP_NAMES),
    "ring": RING_GROUPS,
    "star": STAR_GROUPS,
    "none": frozenset(),
}


def subset_mask(subset: str) -> np.ndarray:
    """0/1 mask over the feature vector selecting one topological subset.

    Applied multiplicatively so the vector keeps its width and every model in
    the comparison keeps an identical parameter count -- otherwise a difference
    in performance could be a difference in capacity.
    """
    if subset not in GROUP_SUBSETS:
        raise ValueError(f"subset must be one of {sorted(GROUP_SUBSETS)}, got {subset!r}")
    keep = GROUP_SUBSETS[subset]
    return np.asarray([1.0 if n in keep else 0.0 for n in GROUP_NAMES], dtype=np.float32)
