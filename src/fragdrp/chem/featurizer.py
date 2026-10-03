"""Turn a SMILES string into the two-level graph FragDRP consumes.

The output is a :class:`DrugGraph`: an atom graph, a fragment (junction) graph,
and the atom-to-fragment assignment that links them.  Everything is plain numpy
so it pickles cheaply into the feature cache.

Design note -- why fragment features are *computed*, not *looked up*
-------------------------------------------------------------------
DAF-DRP represents a fragment by an index into a mined vocabulary of 798
substructures and learns an embedding per index.  That representation has no
value for a fragment outside the vocabulary, and no notion that two fragments
are chemically similar.  In leave-drug-out evaluation, where held-out drugs
routinely contain unseen pieces, this is the binding constraint.

Here a fragment is described by quantities that exist for *any* fragment:

* the pooled states of its constituent atoms, after atom-level message passing
  (so the same ring in two different molecules gets two different vectors,
  reflecting its substituents);
* physicochemical descriptors computed on the fragment itself;
* a functional-group SMARTS indicator vector.

None of these require the fragment to have been seen during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

from ..config import FeaturizerConfig, FragmenterConfig
from . import functional_groups as fg
from .fragmenters import Fragmentation, fragment_molecule

RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------- #
# Atom / bond feature vocabularies
# --------------------------------------------------------------------------- #

ATOM_SYMBOLS: Sequence[str] = (
    "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "other",
)
DEGREES: Sequence[int] = (0, 1, 2, 3, 4, 5)
FORMAL_CHARGES: Sequence[int] = (-2, -1, 0, 1, 2)
NUM_HS: Sequence[int] = (0, 1, 2, 3, 4)
HYBRIDISATIONS: Sequence[Chem.HybridizationType] = (
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
)
CHIRAL_TAGS: Sequence[Chem.ChiralType] = (
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
)
BOND_TYPES: Sequence[Chem.BondType] = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)
BOND_STEREO: Sequence[Chem.BondStereo] = (
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
)

DESCRIPTOR_NAMES: Sequence[str] = (
    "n_heavy_atoms", "mol_weight", "logp", "molar_refractivity", "tpsa",
    "n_hbd", "n_hba", "n_rings", "n_aromatic_rings", "n_rotatable_bonds",
    "fraction_csp3", "formal_charge", "n_halogens", "n_N", "n_O", "n_S",
    "n_aromatic_atoms", "heteroatom_fraction",
)
N_DESCRIPTORS = len(DESCRIPTOR_NAMES)


def _one_hot(value, vocabulary: Sequence, allow_other: bool = True) -> List[float]:
    """One-hot encode ``value``; an extra trailing slot catches anything unseen."""
    vec = [0.0] * (len(vocabulary) + (1 if allow_other else 0))
    try:
        vec[vocabulary.index(value)] = 1.0
    except ValueError:
        if allow_other:
            vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> List[float]:
    """Per-atom feature vector."""
    symbol = atom.GetSymbol()
    return (
        _one_hot(symbol if symbol in ATOM_SYMBOLS else "other", ATOM_SYMBOLS, allow_other=False)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + _one_hot(atom.GetHybridization(), HYBRIDISATIONS)
        + _one_hot(atom.GetChiralTag(), CHIRAL_TAGS)
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.IsInRingSize(3)),
            float(atom.IsInRingSize(4)),
            float(atom.IsInRingSize(5)),
            float(atom.IsInRingSize(6)),
            float(atom.GetMass() * 0.01),
        ]
    )


def bond_features(bond: Chem.Bond) -> List[float]:
    """Per-bond feature vector."""
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + _one_hot(bond.GetStereo(), BOND_STEREO)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    )


ATOM_FEATURE_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_FEATURE_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))
# Junction edges carry the cut bond's features plus a flag marking them as cuts.
JUNCTION_FEATURE_DIM = BOND_FEATURE_DIM + 1


# --------------------------------------------------------------------------- #
# Fragment-level descriptors
# --------------------------------------------------------------------------- #


def _fragment_submol(mol: Chem.Mol, atom_indices: Sequence[int]) -> Optional[Chem.Mol]:
    """Extract a fragment as a standalone molecule, so descriptors can be computed.

    Open valences created by the cut are left implicit rather than capped with
    dummy atoms: capping would change TPSA and H-bond counts in a way that
    depends on the cut, which is exactly the confound we want to avoid when
    comparing fragmentation strategies.
    """
    try:
        smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atom_indices), canonical=True)
        sub = Chem.MolFromSmiles(smiles, sanitize=True)
        if sub is None:
            sub = Chem.MolFromSmiles(smiles, sanitize=False)
            if sub is not None:
                sub.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(
                    sub,
                    Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                    catchErrors=True,
                )
        return sub
    except Exception:
        return None


def fragment_descriptors(mol: Chem.Mol, atom_indices: Sequence[int]) -> np.ndarray:
    """Physicochemical descriptors of one fragment.

    Computed on the extracted sub-molecule where RDKit can parse it, and
    otherwise from atom-level counts, so this never fails.
    """
    sub = _fragment_submol(mol, atom_indices)
    atoms = [mol.GetAtomWithIdx(int(i)) for i in atom_indices]

    n_heavy = float(len(atoms))
    n_aromatic = float(sum(a.GetIsAromatic() for a in atoms))
    n_hetero = float(sum(a.GetSymbol() not in ("C", "H") for a in atoms))
    counts = {s: float(sum(a.GetSymbol() == s for a in atoms)) for s in ("N", "O", "S")}
    n_halogens = float(sum(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in atoms))
    charge = float(sum(a.GetFormalCharge() for a in atoms))

    if sub is not None:
        try:
            values = [
                n_heavy,
                Descriptors.MolWt(sub) * 0.01,
                Crippen.MolLogP(sub),
                Crippen.MolMR(sub) * 0.1,
                rdMolDescriptors.CalcTPSA(sub) * 0.01,
                float(rdMolDescriptors.CalcNumHBD(sub)),
                float(rdMolDescriptors.CalcNumHBA(sub)),
                float(rdMolDescriptors.CalcNumRings(sub)),
                float(rdMolDescriptors.CalcNumAromaticRings(sub)),
                float(rdMolDescriptors.CalcNumRotatableBonds(sub)),
                float(rdMolDescriptors.CalcFractionCSP3(sub)),
                charge,
                n_halogens,
                counts["N"],
                counts["O"],
                counts["S"],
                n_aromatic,
                n_hetero / max(n_heavy, 1.0),
            ]
            return np.asarray(values, dtype=np.float32)
        except Exception:
            pass

    # Fallback: everything we can derive without a sanitised sub-molecule.
    values = [
        n_heavy, sum(a.GetMass() for a in atoms) * 0.01, 0.0, 0.0, 0.0,
        0.0, counts["N"] + counts["O"], 0.0, 0.0, 0.0,
        0.0, charge, n_halogens, counts["N"], counts["O"], counts["S"],
        n_aromatic, n_hetero / max(n_heavy, 1.0),
    ]
    return np.asarray(values, dtype=np.float32)


def fragment_functional_groups(mol: Chem.Mol, atom_indices: Sequence[int]) -> np.ndarray:
    """Functional-group indicator vector for one fragment."""
    sub = _fragment_submol(mol, atom_indices)
    if sub is None:
        return np.zeros(fg.N_FUNCTIONAL_GROUPS, dtype=np.float32)
    try:
        return fg.indicator_vector(sub, binary=True)
    except Exception:
        return np.zeros(fg.N_FUNCTIONAL_GROUPS, dtype=np.float32)


# --------------------------------------------------------------------------- #
# The featurised drug
# --------------------------------------------------------------------------- #


@dataclass
class DrugGraph:
    """A drug as a two-level graph.

    Atom level
        ``atom_x`` ``(n_atoms, ATOM_FEATURE_DIM)``,
        ``atom_edge_index`` ``(2, n_bonds * 2)`` (both directions),
        ``atom_edge_attr`` ``(n_bonds * 2, BOND_FEATURE_DIM)``.

    Fragment level
        ``frag_edge_index`` ``(2, n_junctions * 2)``,
        ``frag_edge_attr`` ``(n_junctions * 2, JUNCTION_FEATURE_DIM)``,
        ``frag_descriptors`` ``(n_fragments, N_DESCRIPTORS)``,
        ``frag_functional_groups`` ``(n_fragments, N_FUNCTIONAL_GROUPS)``.

    Link
        ``atom_to_fragment`` ``(n_atoms,)``.
    """

    drug_id: str
    smiles: str
    atom_x: np.ndarray
    atom_edge_index: np.ndarray
    atom_edge_attr: np.ndarray
    atom_to_fragment: np.ndarray
    frag_edge_index: np.ndarray
    frag_edge_attr: np.ndarray
    frag_descriptors: np.ndarray
    frag_functional_groups: np.ndarray
    frag_smiles: List[str]
    frag_vocab_ids: np.ndarray

    @property
    def n_atoms(self) -> int:
        return int(self.atom_x.shape[0])

    @property
    def n_fragments(self) -> int:
        return int(self.frag_descriptors.shape[0])


def strip_salts(mol: Chem.Mol) -> Chem.Mol:
    """Keep the largest connected component.

    Drug tables list salts (hydrochlorides, mesylates, ...).  The counter-ion is
    not the pharmacophore, and leaving it in gives the fragment view a spurious
    disconnected node in the junction graph.

    The returned fragment is sanitised.  Splitting without sanitising leaves ring
    information uninitialised on the pieces, and BRICS -- which asks whether a
    bond is in a ring -- then raises ``RingInfo not initialized`` on exactly the
    molecules this function is meant to clean up.
    """
    try:
        pieces = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return mol
    if len(pieces) <= 1:
        return mol
    largest = max(pieces, key=lambda m: m.GetNumHeavyAtoms())
    # Belt and braces: perceive rings explicitly, since a downstream BRICS call
    # asks whether each bond is in a ring and raises if that was never computed.
    Chem.GetSymmSSSR(largest)
    return largest


def featurize_drug(
    drug_id: str,
    smiles: str,
    frag_cfg: FragmenterConfig,
    feat_cfg: FeaturizerConfig,
    vocab: Optional[Dict[str, int]] = None,
) -> Optional[DrugGraph]:
    """Featurise one drug.  Returns ``None`` if the SMILES cannot be parsed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = strip_salts(mol)
    if mol.GetNumAtoms() == 0:
        return None

    # --- atom graph -------------------------------------------------------- #
    atom_x = np.asarray([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    src: List[int] = []
    dst: List[int] = []
    edge_attr: List[List[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        edge_attr += [feats, feats]
    atom_edge_index = (
        np.asarray([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    )
    atom_edge_attr = (
        np.asarray(edge_attr, dtype=np.float32)
        if edge_attr
        else np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)
    )

    # --- fragmentation ----------------------------------------------------- #
    frag: Fragmentation = fragment_molecule(mol, frag_cfg, mol_key=smiles)

    fsrc: List[int] = []
    fdst: List[int] = []
    fattr: List[List[float]] = []
    for atom_i, atom_j, frag_i, frag_j in frag.junction_bonds:
        bond = mol.GetBondBetweenAtoms(atom_i, atom_j)
        feats = (bond_features(bond) if bond is not None else [0.0] * BOND_FEATURE_DIM) + [1.0]
        fsrc += [frag_i, frag_j]
        fdst += [frag_j, frag_i]
        fattr += [feats, feats]
    frag_edge_index = (
        np.asarray([fsrc, fdst], dtype=np.int64) if fsrc else np.zeros((2, 0), dtype=np.int64)
    )
    frag_edge_attr = (
        np.asarray(fattr, dtype=np.float32)
        if fattr
        else np.zeros((0, JUNCTION_FEATURE_DIM), dtype=np.float32)
    )

    # --- fragment priors --------------------------------------------------- #
    n_frags = frag.n_fragments
    descriptors = (
        np.stack([fragment_descriptors(mol, atoms) for atoms in frag.fragments])
        if feat_cfg.use_descriptors
        else np.zeros((n_frags, N_DESCRIPTORS), dtype=np.float32)
    )
    if feat_cfg.use_functional_groups:
        groups = np.stack(
            [fragment_functional_groups(mol, atoms) for atoms in frag.fragments]
        )
        subset = getattr(feat_cfg, "functional_group_subset", "all")
        if subset != "all":
            groups = groups * fg.subset_mask(subset)
    else:
        groups = np.zeros((n_frags, fg.N_FUNCTIONAL_GROUPS), dtype=np.float32)

    frag_smiles = [
        Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), canonical=True)
        for atoms in frag.fragments
    ]

    # DAF-DRP parity: index into a mined vocabulary, 0 = out-of-vocabulary.
    if feat_cfg.use_fragment_vocab and vocab is not None:
        vocab_ids = np.asarray([vocab.get(s, 0) for s in frag_smiles], dtype=np.int64)
    else:
        vocab_ids = np.zeros(n_frags, dtype=np.int64)

    return DrugGraph(
        drug_id=drug_id,
        smiles=smiles,
        atom_x=atom_x,
        atom_edge_index=atom_edge_index,
        atom_edge_attr=atom_edge_attr,
        atom_to_fragment=np.asarray(frag.atom_to_fragment, dtype=np.int64),
        frag_edge_index=frag_edge_index,
        frag_edge_attr=frag_edge_attr,
        frag_descriptors=descriptors.astype(np.float32),
        frag_functional_groups=groups.astype(np.float32),
        frag_smiles=frag_smiles,
        frag_vocab_ids=vocab_ids,
    )


def build_fragment_vocabulary(
    smiles_list: Sequence[str], frag_cfg: FragmenterConfig, size: int
) -> Dict[str, int]:
    """Most frequent fragment SMILES, ranked -- the DAF-DRP-style lookup table.

    Index 0 is reserved for out-of-vocabulary fragments, which is the whole
    point of including this baseline: it lets the report quantify how often a
    held-out drug falls off the end of the vocabulary.
    """
    from collections import Counter

    counter: Counter = Counter()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = strip_salts(mol)
        frag = fragment_molecule(mol, frag_cfg, mol_key=smiles)
        for atoms in frag.fragments:
            counter[Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), canonical=True)] += 1
    return {s: i + 1 for i, (s, _n) in enumerate(counter.most_common(size - 1))}
