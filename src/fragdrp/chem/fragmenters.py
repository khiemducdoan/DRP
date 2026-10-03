"""Molecular fragmentation strategies.

Every fragmenter answers the same question: *which bonds do we cut?*  Once the
cut bonds are known, the fragments are simply the connected components of the
molecule with those bonds removed, and the junction graph has one edge per cut
bond.  Keeping the whole family behind that one decision makes the strategies
directly comparable -- experiment E3 in EXPERIMENTS.md swaps ``method`` and
changes nothing else.

Implemented strategies
----------------------
``brics``          Breaking of Retrosynthetically Interesting Chemical
                   Substructures (Degen et al. 2008).  The default: its 16 rules
                   encode bonds a medicinal chemist would actually disconnect,
                   so the resulting pieces tend to be functional groups and
                   ring systems rather than arbitrary chunks.
``recap``          Retrosynthetic Combinatorial Analysis Procedure (Lewell et
                   al. 1998).  Fewer, more conservative rules than BRICS -- a
                   useful "is it BRICS specifically, or any chemistry-aware
                   cut?" comparison.
``murcko``         Ring systems versus side chains (Bemis & Murcko scaffold
                   logic).  Coarse, topology-driven, chemistry-light.
``rotatable_bond`` Cut every rotatable bond.  Conformationally motivated, not
                   reaction motivated.
``random_cut``     **The control.** Cuts the same *number* of acyclic bonds
                   BRICS would cut on that molecule, but chooses them at
                   random.  Fragment count and size distribution therefore
                   match BRICS; only chemical meaning is removed.  If this
                   matches BRICS, the functional-group hypothesis is false.
``whole``          One fragment per molecule (degenerate fragment view).
``atom``           One fragment per atom (degenerate atom view).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from rdkit import Chem
from rdkit.Chem import BRICS

from ..config import FragmenterConfig

Bond = Tuple[int, int]


@dataclass
class Fragmentation:
    """The result of cutting one molecule.

    Attributes
    ----------
    atom_to_fragment:
        ``atom_to_fragment[i]`` is the fragment index of atom ``i``.  Always a
        complete partition: every atom belongs to exactly one fragment.
    fragments:
        ``fragments[k]`` lists the atom indices of fragment ``k``.
    junction_bonds:
        ``(atom_u, atom_v, fragment_of_u, fragment_of_v)`` for every cut bond
        that survived fragment merging.
    method:
        Name of the strategy that produced this fragmentation.
    """

    atom_to_fragment: List[int]
    fragments: List[List[int]]
    junction_bonds: List[Tuple[int, int, int, int]]
    method: str

    @property
    def n_fragments(self) -> int:
        return len(self.fragments)


# --------------------------------------------------------------------------- #
# Cut-bond finders: each returns a set of (atom_i, atom_j) bonds to break
# --------------------------------------------------------------------------- #


def _brics_bonds(mol: Chem.Mol) -> Set[Bond]:
    """Bonds matching one of the 16 BRICS retrosynthetic rules."""
    return {tuple(sorted(pair)) for pair, _labels in BRICS.FindBRICSBonds(mol)}  # type: ignore[misc]


# RECAP's 11 cleavage rules, written as two-atom SMARTS.  The bond between the
# first two matched atoms is the cut point.  Ring bonds are excluded later, in
# ``_acyclic_only``, which is RECAP's own constraint.
_RECAP_PATTERNS: Sequence[Tuple[str, str]] = (
    ("amide", "[C;$(C=O)]-[#7]"),
    ("ester", "[C;$(C=O)]-[O;H0]"),
    ("amine", "[N;!$(N-C=[O,N,S]);!$(N~[!#6;!#1])]-[C;!$(C=[O,N,S])]"),
    ("urea", "[#7]-[C;$(C(=O)(-[#7])-[#7])]"),
    ("ether", "[O;H0;!$(O-C=O)]-[C;!$(C=[O,N,S])]"),
    ("olefin", "[C;!R]=[C;!R]"),
    ("quaternary_n", "[N+;H0;!$(N=*)]-[C]"),
    ("aromatic_n_aliphatic_c", "[n]-[C;!$(C=[O,N,S])]"),
    ("lactam_n_aliphatic_c", "[N;R;$(N-C=O)]-[C;!R]"),
    ("aromatic_c_aromatic_c", "[c]-[c]"),
    ("sulphonamide", "[#7]-[S;$(S(=O)(=O))]"),
)


def _recap_bonds(mol: Chem.Mol) -> Set[Bond]:
    """Bonds matching one of the RECAP cleavage rules."""
    bonds: Set[Bond] = set()
    for _name, smarts in _RECAP_PATTERNS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            if len(match) >= 2:
                bonds.add(tuple(sorted(match[:2])))  # type: ignore[arg-type]
    return bonds


def _ring_system_of_atoms(mol: Chem.Mol) -> Dict[int, int]:
    """Map each ring atom to a fused-ring-system id (non-ring atoms are absent)."""
    ring_info = mol.GetRingInfo()
    systems: List[Set[int]] = []
    for ring in ring_info.AtomRings():
        ring_set = set(ring)
        merged = [s for s in systems if s & ring_set]
        for s in merged:
            systems.remove(s)
            ring_set |= s
        systems.append(ring_set)
    return {atom: idx for idx, system in enumerate(systems) for atom in system}


def _murcko_bonds(mol: Chem.Mol) -> Set[Bond]:
    """Cut acyclic bonds that separate ring systems from each other or from
    side chains -- Bemis & Murcko scaffold/side-chain decomposition."""
    ring_system = _ring_system_of_atoms(mol)
    bonds: Set[Bond] = set()
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        si, sj = ring_system.get(i), ring_system.get(j)
        # ring-to-ring (different systems), or ring-to-sidechain
        if si is not None or sj is not None:
            if si != sj:
                bonds.add((min(i, j), max(i, j)))
    return bonds


_ROTATABLE = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


def _rotatable_bonds(mol: Chem.Mol) -> Set[Bond]:
    """Every rotatable bond (RDKit's strict definition)."""
    return {tuple(sorted(m)) for m in mol.GetSubstructMatches(_ROTATABLE)}  # type: ignore[misc]


def _acyclic_single_bonds(mol: Chem.Mol) -> List[Bond]:
    """Candidate bonds a random cut is allowed to pick."""
    out = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        out.append((min(i, j), max(i, j)))
    return out


def _random_bonds(mol: Chem.Mol, cfg: FragmenterConfig, mol_key: str) -> Set[Bond]:
    """Control fragmenter: cuts placed uniformly at random on acyclic single bonds.

    The RNG is seeded per molecule (``random_cut_seed`` plus the SMILES) so the
    control is deterministic across runs while still differing between
    molecules.  When ``match_reference_count`` is set, the *number* of cuts is
    tuned by :func:`_fragment_random_matched` so the final fragment count
    matches BRICS; here we only provide the starting point.
    """
    candidates = _acyclic_single_bonds(mol)
    if not candidates:
        return set()
    n_cuts = len(_brics_bonds(mol)) if cfg.match_reference_count else max(1, len(candidates) // 3)
    n_cuts = min(n_cuts, len(candidates))
    rng = random.Random(f"{cfg.random_cut_seed}:{mol_key}")
    return set(rng.sample(candidates, n_cuts))


def _acyclic_only(mol: Chem.Mol, bonds: Set[Bond]) -> Set[Bond]:
    """Drop ring bonds: cutting inside a ring produces chemically odd pieces and
    would let a single cut open a whole ring system."""
    keep: Set[Bond] = set()
    for i, j in bonds:
        bond = mol.GetBondBetweenAtoms(i, j)
        if bond is not None and not bond.IsInRing():
            keep.add((i, j))
    return keep


_FINDERS = {
    "brics": lambda mol, cfg, key: _brics_bonds(mol),
    "recap": lambda mol, cfg, key: _recap_bonds(mol),
    "murcko": lambda mol, cfg, key: _murcko_bonds(mol),
    "rotatable_bond": lambda mol, cfg, key: _rotatable_bonds(mol),
    "random_cut": _random_bonds,
    "whole": lambda mol, cfg, key: set(),
    "atom": lambda mol, cfg, key: {
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in mol.GetBonds()
    },
}

AVAILABLE_METHODS = tuple(_FINDERS)


# --------------------------------------------------------------------------- #
# Cut bonds -> fragments
# --------------------------------------------------------------------------- #


def _connected_components(n_atoms: int, adjacency: Dict[int, Set[int]]) -> List[List[int]]:
    """Group atoms into connected components (iterative DFS, no recursion limit)."""
    seen = [False] * n_atoms
    components: List[List[int]] = []
    for start in range(n_atoms):
        if seen[start]:
            continue
        stack, component = [start], []
        seen[start] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in adjacency.get(node, ()):
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def _merge_fragments(
    fragments: List[List[int]],
    neighbours: Dict[int, Set[int]],
    min_atoms: int,
    max_fragments: int,
) -> List[List[int]]:
    """Merge fragments that are too small, then merge until the count fits.

    A fragment always merges into its *largest* neighbour, which keeps the
    junction graph connected and avoids producing a long chain of tiny pieces.
    """
    frags = [list(f) for f in fragments]
    adj = {k: set(v) for k, v in neighbours.items()}
    alive = set(range(len(frags)))

    def merge(src: int, dst: int) -> None:
        frags[dst].extend(frags[src])
        frags[dst].sort()
        for nb in adj[src]:
            if nb != dst:
                adj[nb].discard(src)
                adj[nb].add(dst)
                adj[dst].add(nb)
        adj[dst].discard(src)
        adj[src] = set()
        alive.discard(src)

    def smallest_mergeable() -> int | None:
        candidates = [k for k in alive if adj[k]]
        return min(candidates, key=lambda k: len(frags[k])) if candidates else None

    # Pass 1: absorb fragments below the size floor.
    changed = True
    while changed:
        changed = False
        for k in sorted(alive, key=lambda k: len(frags[k])):
            if k in alive and len(frags[k]) < min_atoms and adj[k]:
                target = max(adj[k], key=lambda n: len(frags[n]))
                merge(k, target)
                changed = True
                break

    # Pass 2: enforce the fragment-count ceiling.
    while len(alive) > max_fragments:
        k = smallest_mergeable()
        if k is None:
            break
        target = max(adj[k], key=lambda n: len(frags[n]))
        merge(k, target)

    return [frags[k] for k in sorted(alive)]


def _build_fragmentation(
    mol: Chem.Mol, cut: Set[Bond], cfg: FragmenterConfig, method: str
) -> Fragmentation:
    """Turn a set of cut bonds into a :class:`Fragmentation`."""
    n_atoms = mol.GetNumAtoms()

    # Molecular graph minus the cut bonds.
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(n_atoms)}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (min(i, j), max(i, j)) in cut:
            continue
        adjacency[i].add(j)
        adjacency[j].add(i)

    fragments = _connected_components(n_atoms, adjacency)
    atom_to_fragment = [0] * n_atoms
    for k, atoms in enumerate(fragments):
        for atom in atoms:
            atom_to_fragment[atom] = k

    frag_neighbours: Dict[int, Set[int]] = {k: set() for k in range(len(fragments))}
    for i, j in cut:
        fi, fj = atom_to_fragment[i], atom_to_fragment[j]
        if fi != fj:
            frag_neighbours[fi].add(fj)
            frag_neighbours[fj].add(fi)

    min_atoms = 1 if method in ("atom", "whole") else cfg.min_fragment_atoms
    fragments = _merge_fragments(fragments, frag_neighbours, min_atoms, cfg.max_fragments)

    # Re-index after merging.
    atom_to_fragment = [0] * n_atoms
    for k, atoms in enumerate(fragments):
        for atom in atoms:
            atom_to_fragment[atom] = k

    junctions: List[Tuple[int, int, int, int]] = []
    for i, j in sorted(cut):
        fi, fj = atom_to_fragment[i], atom_to_fragment[j]
        if fi != fj:
            junctions.append((i, j, fi, fj))

    return Fragmentation(
        atom_to_fragment=atom_to_fragment,
        fragments=fragments,
        junction_bonds=junctions,
        method=method,
    )


def _fragment_random_matched(mol: Chem.Mol, cfg: FragmenterConfig, mol_key: str) -> Fragmentation:
    """Random-cut control whose *final* fragment count matches BRICS.

    Matching the number of cut bonds is not enough: BRICS places its cuts where
    they yield chemically substantial pieces, whereas random cuts often shave
    off single atoms that the merging step then reabsorbs.  Matching the number
    of surviving fragments instead makes the control differ from BRICS in the
    *placement* of cuts and in nothing else -- which is precisely the variable
    the functional-group hypothesis is about.
    """
    target = _build_fragmentation(
        mol, _acyclic_only(mol, _brics_bonds(mol)), cfg, "brics"
    ).n_fragments

    candidates = _acyclic_single_bonds(mol)
    if not candidates:
        return _build_fragmentation(mol, set(), cfg, "random_cut")

    rng = random.Random(f"{cfg.random_cut_seed}:{mol_key}")
    shuffled = candidates[:]
    rng.shuffle(shuffled)

    # Add random cuts one at a time and keep the prefix that lands closest to
    # the BRICS fragment count.
    best = _build_fragmentation(mol, set(), cfg, "random_cut")
    best_gap = abs(best.n_fragments - target)
    for k in range(1, len(shuffled) + 1):
        candidate = _build_fragmentation(mol, set(shuffled[:k]), cfg, "random_cut")
        gap = abs(candidate.n_fragments - target)
        if gap < best_gap:
            best, best_gap = candidate, gap
        if candidate.n_fragments >= target:
            break
    return best


def fragment_molecule(mol: Chem.Mol, cfg: FragmenterConfig, mol_key: str = "") -> Fragmentation:
    """Fragment ``mol`` according to ``cfg``.

    ``mol_key`` (normally the canonical SMILES) seeds the ``random_cut`` control
    so that it is deterministic per molecule.
    """
    if cfg.method not in _FINDERS:
        raise ValueError(f"Unknown fragmenter {cfg.method!r}; choose from {AVAILABLE_METHODS}")

    if cfg.method == "random_cut" and cfg.match_reference_count:
        return _fragment_random_matched(mol, cfg, mol_key)

    cut = _FINDERS[cfg.method](mol, cfg, mol_key)
    if cfg.method != "atom":
        cut = _acyclic_only(mol, cut)
    return _build_fragmentation(mol, cut, cfg, cfg.method)


def fragment_smiles(smiles: str, cfg: FragmenterConfig) -> Fragmentation | None:
    """Convenience wrapper: parse SMILES, then fragment.  ``None`` if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return fragment_molecule(mol, cfg, mol_key=smiles)
