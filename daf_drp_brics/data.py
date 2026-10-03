"""Featurisation for DAF-DRP(BRICS).

Reuses the response loading, splitting and batching from ``fragdrp`` on purpose:
the point of this folder is to compare architectures, so the data, the folds and
the scalers must be byte-identical to the ones the other models see.  What is
built here is only the drug side, in the shape DAF-DRP expects:

* an **atom graph** with no edge features (the paper uses plain GIN);
* a **fragment graph** whose nodes are vocabulary *indices*, not feature
  vectors — the lookup-table representation being reproduced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger

from fragdrp.chem.featurizer import ATOM_FEATURE_DIM, atom_features
from fragdrp.config import FragmenterConfig

from fragment_vocab import FragmentVocabulary, decompose, fragment_smiles_of

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger(__name__)


@dataclass
class DualGraph:
    """One drug as DAF-DRP sees it: an atom graph plus an indexed fragment graph."""

    drug_id: str
    smiles: str
    atom_x: np.ndarray            # (n_atoms, ATOM_FEATURE_DIM)
    atom_edge_index: np.ndarray   # (2, 2 * n_bonds)
    frag_vocab_ids: np.ndarray    # (n_fragments,) indices into the lookup table
    frag_edge_index: np.ndarray   # (2, 2 * n_junctions)
    frag_smiles: List[str]
    n_oov: int
    """How many fragments fell outside the vocabulary — logged so the coverage
    problem is visible in the run rather than inferred afterwards."""

    @property
    def n_atoms(self) -> int:
        return int(self.atom_x.shape[0])

    @property
    def n_fragments(self) -> int:
        return int(self.frag_vocab_ids.shape[0])


def featurize(
    drug_id: str,
    smiles: str,
    vocabulary: FragmentVocabulary,
    cfg: Optional[FragmenterConfig] = None,
) -> Optional[DualGraph]:
    """Build both views for one drug."""
    cfg = cfg or FragmenterConfig(method="brics")
    decomposed = decompose(smiles, cfg)
    if decomposed is None:
        return None
    mol, fragmentation = decomposed

    atom_x = np.asarray([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    src: List[int] = []
    dst: List[int] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    atom_edge_index = (
        np.asarray([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    )

    names = fragment_smiles_of(mol, fragmentation)
    ids = np.asarray(vocabulary.encode(names), dtype=np.int64)

    # Eq. 1: fragments are adjacent when any atom of one bonds to any atom of the other.
    fsrc: List[int] = []
    fdst: List[int] = []
    for _ai, _aj, fi, fj in fragmentation.junction_bonds:
        fsrc += [fi, fj]
        fdst += [fj, fi]
    frag_edge_index = (
        np.asarray([fsrc, fdst], dtype=np.int64) if fsrc else np.zeros((2, 0), dtype=np.int64)
    )

    return DualGraph(
        drug_id=drug_id,
        smiles=smiles,
        atom_x=atom_x,
        atom_edge_index=atom_edge_index,
        frag_vocab_ids=ids,
        frag_edge_index=frag_edge_index,
        frag_smiles=names,
        n_oov=int((ids == 0).sum()),
    )


def build_drug_graphs(
    smiles_by_drug: Dict[str, str],
    vocabulary: FragmentVocabulary,
    cfg: Optional[FragmenterConfig] = None,
) -> Dict[str, DualGraph]:
    """Featurise every drug once, and report vocabulary coverage."""
    graphs: Dict[str, DualGraph] = {}
    for drug_id, smiles in smiles_by_drug.items():
        graph = featurize(drug_id, smiles, vocabulary, cfg)
        if graph is None:
            LOGGER.warning("Could not featurise %s (%s)", drug_id, smiles)
            continue
        graphs[drug_id] = graph

    if graphs:
        total = sum(g.n_fragments for g in graphs.values())
        oov = sum(g.n_oov for g in graphs.values())
        LOGGER.info(
            "Featurised %d/%d drugs | %.1f fragments/molecule | "
            "%d/%d fragments out-of-vocabulary (%.1f%%)",
            len(graphs), len(smiles_by_drug),
            total / max(len(graphs), 1), oov, total, 100 * oov / max(total, 1),
        )
    return graphs


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


@dataclass
class DualBatch:
    """Batched dual-view graphs, shaped for :class:`model.DAFDRPBrics`."""

    atom_x: "torch.Tensor"
    atom_edge_index: "torch.Tensor"
    atom_batch: "torch.Tensor"
    frag_vocab_ids: "torch.Tensor"
    frag_edge_index: "torch.Tensor"
    frag_batch: "torch.Tensor"
    cell: "torch.Tensor"
    target: "torch.Tensor"
    raw_response: "torch.Tensor"
    drug_ids: List[str]
    cell_line_ids: List[str]
    n_graphs: int

    def to(self, device) -> "DualBatch":
        import torch

        moved = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in self.__dict__.items()
        }
        return DualBatch(**moved)


def collate(items: Sequence[dict]) -> DualBatch:
    """Merge samples into one disconnected graph, offsetting node indices."""
    import torch

    atom_x, atom_ei, atom_batch = [], [], []
    frag_ids, frag_ei, frag_batch = [], [], []
    cells, targets, raws, drug_ids, cell_ids = [], [], [], [], []

    atom_offset = frag_offset = 0
    for index, item in enumerate(items):
        graph: DualGraph = item["graph"]
        atom_x.append(graph.atom_x)
        atom_ei.append(graph.atom_edge_index + atom_offset)
        atom_batch.append(np.full(graph.n_atoms, index, dtype=np.int64))

        frag_ids.append(graph.frag_vocab_ids)
        frag_ei.append(graph.frag_edge_index + frag_offset)
        frag_batch.append(np.full(graph.n_fragments, index, dtype=np.int64))

        cells.append(item["cell"])
        targets.append(item["target"])
        raws.append(item["raw_response"])
        drug_ids.append(item["drug_id"])
        cell_ids.append(item["cell_line_id"])

        atom_offset += graph.n_atoms
        frag_offset += graph.n_fragments

    def cat_edges(parts) -> "torch.Tensor":
        alive = [p for p in parts if p.size]
        if not alive:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.from_numpy(np.concatenate(alive, axis=1))

    return DualBatch(
        atom_x=torch.from_numpy(np.concatenate(atom_x, axis=0)),
        atom_edge_index=cat_edges(atom_ei),
        atom_batch=torch.from_numpy(np.concatenate(atom_batch)),
        frag_vocab_ids=torch.from_numpy(np.concatenate(frag_ids)),
        frag_edge_index=cat_edges(frag_ei),
        frag_batch=torch.from_numpy(np.concatenate(frag_batch)),
        cell=torch.from_numpy(np.stack(cells)),
        target=torch.tensor(targets, dtype=torch.float32),
        raw_response=torch.tensor(raws, dtype=torch.float32),
        drug_ids=drug_ids,
        cell_line_ids=cell_ids,
        n_graphs=len(items),
    )


class DualDataset:
    """One item per measured (drug, cell line) pair.

    Mirrors ``fragdrp.data.dataset.DrugResponseTorchDataset``: scalers are fit on
    the training fold and passed in, so no test statistic reaches training.
    """

    def __init__(self, data, rows, drug_graphs, cell_scaler, response_scaler):
        keep = [int(r) for r in rows if str(data.drug_ids[r]) in drug_graphs]
        self.rows = np.asarray(keep, dtype=np.int64)
        self.drug_graphs = drug_graphs
        self.drug_ids = [str(x) for x in data.drug_ids[self.rows]]
        self.cell_ids = [str(x) for x in data.cell_line_ids[self.rows]]
        self.raw_response = data.response[self.rows].astype(np.float32)
        self.target = response_scaler.transform(self.raw_response)
        self.cell_vectors = {
            c: cell_scaler.transform(data.cell_line_features[c]) for c in set(self.cell_ids)
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return {
            "graph": self.drug_graphs[self.drug_ids[index]],
            "cell": self.cell_vectors[self.cell_ids[index]],
            "target": self.target[index],
            "raw_response": self.raw_response[index],
            "drug_id": self.drug_ids[index],
            "cell_line_id": self.cell_ids[index],
        }


ATOM_DIM = ATOM_FEATURE_DIM
