"""Fragment-level attribution from the cell-line-conditioned attention.

The attention weights of
:class:`~fragdrp.models.layers.ContextFragmentAttention` are a per-fragment
importance for a *specific* cell line.  That makes them checkable in a way a
scalar fusion weight never is: if the hypothesis is right -- that the model
benefits from functional-group structure -- then the fragments it attends to
should be the ones a medicinal chemist would point at, and the pattern should
shift with the cell line's biology.

Three analyses live here, in increasing strength of evidence:

``fragment_attention_table``
    Raw per-(drug, cell line, fragment) weights.

``functional_group_importance``
    Attention mass redistributed onto functional groups.  Answers "which groups
    does this model weight, across the whole test set?"

``attention_ablation_curve``
    The falsifiable one.  Masking the top-attended fragment should hurt
    predictions more than masking a random fragment.  If it does not, the
    attention is decorative and the attribution story is unsupported -- so this
    test can fail, and the report says so if it does.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..chem import functional_groups as fg
from ..chem.featurizer import DrugGraph
from ..data.dataset import Batch, DrugResponseTorchDataset, make_loader
from ..train.metrics import compute_metrics

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def fragment_attention_table(
    model,
    dataset: DrugResponseTorchDataset,
    device: torch.device | str = "cuda",
    batch_size: int = 256,
    max_batches: Optional[int] = None,
) -> pd.DataFrame:
    """Per-fragment attention weights over a dataset.

    Returns one row per (drug, cell line, fragment) with the head-averaged
    attention weight, the fragment SMILES and its functional groups.
    """
    model = model.to(device).eval()
    loader = make_loader(dataset, batch_size, shuffle=False)
    rows: List[Dict[str, object]] = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch.to(device)
        out = model(batch, return_attention=True)
        if "fragment_attention" not in out:
            LOGGER.warning("Model has no cell-conditioned attention; nothing to attribute")
            return pd.DataFrame()

        weights = out["fragment_attention"].float().cpu().numpy()   # (n_frags, heads)
        frag_batch = batch.frag_batch.cpu().numpy()
        predictions = out["prediction"].float().cpu().numpy()

        # Recover each fragment's position inside its own molecule.
        position = np.zeros(len(frag_batch), dtype=int)
        seen: Dict[int, int] = {}
        for i, sample in enumerate(frag_batch):
            position[i] = seen.get(int(sample), 0)
            seen[int(sample)] = position[i] + 1

        for i, sample in enumerate(frag_batch):
            sample = int(sample)
            graph: DrugGraph = dataset.drug_graphs[batch.drug_ids[sample]]
            k = int(position[i])
            smiles = graph.frag_smiles[k] if k < len(graph.frag_smiles) else ""
            groups = (
                [n for n, v in zip(fg.GROUP_NAMES, graph.frag_functional_groups[k]) if v > 0]
                if k < graph.n_fragments
                else []
            )
            rows.append(
                {
                    "drug_id": batch.drug_ids[sample],
                    "cell_line_id": batch.cell_line_ids[sample],
                    "fragment_index": k,
                    "fragment_smiles": smiles,
                    "attention": float(weights[i].mean()),
                    "attention_max_head": float(weights[i].max()),
                    "n_fragments": graph.n_fragments,
                    "functional_groups": "|".join(groups),
                    "prediction": float(predictions[sample]),
                }
            )
    return pd.DataFrame(rows)


def functional_group_importance(attention: pd.DataFrame) -> pd.DataFrame:
    """Attention mass attributed to each functional group.

    A fragment's weight is split evenly across the groups it contains, so a
    fragment carrying three groups does not count triple.  The key column is
    ``enrichment``: mean attention on fragments containing the group, divided by
    what uniform attention would give.  Values above 1 mean the model
    preferentially attends to that group.
    """
    if attention.empty:
        return pd.DataFrame()

    # Uniform attention would give 1/n_fragments to each fragment; comparing
    # against that makes molecules with different fragment counts comparable.
    frame = attention.copy()
    frame["uniform"] = 1.0 / frame["n_fragments"].clip(lower=1)
    frame["relative_attention"] = frame["attention"] / frame["uniform"]

    records: List[Dict[str, object]] = []
    for group in fg.GROUP_NAMES:
        mask = frame["functional_groups"].str.contains(rf"(?:^|\|){group}(?:\||$)", regex=True)
        present = frame[mask]
        if present.empty:
            continue
        records.append(
            {
                "functional_group": group,
                "n_fragments": int(present["fragment_index"].count()),
                "n_drugs": int(present["drug_id"].nunique()),
                "mean_attention": float(present["attention"].mean()),
                "enrichment": float(present["relative_attention"].mean()),
                "enrichment_sd": float(present["relative_attention"].std(ddof=1))
                if len(present) > 1
                else 0.0,
            }
        )
    out = pd.DataFrame(records)
    return out.sort_values("enrichment", ascending=False).reset_index(drop=True) if not out.empty else out


def top_fragments_per_drug(attention: pd.DataFrame, top_k: int = 1) -> pd.DataFrame:
    """The most-attended fragment(s) of each drug, averaged over cell lines."""
    if attention.empty:
        return pd.DataFrame()
    per_fragment = (
        attention.groupby(["drug_id", "fragment_index", "fragment_smiles", "functional_groups"])
        .agg(mean_attention=("attention", "mean"), n_cell_lines=("cell_line_id", "nunique"))
        .reset_index()
    )
    return (
        per_fragment.sort_values(["drug_id", "mean_attention"], ascending=[True, False])
        .groupby("drug_id")
        .head(top_k)
        .reset_index(drop=True)
    )


def attention_specificity(attention: pd.DataFrame) -> pd.DataFrame:
    """How much does a drug's attention pattern change across cell lines?

    If the model is genuinely conditioning on the cell line, the weights for one
    drug should vary between cell lines.  If they do not, the attention has
    collapsed into a drug-only saliency map -- still useful, but a weaker claim,
    and the report should say which of the two it is.
    """
    if attention.empty:
        return pd.DataFrame()
    records = []
    for drug, block in attention.groupby("drug_id"):
        pivot = block.pivot_table(
            index="cell_line_id", columns="fragment_index", values="attention"
        ).dropna(axis=1, how="any")
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            continue
        records.append(
            {
                "drug_id": drug,
                "n_cell_lines": int(pivot.shape[0]),
                "n_fragments": int(pivot.shape[1]),
                # Between-cell-line sd of each fragment's weight, averaged.
                "across_cell_line_sd": float(pivot.std(axis=0, ddof=1).mean()),
                # Spread across fragments within a cell line, for scale.
                "across_fragment_sd": float(pivot.std(axis=1, ddof=1).mean()),
            }
        )
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["conditioning_ratio"] = (
            frame["across_cell_line_sd"] / frame["across_fragment_sd"].clip(lower=1e-9)
        )
    return frame


@torch.no_grad()
def attention_ablation_curve(
    model,
    dataset: DrugResponseTorchDataset,
    device: torch.device | str = "cuda",
    batch_size: int = 256,
    n_random_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Does masking the most-attended fragment hurt more than masking a random one?

    A fragment is removed through the model's ``frag_mask``, which silences it
    after composition, after message passing, in the readout and in the
    attention.  The atom branch is untouched, so the intervention isolates the
    *fragment view's* contribution rather than deleting part of the molecule --
    and the comparison stays within-molecule, against a random fragment of the
    same drug, so molecule size cannot drive the result.

    This is the falsifiable form of the interpretability claim.  Three outcomes
    are possible and all are reported honestly:

    * top-masking hurts more than random-masking -> the attention identifies
      fragments the prediction actually depends on;
    * the two are equal -> the attention is decorative;
    * masking barely matters at all -> the fragment branch is not carrying the
      prediction, whatever the attention says.
    """
    model = model.to(device).eval()
    loader = make_loader(dataset, batch_size, shuffle=False)
    rng = np.random.default_rng(seed)
    records: List[Dict[str, object]] = []

    def predict(batch: Batch, victims: Optional[np.ndarray] = None) -> np.ndarray:
        """Predictions, optionally with the listed global fragment indices removed."""
        mask = None
        if victims is not None:
            mask = torch.ones(batch.frag_descriptors.size(0), device=device)
            mask[torch.as_tensor(victims, device=device, dtype=torch.long)] = 0.0
        return model(batch, frag_mask=mask)["prediction"].float().cpu().numpy()

    for batch in loader:
        batch = batch.to(device)
        out = model(batch, return_attention=True)
        if "fragment_attention" not in out:
            return pd.DataFrame()

        baseline = out["prediction"].float().cpu().numpy()
        weights = out["fragment_attention"].float().mean(-1).cpu().numpy()
        frag_batch = batch.frag_batch.cpu().numpy()

        # Per molecule: the globally-indexed fragments, and the top-attended one.
        groups: Dict[int, List[int]] = {}
        for i, sample in enumerate(frag_batch):
            groups.setdefault(int(sample), []).append(i)
        multi = {s: idx for s, idx in groups.items() if len(idx) > 1}
        if not multi:
            continue

        top_victims = np.array([max(idx, key=lambda i: weights[i]) for idx in multi.values()])
        top_pred = predict(batch, top_victims)

        random_deltas = []
        for repeat in range(n_random_repeats):
            victims = np.array(
                [idx[rng.integers(len(idx))] for idx in multi.values()]
            )
            random_deltas.append(np.abs(predict(batch, victims) - baseline))
        random_delta = np.mean(random_deltas, axis=0)

        samples = np.array(sorted(multi))
        for position, sample in enumerate(samples):
            records.append(
                {
                    "drug_id": batch.drug_ids[sample],
                    "cell_line_id": batch.cell_line_ids[sample],
                    "target": float(batch.raw_response[sample]),
                    "baseline_prediction": float(baseline[sample]),
                    "delta_top": float(abs(top_pred[sample] - baseline[sample])),
                    "delta_random": float(random_delta[sample]),
                    "n_fragments": len(multi[sample]),
                }
            )
    return pd.DataFrame(records)


def summarise_ablation(curve: pd.DataFrame) -> Dict[str, float]:
    """Paired comparison of top-fragment vs random-fragment masking."""
    from scipy import stats as sps

    if curve.empty:
        return {}
    top, random_ = curve["delta_top"].to_numpy(), curve["delta_random"].to_numpy()
    difference = top - random_
    result = {
        "n": int(len(curve)),
        "mean_delta_top": float(top.mean()),
        "mean_delta_random": float(random_.mean()),
        "mean_difference": float(difference.mean()),
        "ratio": float(top.mean() / random_.mean()) if random_.mean() > 1e-12 else float("nan"),
    }
    if len(curve) > 1 and not np.allclose(difference, 0):
        result["p_wilcoxon"] = float(sps.wilcoxon(difference).pvalue)
        sd = float(np.std(difference, ddof=1))
        result["cohens_d"] = float(difference.mean() / sd) if sd > 1e-12 else float("nan")
    return result
