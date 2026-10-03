"""DAF-DRP with BRICS — a faithful reimplementation of the paper's architecture.

Follows Doan & Le, *Dual-view Atomic and Fragment Graph Representation Learning
for Drug Response Prediction* (CIBB 2026), Section 2, with one substitution: the
fragment vocabulary comes from BRICS rather than Principal Subgraph Mining.

Architecture (paper Section 2, Figure 1)::

    AtomGraph  G_a ──► AtomGNN = SumPool(GIN_{θa}(G_a)) ──► z_a ∈ R^128
    FragGraph  G_f ──► FragGNN = SumPool(GIN_{θf}(G_f)) ──► z_f ∈ R^128
    expression x_c ──► FC_ge  ─────────────────────────────► h_c ∈ R^128

    h_final = O_global( O_drug(z_a, z_f), h_c )
    ŷ       = MLP_pred(h_final)

Two properties of the original are preserved deliberately, because changing
them would make this something other than a reproduction:

* **θ_a ≠ θ_f, and the two encoders never communicate.** The fragment branch
  sees only vocabulary indices; it never sees which atoms are inside a
  fragment. A benzene ring therefore receives the same embedding regardless of
  what it is attached to.
* **Fragment nodes are a learnable lookup table** of shape |S| x 128, indexed by
  the fragment's identity in the vocabulary, with row 0 for out-of-vocabulary.

The three fusion operators are Section 2.3 of the paper.  See
:class:`CrossAttentionFusion` for a note on Eq. 6, which is implemented exactly
as printed.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fragdrp.models.layers import scatter_sum

# --------------------------------------------------------------------------- #
# GIN (paper Section 2.1: plain GIN, not GINE)
# --------------------------------------------------------------------------- #


class GINLayer(nn.Module):
    """h_v' = MLP((1 + eps) h_v + sum_{u in N(v)} h_u)  — Xu et al. 2019."""

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.BatchNorm1d(2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
        )
        self.norm = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.dropout(F.relu(self.norm(self.mlp((1 + self.eps) * x))))
        aggregated = scatter_sum(x[edge_index[0]], edge_index[1], x.size(0))
        return self.dropout(F.relu(self.norm(self.mlp((1 + self.eps) * x + aggregated))))


class GIN(nn.Module):
    """A stack of ``n_layers`` GIN layers followed by sum pooling.

    ``SumPool`` is what the paper specifies (Eqs. 2-3); it is also what gives GIN
    its injectivity over multisets, so changing it would weaken the stated
    justification for choosing GIN.
    """

    def __init__(self, in_dim: int, dim: int, n_layers: int, dropout: float = 0.2):
        super().__init__()
        self.input_projection = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList(GINLayer(dim, dropout) for _ in range(n_layers))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, n_graphs: int
    ) -> torch.Tensor:
        h = self.input_projection(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        return scatter_sum(h, batch, n_graphs)


# --------------------------------------------------------------------------- #
# Fusion operators (paper Section 2.3)
# --------------------------------------------------------------------------- #


class ConcatFusion(nn.Module):
    """O_concat(a, b) = W_proj [a || b] + b_proj."""

    code = "C"

    def __init__(self, dim: int):
        super().__init__()
        self.projection = nn.Linear(2 * dim, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat([a, b], dim=-1))


class WeightedAdditionFusion(nn.Module):
    """O_alpha(a, b) = alpha * a + (1 - alpha) * b, with one globally shared scalar.

    The paper specifies alpha in [0, 1]; a sigmoid on an unconstrained parameter
    is the standard way to enforce that while keeping it learnable.
    """

    code = "W"

    def __init__(self, dim: int):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(1))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        return alpha * a + (1 - alpha) * b


class CrossAttentionFusion(nn.Module):
    """O_attn(a, b) = softmax(a W_Q (b W_K)^T / sqrt(d_k)) (b W_V)  — Eq. 6.

    Implemented exactly as printed, including a property worth stating plainly:
    ``a`` and ``b`` are single pooled vectors, so ``a W_Q (b W_K)^T`` is a 1x1
    **scalar**, and the softmax of a one-element vector is identically 1.  The
    operator therefore reduces to ``b W_V`` — a linear projection of the second
    input that discards the query entirely.

    This is kept faithful rather than silently repaired, because a reproduction
    that quietly fixes the thing being reproduced measures something else.  Pass
    ``fix_degenerate=True`` to use a gated bilinear interaction instead, which
    keeps both inputs and is a fair "what if it had attended" comparison.
    """

    code = "A"

    def __init__(self, dim: int, fix_degenerate: bool = False):
        super().__init__()
        self.scale = 1.0 / math.sqrt(dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.fix_degenerate = fix_degenerate
        if fix_degenerate:
            self.gate = nn.Linear(2 * dim, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        q, k, v = self.query(a), self.key(b), self.value(b)
        if not self.fix_degenerate:
            # score is (batch, 1); softmax over dim=-1 of a single element = 1.
            score = (q * k).sum(-1, keepdim=True) * self.scale
            weight = torch.softmax(score, dim=-1)
            return weight * v
        gate = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        return gate * v + (1 - gate) * q


FUSIONS = {"C": ConcatFusion, "W": WeightedAdditionFusion, "A": CrossAttentionFusion}

#: The six configurations reported in the paper (Tables 2-3).  First letter is
#: DrugFusion, second is GlobalFusion.
FUSION_CONFIGS = ["CC", "WC", "CA", "WA", "AC", "AA"]


def make_fusion(code: str, dim: int, fix_degenerate: bool = False) -> nn.Module:
    if code not in FUSIONS:
        raise ValueError(f"Fusion must be one of {sorted(FUSIONS)}, got {code!r}")
    if code == "A":
        return CrossAttentionFusion(dim, fix_degenerate=fix_degenerate)
    return FUSIONS[code](dim)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


class CellLineEncoder(nn.Module):
    """FC_ge, paper Eq. 4 — an MLP over the gene-expression profile."""

    def __init__(self, n_genes: int, dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, dim), nn.BatchNorm1d(dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DAFDRPBrics(nn.Module):
    """Dual-view atomic + fragment DRP model, BRICS vocabulary.

    Parameters
    ----------
    n_genes:
        Width of the expression vector.
    atom_feature_dim:
        Width of the per-atom feature vector.
    vocab_size:
        |S|; row 0 is the out-of-vocabulary embedding.
    drug_fusion, global_fusion:
        One of ``C`` (concatenation), ``W`` (weighted addition), ``A``
        (cross-attention).  Together they name the six configurations in the
        paper: CC, WC, CA, WA, AC, AA.
    """

    def __init__(
        self,
        n_genes: int,
        atom_feature_dim: int,
        vocab_size: int = 798,
        dim: int = 128,
        atom_layers: int = 5,
        frag_layers: int = 5,
        dropout: float = 0.2,
        drug_fusion: str = "W",
        global_fusion: str = "A",
        head_dims: Optional[List[int]] = None,
        fix_degenerate_attention: bool = False,
    ):
        super().__init__()
        self.config_code = f"{drug_fusion}{global_fusion}"
        self.dim = dim

        # Two structurally and parametrically independent encoders (theta_a != theta_f).
        self.atom_gnn = GIN(atom_feature_dim, dim, atom_layers, dropout)
        self.fragment_embedding = nn.Embedding(vocab_size, dim, padding_idx=None)
        self.frag_gnn = GIN(dim, dim, frag_layers, dropout)

        self.cell_encoder = CellLineEncoder(n_genes, dim, dropout)

        self.drug_fusion = make_fusion(drug_fusion, dim, fix_degenerate_attention)
        self.global_fusion = make_fusion(global_fusion, dim, fix_degenerate_attention)

        dims = [dim, *(head_dims or [64]), 1]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
        self.head = nn.Sequential(*layers)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        n = batch.n_graphs

        z_a = self.atom_gnn(batch.atom_x, batch.atom_edge_index, batch.atom_batch, n)

        frag_x = self.fragment_embedding(batch.frag_vocab_ids)
        z_f = self.frag_gnn(frag_x, batch.frag_edge_index, batch.frag_batch, n)

        h_c = self.cell_encoder(batch.cell)

        z_drug = self.drug_fusion(z_a, z_f)
        h_final = self.global_fusion(z_drug, h_c)
        return {"prediction": self.head(h_final).squeeze(-1)}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        note = ""
        if isinstance(self.drug_fusion, CrossAttentionFusion) or isinstance(
            self.global_fusion, CrossAttentionFusion
        ):
            degenerate = not getattr(self.drug_fusion, "fix_degenerate", False)
            note = "  [Eq.6 as printed: softmax over 1 key = 1]" if degenerate else "  [Eq.6 fixed]"
        return (
            f"DAF-DRP(BRICS)_{self.config_code}  dim={self.dim}  "
            f"params={self.count_parameters():,}{note}"
        )
