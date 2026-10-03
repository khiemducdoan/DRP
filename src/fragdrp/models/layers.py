"""Graph neural-network building blocks, in plain PyTorch.

No PyTorch Geometric: everything needed here is a segment reduction over an
index vector, which ``torch.index_add_`` and ``Tensor.scatter_reduce_`` already
provide.  Keeping it dependency-free means the conda environment is
``torch + rdkit + the usual scientific stack`` and nothing that needs a C++
toolchain on Windows -- and the message passing is ~40 lines you can read.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Segment reductions
# --------------------------------------------------------------------------- #


def scatter_sum(src: torch.Tensor, index: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Sum rows of ``src`` into ``n_segments`` buckets given by ``index``."""
    out = src.new_zeros((n_segments, src.size(-1)))
    return out.index_add_(0, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n_segments: int) -> torch.Tensor:
    total = scatter_sum(src, index, n_segments)
    counts = src.new_zeros(n_segments).index_add_(
        0, index, src.new_ones(src.size(0))
    )
    return total / counts.clamp(min=1).unsqueeze(-1)


def scatter_max(src: torch.Tensor, index: torch.Tensor, n_segments: int) -> torch.Tensor:
    out = src.new_full((n_segments, src.size(-1)), float("-inf"))
    out = out.scatter_reduce(
        0, index.unsqueeze(-1).expand_as(src), src, reduce="amax", include_self=True
    )
    # Empty segments stay at -inf; zero them so downstream layers stay finite.
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, n_segments: int) -> torch.Tensor:
    """Softmax over each segment independently.

    ``scores`` is ``(n_nodes, n_heads)``.  The max subtraction is per segment,
    which keeps the exponentials in range even under mixed precision.
    """
    maxima = scatter_max(scores, index, n_segments)
    shifted = torch.exp(scores - maxima[index])
    denominator = scatter_sum(shifted, index, n_segments)[index]
    return shifted / denominator.clamp(min=1e-16)


# --------------------------------------------------------------------------- #
# Message passing
# --------------------------------------------------------------------------- #


class GINEConv(nn.Module):
    """Graph Isomorphism Network layer with edge features (Hu et al. 2020).

    .. math::
        h_v' = \\mathrm{MLP}\\big((1 + \\epsilon) h_v +
                 \\sum_{u \\in N(v)} \\mathrm{ReLU}(h_u + W_e e_{uv})\\big)

    GIN is used for the same reason DAF-DRP gives -- it is as expressive as the
    1-WL test, so it can tell apart molecular topologies that GCN collapses --
    but with edge features, because bond order is exactly the information a
    fragment view must not lose at the cut points.
    """

    def __init__(self, dim: int, edge_dim: int, dropout: float = 0.0, train_eps: bool = True):
        super().__init__()
        self.edge_encoder = nn.Linear(edge_dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.BatchNorm1d(2 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )
        self.eps = nn.Parameter(torch.zeros(1)) if train_eps else None

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            # A single-node graph (or a molecule with no surviving junctions):
            # no neighbours to aggregate, so the layer is just the MLP.
            eps = self.eps if self.eps is not None else 0.0
            return self.mlp((1 + eps) * x)

        source, target = edge_index[0], edge_index[1]
        messages = F.relu(x[source] + self.edge_encoder(edge_attr))
        aggregated = scatter_sum(messages, target, x.size(0))
        eps = self.eps if self.eps is not None else 0.0
        return self.mlp((1 + eps) * x + aggregated)


class GNNStack(nn.Module):
    """A residual stack of :class:`GINEConv` layers with batch norm."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        edge_dim: int,
        n_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_projection = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            GINEConv(hidden_dim, edge_dim, dropout) for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = self.input_projection(x)
        for conv, norm in zip(self.convs, self.norms):
            # Residual connections: with 4+ layers a plain GIN stack
            # over-smooths, and molecules are small enough that losing the
            # original atom identity hurts.
            h = h + self.dropout(F.relu(norm(conv(h, edge_index, edge_attr))))
        return h


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #


class AttentionPool(nn.Module):
    """Gated attention pooling (Ilse et al. 2018) over a segment.

    Used twice: to pool atoms into their fragment, and to pool fragments into a
    molecule vector.  The gate lets a large fragment be dominated by the two
    atoms that matter, which plain sum pooling cannot express.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Linear(dim, 1)
        self.gate = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, index: torch.Tensor, n_segments: int
    ) -> torch.Tensor:
        weights = scatter_softmax(self.attention(x), index, n_segments)
        return scatter_sum(weights * torch.sigmoid(self.gate(x)) * x, index, n_segments)


class Readout(nn.Module):
    """Pool node states into one vector per graph."""

    def __init__(self, dim: int, mode: str = "sum"):
        super().__init__()
        if mode not in ("sum", "mean", "max", "attention"):
            raise ValueError(f"Unknown readout {mode!r}")
        self.mode = mode
        self.pool = AttentionPool(dim) if mode == "attention" else None

    def forward(self, x: torch.Tensor, index: torch.Tensor, n_segments: int) -> torch.Tensor:
        if self.mode == "sum":
            return scatter_sum(x, index, n_segments)
        if self.mode == "mean":
            return scatter_mean(x, index, n_segments)
        if self.mode == "max":
            return scatter_max(x, index, n_segments)
        assert self.pool is not None
        return self.pool(x, index, n_segments)


# --------------------------------------------------------------------------- #
# Cell-line-conditioned attention over fragments
# --------------------------------------------------------------------------- #


class ContextFragmentAttention(nn.Module):
    """Let a cell line attend over the *set* of fragment tokens of its drug.

    .. math::
        \\alpha_k = \\mathrm{softmax}_k\\!\\left(
            \\frac{(W_Q h_c)^\\top (W_K h_{f_k})}{\\sqrt{d_h}}\\right),
        \\qquad
        z = \\sum_k \\alpha_k\\, W_V h_{f_k}

    Two things distinguish this from the "cross-attention" fusion in DAF-DRP
    (their Eq. 6), which attends between two *pooled* vectors.  With a single
    key, the softmax is identically 1, so that operator reduces to
    :math:`b W_V` -- a linear projection that discards the query entirely.  Here
    the keys are the individual fragments, so:

    * the softmax is over many tokens and actually selects;
    * :math:`\\alpha_k` is a per-fragment importance for a *specific* cell line,
      which is the quantity experiment E5 inspects.

    The attention runs as one flat segment-softmax over the batch, so molecules
    of different sizes need no padding.
    """

    def __init__(self, frag_dim: int, cell_dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if frag_dim % n_heads:
            raise ValueError(f"frag_dim {frag_dim} must be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = frag_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(cell_dim, frag_dim)
        self.key = nn.Linear(frag_dim, frag_dim)
        self.value = nn.Linear(frag_dim, frag_dim)
        self.out = nn.Linear(frag_dim, frag_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        frag_h: torch.Tensor,
        frag_batch: torch.Tensor,
        cell_h: torch.Tensor,
        return_attention: bool = False,
        frag_mask: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        frag_h: ``(n_fragments_in_batch, frag_dim)``
        frag_batch: ``(n_fragments_in_batch,)`` sample index of each fragment
        cell_h: ``(batch, cell_dim)``
        frag_mask: optional ``(n_fragments_in_batch,)`` of 1 for keep, 0 for drop.
            Masked fragments are pushed out of the softmax rather than merely
            zeroed, because ``W_V`` has a bias and would otherwise leak a
            constant through for a fragment that is supposed to be absent.
            Used by the attention-masking intervention in
            :mod:`fragdrp.eval.attribution`.

        Returns
        -------
        ``(batch, frag_dim)``, and optionally ``(n_fragments_in_batch, n_heads)``
        attention weights.
        """
        n_graphs = cell_h.size(0)
        n_frags = frag_h.size(0)

        q = self.query(cell_h).view(n_graphs, self.n_heads, self.head_dim)
        k = self.key(frag_h).view(n_frags, self.n_heads, self.head_dim)
        v = self.value(frag_h).view(n_frags, self.n_heads, self.head_dim)

        # Each fragment scores only against its own molecule's cell line.
        scores = (q[frag_batch] * k).sum(-1) * self.scale          # (n_frags, heads)
        if frag_mask is not None:
            scores = scores.masked_fill(frag_mask.unsqueeze(-1) < 0.5, -1e4)
        weights = scatter_softmax(scores, frag_batch, n_graphs)
        if frag_mask is not None:
            weights = weights * frag_mask.unsqueeze(-1)
        pooled = scatter_sum(
            (self.dropout(weights).unsqueeze(-1) * v).reshape(n_frags, -1),
            frag_batch,
            n_graphs,
        )
        out = self.out(pooled)
        return (out, weights) if return_attention else (out, None)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #


def mlp(dims, dropout: float = 0.0, final_activation: bool = False) -> nn.Sequential:
    """A plain MLP with BatchNorm + ReLU + Dropout between linear layers."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_activation:
            layers += [nn.BatchNorm1d(dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)
