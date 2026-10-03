"""FragDRP: hierarchical fragment-aware drug response prediction.

Architecture
------------
::

    SMILES ──► atom graph ──► AtomGNN ────────────────────────► z_atom
                  │                 │
                  │ BRICS           │ attention pool per fragment
                  ▼                 ▼
            fragment partition ► fragment states h_f^(0)
                                  + descriptors
                                  + functional-group indicators
                                      │
                                      ▼
                              junction graph ► FragGNN ► h_f
                                                   │        │
                                        readout ◄──┘        │
                                           │                │
                                        z_frag        cell-conditioned
                                                        attention  ◄── h_c
                                                             │
                                                             ▼
                                                       z_frag|cell
    expression ──► CellEncoder ──► h_c

    ŷ = MLP([z_atom ; z_frag ; z_frag|cell ; h_c ; interaction])

Three differences from DAF-DRP that matter
------------------------------------------
1. **Hierarchical, not two independent towers.**  DAF-DRP encodes the atom
   graph and the fragment graph with separate GINs that never communicate, and
   initialises fragment nodes from a learnable lookup table over a mined
   798-fragment vocabulary.  A benzene ring therefore gets the same vector
   regardless of what is attached to it, and an unseen fragment gets nothing at
   all.  Here each fragment's initial state is pooled from *its own atoms after
   atom-level message passing*, so the representation is context-dependent and
   requires no vocabulary.

2. **Explicit, transferable chemical priors.**  Each fragment carries
   physicochemical descriptors and a functional-group SMARTS indicator vector.
   These are computable for any molecule, which is what carries the fragment
   view across the leave-drug-out boundary.

3. **Attention that is not a no-op.**  See
   :class:`~fragdrp.models.layers.ContextFragmentAttention`.

Every component is switchable from the config, because the ablation in
experiment E2 is the evidence for the hypothesis -- not a nice-to-have.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..chem.featurizer import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    JUNCTION_FEATURE_DIM,
    N_DESCRIPTORS,
)
from ..chem.functional_groups import N_FUNCTIONAL_GROUPS
from ..config import FeaturizerConfig, ModelConfig
from ..data.dataset import Batch
from .layers import (
    AttentionPool,
    ContextFragmentAttention,
    GNNStack,
    Readout,
    mlp,
    scatter_mean,
)


class CellLineEncoder(nn.Module):
    """Gene expression -> dense embedding.

    A plain MLP, matching what the DRP literature (GraphDRP, DeepCDR) uses, so
    that any difference between the models compared here comes from the drug
    branch rather than from a fancier omics encoder.
    """

    def __init__(self, in_dim: int, hidden_dims, dropout: float):
        super().__init__()
        self.net = mlp([in_dim, *hidden_dims], dropout=dropout, final_activation=True)
        self.out_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FragmentComposer(nn.Module):
    """Build each fragment's initial state.

    .. math::
        h_k^{(0)} = \\mathrm{LayerNorm}\\big(
            \\underbrace{\\mathrm{AttnPool}(\\{h_v : v \\in S_k\\})}_{\\text{from atoms}}
            + \\underbrace{\\mathrm{MLP}_d(d_k)}_{\\text{descriptors}}
            + \\underbrace{\\mathrm{MLP}_g(g_k)}_{\\text{functional groups}}
            + \\underbrace{E[\\mathrm{id}_k]}_{\\text{optional vocabulary}}\\big)

    The three terms are complementary: atoms give context, descriptors give a
    smooth chemical space in which two unseen fragments can be *similar*, and
    the functional-group vector gives a sparse, named prior.  The vocabulary
    term is off by default and exists only to reproduce DAF-DRP's design as a
    baseline.
    """

    def __init__(self, hidden_dim: int, model_cfg: ModelConfig, feat_cfg: FeaturizerConfig):
        super().__init__()
        self.cfg = model_cfg
        self.feat_cfg = feat_cfg

        self.atom_pool = AttentionPool(hidden_dim) if model_cfg.compose_from_atoms else None
        self.descriptor_mlp = (
            mlp([N_DESCRIPTORS, hidden_dim, hidden_dim],
                dropout=model_cfg.dropout, final_activation=False)
            if feat_cfg.use_descriptors
            else None
        )
        self.group_mlp = (
            mlp([N_FUNCTIONAL_GROUPS, hidden_dim, hidden_dim],
                dropout=model_cfg.dropout, final_activation=False)
            if feat_cfg.use_functional_groups
            else None
        )
        self.vocab_embedding = (
            nn.Embedding(feat_cfg.vocab_size, hidden_dim)
            if feat_cfg.use_fragment_vocab
            else None
        )
        # If every source is disabled the fragment view carries no information;
        # a learned constant keeps the shapes valid so the ablation still runs.
        self.fallback = nn.Parameter(torch.zeros(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, atom_h: torch.Tensor, batch: Batch, n_fragments: int) -> torch.Tensor:
        parts = []
        if self.atom_pool is not None:
            parts.append(self.atom_pool(atom_h, batch.atom_to_frag, n_fragments))
        if self.descriptor_mlp is not None:
            parts.append(self.descriptor_mlp(batch.frag_descriptors))
        if self.group_mlp is not None:
            parts.append(self.group_mlp(batch.frag_functional_groups))
        if self.vocab_embedding is not None:
            ids = batch.frag_vocab_ids.clamp(0, self.vocab_embedding.num_embeddings - 1)
            parts.append(self.vocab_embedding(ids))
        if not parts:
            parts.append(self.fallback.expand(n_fragments, -1))
        return self.norm(torch.stack(parts, dim=0).sum(dim=0))


class FragDRP(nn.Module):
    """The full model.  Ablations are the same class with flags flipped."""

    def __init__(self, n_genes: int, model_cfg: ModelConfig, feat_cfg: FeaturizerConfig):
        super().__init__()
        self.cfg = model_cfg
        self.feat_cfg = feat_cfg
        dim = model_cfg.hidden_dim

        if not (model_cfg.use_atom_view or model_cfg.use_frag_view):
            # The A0 ablation: the cell-line-only floor.  Anything a model
            # scores above this is what the drug representation is worth.
            pass

        # --- atom branch ---------------------------------------------------- #
        # The atom GNN also feeds the fragment composer, so it runs whenever
        # either view is on and composition from atoms is requested.
        self.needs_atom_gnn = model_cfg.use_atom_view or (
            model_cfg.use_frag_view and model_cfg.compose_from_atoms
        )
        self.atom_gnn = (
            GNNStack(ATOM_FEATURE_DIM, dim, BOND_FEATURE_DIM,
                     model_cfg.atom_layers, model_cfg.dropout)
            if self.needs_atom_gnn
            else None
        )
        self.atom_readout = (
            Readout(dim, model_cfg.atom_readout) if model_cfg.use_atom_view else None
        )

        # --- fragment branch ------------------------------------------------ #
        if model_cfg.use_frag_view:
            self.composer = FragmentComposer(dim, model_cfg, feat_cfg)
            self.frag_gnn = GNNStack(
                dim, dim, JUNCTION_FEATURE_DIM, model_cfg.frag_layers, model_cfg.dropout
            )
            self.frag_readout = Readout(dim, model_cfg.frag_readout)
        else:
            self.composer = self.frag_gnn = self.frag_readout = None  # type: ignore[assignment]

        # --- cell line ------------------------------------------------------ #
        self.cell_encoder = CellLineEncoder(n_genes, model_cfg.cell_hidden_dims, model_cfg.dropout)
        cell_dim = self.cell_encoder.out_dim

        # --- cell-conditioned fragment attention ---------------------------- #
        self.context_attention = (
            ContextFragmentAttention(dim, cell_dim, model_cfg.attention_heads, model_cfg.dropout)
            if (model_cfg.use_frag_view and model_cfg.use_cell_conditioned_attention)
            else None
        )

        # --- fusion + head --------------------------------------------------- #
        drug_dim = (
            (dim if model_cfg.use_atom_view else 0)
            + (dim if model_cfg.use_frag_view else 0)
            + (dim if self.context_attention is not None else 0)
        )
        self.drug_dim = drug_dim

        if drug_dim and model_cfg.interaction == "bilinear":
            # A low-rank bilinear term: project both sides into a shared space
            # and multiply, which lets the head express "this fragment matters
            # *because* this pathway is active" rather than only additively.
            self.drug_projection = nn.Linear(drug_dim, dim)
            self.cell_projection = nn.Linear(cell_dim, dim)
            fusion_dim = drug_dim + cell_dim + dim
        elif drug_dim and model_cfg.interaction == "hadamard":
            self.drug_projection = nn.Linear(drug_dim, cell_dim)
            self.cell_projection = None  # type: ignore[assignment]
            fusion_dim = cell_dim
        else:
            self.drug_projection = self.cell_projection = None  # type: ignore[assignment]
            fusion_dim = drug_dim + cell_dim

        self.head = mlp(
            [fusion_dim, *model_cfg.head_dims, 1],
            dropout=model_cfg.dropout,
            final_activation=False,
        )

    # ------------------------------------------------------------------ #

    def encode_drug(
        self,
        batch: Batch,
        cell_h: torch.Tensor,
        return_attention: bool = False,
        frag_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Drug embedding, optionally with per-fragment attention weights.

        ``frag_mask`` is ``(n_fragments_in_batch,)`` with 1 for keep and 0 for
        drop.  It silences a fragment *everywhere* in the fragment branch --
        after composition, after message passing, in the readout and in the
        attention -- which matters because zeroing a fragment's descriptors
        alone would not remove it: it would still receive the pooled states of
        its own atoms.  The masking intervention in
        :mod:`fragdrp.eval.attribution` depends on this being a real removal.

        Atom-level state is deliberately left untouched, so the mask isolates
        the *fragment view's* contribution rather than deleting part of the
        molecule.
        """
        n_graphs = batch.n_graphs
        n_fragments = int(batch.frag_descriptors.size(0))
        parts, attention = [], None

        atom_h = (
            self.atom_gnn(batch.atom_x, batch.atom_edge_index, batch.atom_edge_attr)
            if self.atom_gnn is not None
            else None
        )
        if self.atom_readout is not None:
            assert atom_h is not None
            parts.append(self.atom_readout(atom_h, batch.atom_batch, n_graphs))

        if self.cfg.use_frag_view:
            if atom_h is None:  # composition from atoms disabled
                atom_h = batch.atom_x.new_zeros((batch.atom_x.size(0), self.cfg.hidden_dim))
            frag_h = self.composer(atom_h, batch, n_fragments)

            mask = None
            if frag_mask is not None:
                mask = frag_mask.to(frag_h.dtype).unsqueeze(-1)
                frag_h = frag_h * mask

            frag_h = self.frag_gnn(frag_h, batch.frag_edge_index, batch.frag_edge_attr)
            if mask is not None:
                frag_h = frag_h * mask

            parts.append(self.frag_readout(frag_h, batch.frag_batch, n_graphs))
            if self.context_attention is not None:
                pooled, attention = self.context_attention(
                    frag_h, batch.frag_batch, cell_h,
                    return_attention=return_attention, frag_mask=frag_mask,
                )
                parts.append(pooled)

        if not parts:
            return cell_h.new_zeros((n_graphs, 0)), None
        return torch.cat(parts, dim=-1), attention

    def forward(
        self,
        batch: Batch,
        return_attention: bool = False,
        frag_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cell_h = self.cell_encoder(batch.cell)
        drug_h, attention = self.encode_drug(batch, cell_h, return_attention, frag_mask)

        if self.drug_dim == 0:
            fused = cell_h
        elif self.cfg.interaction == "bilinear":
            fused = torch.cat(
                [drug_h, cell_h, self.drug_projection(drug_h) * self.cell_projection(cell_h)],
                dim=-1,
            )
        elif self.cfg.interaction == "hadamard":
            fused = self.drug_projection(drug_h) * cell_h
        else:
            fused = torch.cat([drug_h, cell_h], dim=-1)

        out: Dict[str, torch.Tensor] = {"prediction": self.head(fused).squeeze(-1)}
        if return_attention and attention is not None:
            out["fragment_attention"] = attention
        return out

    # ------------------------------------------------------------------ #

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        views = []
        if self.cfg.use_atom_view:
            views.append(f"atom(L={self.cfg.atom_layers})")
        if self.cfg.use_frag_view:
            extras = []
            if self.cfg.compose_from_atoms:
                extras.append("composed")
            if self.feat_cfg.use_descriptors:
                extras.append("desc")
            if self.feat_cfg.use_functional_groups:
                extras.append("fgroups")
            if self.feat_cfg.use_fragment_vocab:
                extras.append("vocab")
            views.append(f"frag(L={self.cfg.frag_layers},{'+'.join(extras) or 'bare'})")
        if self.context_attention is not None:
            views.append(f"ctx-attn(h={self.cfg.attention_heads})")
        return (
            f"{self.cfg.name}[{' + '.join(views) or 'cell-only'}] "
            f"interaction={self.cfg.interaction} "
            f"params={self.count_parameters():,}"
        )


def build_model(n_genes: int, model_cfg: ModelConfig, feat_cfg: FeaturizerConfig) -> FragDRP:
    return FragDRP(n_genes=n_genes, model_cfg=model_cfg, feat_cfg=feat_cfg)
