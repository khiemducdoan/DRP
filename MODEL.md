# FragDRP — model description

A drug-response model that represents a molecule hierarchically: atoms first,
then the chemical fragments those atoms compose, then a fragment-level
interaction with the cell line's transcriptome.

The idea it is built on is DAF-DRP's: *functional groups, not individual atoms,
are the units that determine chemical activity, so a drug encoder should see
them.* What changes here is the machinery needed to actually test that idea —
a fragment representation that survives contact with a molecule the model has
never seen, and an attention mechanism that is not an identity map.

---

## 1. Problem

Given a drug $d$ and a cancer cell line $c$, predict the measured response
$y \in \mathbb{R}$ (log IC50 or AUC as the benchmark ships it; lower means more
sensitive). We want $f(d, c; \theta) \to \hat{y}$ that works when **$d$ was
never seen during training** — the leave-drug-out setting, which is where a
claim about drug representation is falsifiable at all. See `EXPERIMENTS.md` §1
for why that matters more than it might appear.

---

## 2. Architecture

```
SMILES ──► atom graph G_a ──► AtomGNN ─────────────────────────────► z_atom
               │                  │
               │ BRICS            │ attention-pool over each fragment
               ▼                  ▼
        fragment partition ──► h_f^(0) ◄── physchem descriptors d_k
        {S_1 … S_M}                    ◄── functional-group indicators g_k
               │                          │
               ▼                          ▼
        junction graph G_f ──────► FragGNN ──► h_f
                                      │           │
                            readout ◄─┘           │
                               │             cell-conditioned
                            z_frag             attention  ◄──┐
                                                   │         │
                                                   ▼         │
                                            z_frag|cell      │
                                                             │
expression x_c ──► CellEncoder ──────────────────────────► h_c

ŷ = MLP([ z_atom ; z_frag ; z_frag|cell ; h_c ; W_d z_drug ⊙ W_c h_c ])
```

Implementation: [`src/fragdrp/models/fragdrp.py`](src/fragdrp/models/fragdrp.py),
with the graph primitives in [`layers.py`](src/fragdrp/models/layers.py).

### 2.1 Atom encoder

Standard message passing over the molecular graph. Atom features are the usual
one-hot block (element, degree, formal charge, implicit H count, hybridisation,
chirality, aromaticity, ring membership by size, mass); bonds carry type,
stereo, conjugation and ring membership.

$L_a$ layers of **GINE** — GIN with edge features:

$$h_v^{(l+1)} = \mathrm{MLP}^{(l)}\Big((1+\epsilon)\,h_v^{(l)} + \sum_{u \in \mathcal{N}(v)} \mathrm{ReLU}\big(h_u^{(l)} + W_e\, e_{uv}\big)\Big)$$

with batch norm and a residual connection per layer. GIN is used for the reason
DAF-DRP gives — it is as expressive as the 1-WL test, so it distinguishes
topologies that GCN collapses — but with edge features, because bond order at
the cut points is exactly what a fragment view must not lose.

### 2.2 Fragmentation

A **fragmenter** answers one question: which bonds do we cut? Fragments are then
the connected components of the molecule with those bonds removed, and the
junction graph has one node per fragment and one edge per cut bond. Putting the
whole family behind that single decision is what makes the strategies
comparable — `EXPERIMENTS.md` §4 swaps the strategy and changes nothing else.

The default is **BRICS** (Degen et al. 2008): 16 retrosynthetic rules encoding
bonds a medicinal chemist would disconnect, so the resulting pieces tend to be
functional groups and ring systems rather than arbitrary chunks. On GDSC2's
drugs it yields ≈ 4.7 fragments per molecule.

Also implemented, and all directly comparable:
**RECAP** (11 more conservative cleavage rules), **Murcko** (ring systems vs
side chains), **rotatable-bond**, and three controls — **random_cut**,
**whole**, **atom** — described in `EXPERIMENTS.md` §4.

Two post-processing rules keep the junction graph well-behaved: fragments below
`min_fragment_atoms` merge into their largest neighbour, and merging continues
until at most `max_fragments` remain. Salts are stripped first (largest
connected component only) so a counter-ion cannot become a fragment.

Example — BRICS on axitinib:

| Fragment | SMILES | Functional groups detected |
|---|---|---|
| 0 | `CN` | primary amine |
| 1 | `C=O` | — |
| 2 | `c1ccccc1` | benzene |
| 3 | `Cc1n[nH]c2cc(S)ccc12` | indazole, pyrazole, benzene, thioether |
| 4 | `Cc1ccccn1` | pyridine |

### 2.3 Fragment composition — the part that matters

This is the main departure from DAF-DRP, and the reason the model can work on
unseen drugs.

DAF-DRP represents fragment $k$ as an index into a vocabulary of 798 mined
substructures and learns an embedding per index. Two consequences follow, and
both bite hardest exactly where the hypothesis needs to be tested:

1. **A fragment outside the vocabulary has no representation.** DAF-DRP falls
   back to treating its atoms as singletons. This is not a corner case: mining a
   vocabulary from the training drugs of each GDSC2 leave-drug-out fold and
   applying it to that fold's held-out drugs covers only **69% of their
   fragments and 49% of their atoms**, with just **10% of held-out drugs covered
   completely** (`results/ablation/vocabulary_coverage_GDSC2_LDO.csv`). Roughly
   half of every new molecule is invisible to a lookup-table encoder.
2. **A vocabulary index carries no chemistry.** Two fragments that differ by one
   methyl get unrelated embeddings; the same benzene ring gets an identical
   vector no matter what it is attached to.

FragDRP computes the fragment state instead, from three sources that exist for
*any* fragment:

$$h_k^{(0)} = \mathrm{LayerNorm}\Big(\underbrace{\mathrm{AttnPool}\big(\{h_v : v \in S_k\}\big)}_{\text{from its own atoms}} + \underbrace{\mathrm{MLP}_d(d_k)}_{\text{descriptors}} + \underbrace{\mathrm{MLP}_g(g_k)}_{\text{functional groups}}\Big)$$

- **Atom pooling** — gated attention over the atoms inside the fragment, *after*
  atom-level message passing. The same ring in two molecules therefore gets two
  different vectors, reflecting its substituents.
- **$d_k$ — 18 physicochemical descriptors** computed on the fragment
  sub-molecule: heavy atoms, MW, cLogP, molar refractivity, TPSA, HBD, HBA,
  rings, aromatic rings, rotatable bonds, fraction sp³, formal charge, halogen
  count, N/O/S counts, aromatic atoms, heteroatom fraction. These give a
  *smooth* space in which two fragments the model has never seen can still be
  similar.
- **$g_k$ — a 61-dimensional functional-group indicator**, one bit per SMARTS
  pattern from a curated pharmacophore list
  ([`functional_groups.py`](src/fragdrp/chem/functional_groups.py)): hinge-binding
  heterocycles (quinazoline, indazole, purine-like, imidazole…), solubilising
  amines (piperazine, morpholine…), acids and bases, carbonyl chemistry
  (amide, sulfonamide, urea, carbamate…), halogen handles, covalent warheads.
  This is the explicit chemical prior the hypothesis is about, and it is
  computed by substructure match, so it works on any molecule.

None of the three requires the fragment to have been seen in training. The
vocabulary path is still implemented (`featurizer.use_fragment_vocab`) — it is
ablation **A6**, the DAF-DRP-style representation, kept so the report can
measure what that choice costs rather than assert it.

### 2.4 Fragment encoder

$L_f$ GINE layers over the junction graph. Junction edges carry the cut bond's
own features plus a flag marking them as cuts, so the encoder knows an aryl–aryl
junction from an amide junction.

### 2.5 Cell-line-conditioned fragment attention

The cell line queries the **set** of fragment tokens:

$$\alpha_k = \mathrm{softmax}_k\left(\frac{(W_Q h_c)^\top (W_K h_{f_k})}{\sqrt{d_h}}\right), \qquad z_{\text{frag}|c} = \sum_k \alpha_k\, W_V h_{f_k}$$

4 heads, run as one flat segment-softmax over the batch so molecules of
different sizes need no padding.

**Why this differs from DAF-DRP's "cross-attention" (their Eq. 6).** That
operator attends between two *pooled* vectors. With a single key, the softmax is
identically 1, and the expression collapses to $b W_V$ — a linear projection of
the value input that discards the query entirely. Whatever the reported gain
from their attention-fusion configurations, it is not attention.

Here the keys are the individual fragments, so the softmax has something to
choose between. Two things follow:

- the operator can express "for *this* cell line, *this* part of the molecule
  matters", which is the drug × cell-line interaction the task is about;
- $\alpha_k$ is a per-fragment attribution, which is what makes the
  interpretability experiment (`EXPERIMENTS.md` §6) possible — including the
  masking test that can show the attention is decorative.

### 2.6 Cell-line encoder and prediction head

Gene expression (LINCS L1000 landmark genes by default) through an MLP
`n_genes → 512 → 256 → 128` with batch norm and dropout. Deliberately plain and
held fixed across every variant, so differences in the results table come from
the drug branch.

Fusion is concatenation plus a low-rank bilinear term
$W_d z_{\text{drug}} \odot W_c h_c$, which lets the head express "this fragment
matters *because* this pathway is active" rather than only additively. Head:
`→ 256 → 128 → 1`, trained with MSE.

---

## 3. Default configuration

| Component | Value |
|---|---|
| Atom GNN layers $L_a$ | 4 |
| Fragment GNN layers $L_f$ | 3 |
| Hidden dimension | 128 |
| Attention heads | 4 |
| Fragment descriptors | 18 |
| Functional-group patterns | 61 |
| Max fragments / molecule | 32 |
| Cell-line MLP | 512 → 256 → 128 |
| Prediction head | 256 → 128 → 1 |
| Dropout | 0.3 |
| Optimiser | AdamW, lr 1.5e-3, weight decay 1e-4 |
| Schedule | 4-epoch warmup, then cosine |
| Batch size | 512, mixed precision |
| Early stopping | validation MSE, patience 12 |
| Parameters | ≈ 1.2 M |

Every ablation is this same class with config flags flipped — see
[`configs/model/`](configs/model/). Nothing is re-tuned per variant, which keeps
the comparison fair at the cost of no single row being that architecture's best
achievable number.

---

## 4. Implementation notes

**No PyTorch Geometric.** Message passing needs only a segment reduction over an
index vector, which `torch.index_add_` and `scatter_reduce_` already provide.
The primitives are ~40 readable lines in
[`layers.py`](src/fragdrp/models/layers.py), and the environment stays
`torch + rdkit + the usual scientific stack` with nothing requiring a C++
toolchain on Windows.

**Scaling is fit per fold, on training rows only.** Standardising the expression
matrix once up front — the common shortcut — leaks the test distribution's mean
and variance. Predictions are inverted back to the benchmark's own scale before
any metric is computed.

**Drugs are featurised once, not once per row.** GDSC2 has ~77k rows but only
~241 distinct drugs; the RDKit work is cached per drug, keyed by the fragmenter
and featuriser config so switching BRICS→RECAP rebuilds rather than silently
reusing.

**Leakage is asserted.** The splitter raises if a held-out drug, cell line or
tissue appears in training, rather than letting a good-looking number through.

---

## 5. What this model does *not* do

- It uses gene expression only. Mutation, copy number, methylation and
  proteomics views ship with the benchmark and are unused — the cell-line side
  is kept simple on purpose so it stays constant while the drug side varies.
- It is trained and evaluated within a dataset. Cross-dataset transfer is the
  obvious next experiment and is not claimed here.
- The functional-group list is curated and finite. It covers common medicinal
  chemistry, not all of it, and a molecule outside that chemistry falls back on
  the descriptor and atom-pooling terms.
- It optimises MSE, not a ranking objective, even though the metric that matters
  clinically is a within-cell-line ranking over drugs.

---

## 6. References

- Degen et al., *On the Art of Compiling and Using "Drug-Like" Chemical Fragment
  Spaces*, ChemMedChem 2008 — BRICS.
- Lewell et al., *RECAP — Retrosynthetic Combinatorial Analysis Procedure*,
  JCICS 1998.
- Bemis & Murcko, *The Properties of Known Drugs. 1. Molecular Frameworks*,
  J. Med. Chem. 1996.
- Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019 — GIN.
- Hu et al., *Strategies for Pre-training Graph Neural Networks*, ICLR 2020 — GINE.
- Ilse et al., *Attention-based Deep Multiple Instance Learning*, ICML 2018 —
  gated attention pooling.
- Nguyen et al., *Graph Convolutional Networks for Drug Response Prediction*,
  IEEE/ACM TCBB 2021 — GraphDRP.
- Liu et al., *DeepCDR*, Bioinformatics 2020.
- drevalpy / nf-core/drugresponseeval — benchmark datasets, splits, baselines
  and metrics.
