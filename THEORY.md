# Theoretical notes

Four propositions that make the empirical work sharper. Each is stated, proved,
and paired with the experiment that tests its prediction — the point is not
theory for its own sake but hypotheses that can be *refuted by measurement*.

Notation: a molecule is a graph $G = (V, E)$ with atom features $X$. A
fragmentation is a partition $\mathcal{S} = \{S_1, \dots, S_M\}$ of $V$. The
junction graph $G_f$ has one node per $S_k$ and an edge $(i,j)$ whenever some
$u \in S_i$, $v \in S_j$ have $(u,v) \in E$.

---

## Proposition 1 — Attention over a single key is the identity on its value

**Statement.** Let $a, b \in \mathbb{R}^{d}$ be vectors (not sequences) and let

$$O_{\text{attn}}(a,b) = \mathrm{softmax}\!\left(\frac{a W_Q (b W_K)^{\top}}{\sqrt{d_k}}\right) (b W_V).$$

Then $O_{\text{attn}}(a,b) = b W_V$ for all $a$, $W_Q$, $W_K$.

**Proof.** $a W_Q \in \mathbb{R}^{1 \times d}$ and $(b W_K)^\top \in \mathbb{R}^{d \times 1}$, so
the product is a $1 \times 1$ matrix, i.e. a scalar $s$. The softmax of a
one-element vector is $e^{s}/e^{s} = 1$. Hence the expression is
$1 \cdot bW_V = bW_V$. ∎

**Corollary.** The output is independent of $a$. If $a$ carries the drug and
$b$ the cell line, the operator discards the drug entirely.

**Consequence for DAF-DRP.** Equation 6 of the paper is exactly this form, with
$a$ and $b$ the sum-pooled view vectors. Whenever `GlobalFusion = A` — that is,
in configurations **WA, CA and AA** — the model's final representation is
$h_c W_V$, a function of the cell line alone.

**Test and result.** Verified numerically: feeding three different drugs with
the same cell line gives prediction spreads of $1.9\times10^{-9}$ (WA), $0$
(CA), $0$ (AA), against $5.2\times10^{-2}$ (CC), $7.0\times10^{-3}$ (WC),
$1.4\times10^{-3}$ (AC). Empirically, `DAF-DRP(BRICS)_WA` on GDSC2 LDO reaches
Pearson $0.175$ against the cell-line-mean baseline's $0.181$ — it reproduces
the naive predictor because it *is* one. Replacing Eq. 6 with a gated bilinear
interaction raises Pearson to $0.381$.

This matters because WA is the configuration DAF-DRP reports as best.

---

## Proposition 2 — Fragment features add no information, only inductive bias

**Statement.** Let $\phi$ be any fragmentation rule and $\psi$ any fragment
featuriser (physicochemical descriptors, SMARTS indicators, vocabulary index)
that are deterministic functions of $G$. Then
$\sigma(\phi(G), \psi(G)) \subseteq \sigma(G)$, so for any target $Y$,

$$I\big(Y;\, G, \phi(G), \psi(G)\big) = I(Y;\, G).$$

**Proof.** $\phi$ and $\psi$ are measurable functions of $G$, so the sigma-algebra
they generate is contained in $\sigma(G)$; conditioning on a function of $G$
given $G$ adds nothing. By the data-processing inequality the mutual
information is unchanged. ∎

**Consequence.** A fragment view cannot make an otherwise-sufficient model
better *in the infinite-data limit*. Whatever it contributes is **sample
complexity**: it hard-codes a function the network would otherwise have to
learn. Two predictions follow, and both can fail:

- **(P2a)** the advantage of fragment features over an atom-only encoder should
  **shrink as the number of training drugs grows**;
- **(P2b)** the advantage should be largest for functions that are *hard* for the
  atom encoder to learn — which Proposition 3 makes precise.

**Test.** P2a is experiment **E4** (drug-budget curve). P2b is experiment
**E6** below.

---

## Proposition 3 — The benefit should concentrate in *ring* functional groups

**Background.** GIN is exactly as expressive as the 1-dimensional
Weisfeiler–Lehman test (Xu et al. 2019). Chen et al. (2020) show that
message-passing GNNs and 2-WL **cannot induced-subgraph-count any connected
substructure of 3 or more nodes**, but **can subgraph-count star-shaped
substructures**.

**Statement (hypothesis H3).** Partition the functional-group vocabulary into

- $\mathcal{F}_{\text{star}}$ — groups whose SMARTS pattern is a tree with one
  central heavy atom (hydroxyl, nitrile, trifluoromethyl, primary amine,
  carbonyl, sulfone centre, halogen, …);
- $\mathcal{F}_{\text{ring}}$ — groups containing a cycle (benzene, pyridine,
  pyrimidine, quinazoline, indole, imidazole, morpholine, piperazine, …).

Then an atom-level MPNN can already detect $\mathcal{F}_{\text{star}}$ from the
graph, whereas $\mathcal{F}_{\text{ring}}$ membership is not in general
computable by it. Supplying explicit SMARTS features should therefore help
**substantially more for $\mathcal{F}_{\text{ring}}$ than for
$\mathcal{F}_{\text{star}}$**.

**Why this is worth testing.** It is the first prediction in this project that
distinguishes "functional groups help" from a *specific mechanism* for why, and
it is sharply falsifiable: if star groups help as much as ring groups, the
expressivity account is wrong and the benefit is something else (normalisation,
regularisation, extra parameters).

It also has a practical corollary: if the ring half carries the signal, then the
right prior is not "functional groups" in the medicinal-chemistry sense but
**ring systems** — a much smaller, cheaper feature set.

**Test.** Experiment **E6** below.

---

## Proposition 4 — Lookup-table fragments collide

**Statement.** Let a fragment branch embed fragment $k$ as $E[\iota(k)]$ where
$\iota$ maps a fragment to a vocabulary index, with index $0$ for
out-of-vocabulary, and let the readout be sum pooling. Then for two molecules
$G, G'$:

if the multiset of in-vocabulary fragment indices is equal,
$\{\!\{\iota(k) : \iota(k) \neq 0\}\!\} = \{\!\{\iota(k') : \iota(k') \neq 0\}\!\}$,
and the number of out-of-vocabulary fragments is equal,
$n_{\text{oov}}(G) = n_{\text{oov}}(G')$,

then $z_f(G) = z_f(G')$ **exactly**, for every parameter setting, no matter how
chemically different the out-of-vocabulary parts are.

**Proof.** $z_f = \sum_k E[\iota(k)] = \sum_{\iota(k)\neq 0} E[\iota(k)] + n_{\text{oov}} E[0]$.
Both terms are determined by the two quantities assumed equal. ∎

*(With a fragment GNN before the readout the statement holds whenever the
junction graphs are also isomorphic under the index labelling; the pure
sum-pooling case is the clean version and is what DAF-DRP's Eqs. 2–3 specify.)*

**Consequence.** The representation's resolution is bounded by vocabulary
coverage, and coverage is exactly what leave-drug-out destroys. Measured on
GDSC2 with a vocabulary mined per fold from the **training** drugs only:

| | value |
|---|---|
| fragments of a held-out drug that are in vocabulary | 69.2% |
| **atoms** of a held-out drug inside a covered fragment | **49.2%** |
| held-out drugs fully covered | 10.4% |

Roughly half of every new molecule is mapped to a single shared vector.

**Test.** Experiment **E7** measures the collision rate directly: how many pairs
of held-out drugs are provably indistinguishable to the fragment branch.

---

## What the propositions jointly predict

1. DAF-DRP's best configuration cannot use drug information (P1) — **confirmed**.
2. A lookup-table fragment representation should add ≈ 0 over an atom-only
   encoder under LDO (P4) — **consistent so far**: A6 vs A1 is $+0.0010$,
   $d = 0.06$.
3. The fragment view's benefit should be a sample-complexity effect, shrinking
   with more training drugs (P2a) — **E4 tests this**.
4. The benefit should be carried by ring groups, not star groups (P3) —
   **E6 tests this**.

Prediction 3 and 4 can both fail, and the report states whichever way they land.

---

## References

- Xu, K., Hu, W., Leskovec, J., Jegelka, S. *How Powerful are Graph Neural
  Networks?* ICLR 2019.
- Chen, Z., Chen, L., Villar, S., Bruna, J. *Can Graph Neural Networks Count
  Substructures?* NeurIPS 2020.
  <https://proceedings.neurips.cc/paper/2020/hash/75877cb75154206c4e65e76b88a12712-Abstract.html>
- Balcilar, M. et al. *Breaking the Limits of Message Passing Graph Neural
  Networks.* ICML 2021. <http://proceedings.mlr.press/v139/balcilar21a/balcilar21a.pdf>
- Huang, Y. et al. *Boosting the Cycle Counting Power of Graph Neural Networks
  with I²-GNNs.* ICLR 2023. <https://arxiv.org/html/2210.13978v1>
- Jiang, Y. et al. *Pharmacophoric-constrained heterogeneous graph transformer
  model for molecular property prediction.* Commun. Chem. 2023.
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10070395/>
- Kengkanna, A., Ohue, M. *Enhancing property and activity prediction and
  interpretation using multiple molecular graph representations with MMGX.*
  Commun. Chem. 2024. <https://www.nature.com/articles/s42004-024-01155-w>
- Degen, J. et al. *On the Art of Compiling and Using "Drug-Like" Chemical
  Fragment Spaces.* ChemMedChem 2008.
