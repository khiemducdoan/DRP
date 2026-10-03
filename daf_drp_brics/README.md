# DAF-DRP (BRICS)

A faithful reimplementation of **DAF-DRP** (Doan & Le, *Dual-view Atomic and
Fragment Graph Representation Learning for Drug Response Prediction*, CIBB 2026)
with exactly one substitution:

> **Principal Subgraph Mining → BRICS** for the fragment vocabulary.

Everything else follows the paper: two independent GIN encoders, a learnable
fragment lookup table, sum pooling, the three fusion operators, the six
configurations, and the Table 1 hyperparameters.

---

## What changed, and what did not

| | DAF-DRP (paper) | This folder |
|---|---|---|
| Fragment decomposition | Principal Subgraph Mining, 798 mined substructures | **BRICS**, 16 retrosynthetic rules |
| Fragment representation | learnable lookup table, index into vocabulary | *unchanged* |
| Atom encoder | GIN, 5 layers, SumPool | *unchanged* |
| Fragment encoder | GIN, 5 layers, SumPool, θ_f ≠ θ_a | *unchanged* |
| Cell line | MLP on gene expression | *unchanged* |
| Fusion | Concat / Weighted-add / Cross-attention, 6 configs | *unchanged* |
| Hyperparameters | Table 1 | *unchanged* (`config.yaml` annotates each) |

The two encoders still never communicate: the fragment branch sees only
vocabulary indices and never learns which atoms are inside a fragment. That is
the paper's design and it is preserved on purpose — changing it would make this
a different model rather than a reproduction. (The hierarchical alternative is
the `fragdrp` package one directory up.)

### Why BRICS is a reasonable swap

Principal Subgraph Mining needs a mining pass over a large external corpus and
yields a vocabulary specific to that corpus. BRICS is a fixed rule set that runs
on any molecule with no corpus at all, and its cuts fall on bonds a medicinal
chemist would disconnect. If the fragment view carries *chemistry* rather than
corpus statistics, the two should behave alike.

### What the swap does not fix

Both schemes address a fragment by an **index**, so a fragment missing from the
vocabulary collapses to a single shared out-of-vocabulary embedding. On GDSC2:

- BRICS yields **422 distinct fragments** across 241 drugs — comfortably under
  the 798 cap, so nothing is truncated;
- but **302 of those 422 occur exactly once**, so most rows of the lookup table
  receive gradient from a single molecule.

The run logs fragment coverage and the out-of-vocabulary rate before training,
so this is visible rather than inferred.

---

## Run

```bash
conda activate drp
cd daf_drp_brics

python train.py                                    # WA — the paper's best config
python train.py --drug-fusion C --global-fusion C  # CC
python run_all_fusions.py --dataset GDSC2 --test-mode LDO
python run_all_fusions.py --dataset GDSC2 --test-mode LPO   # the paper's protocol
```

Useful flags:

| Flag | Effect |
|---|---|
| `--test-mode {LPO,LCO,LDO}` | split protocol; see the note below |
| `--fix-attention` | use a non-degenerate cross-attention instead of Eq. 6 as printed |
| `--folds N` | number of CV folds |
| `--quick` | 5 epochs, 2 folds — a pipeline check, not a result |
| `--set a.b=value` | override any config key |

Data loading, folds, scalers and metrics are imported from the `fragdrp` package
one directory up, so results are computed on **byte-identical rows** to the other
models in this repository and can be put in the same table.

---

## Two things worth knowing before reading the numbers

### 1. Eq. 6 is degenerate as printed

The paper's cross-attention is

$$O_{\text{attn}}(a,b) = \mathrm{softmax}\!\left(\frac{aW_Q (bW_K)^\top}{\sqrt{d_k}}\right)(bW_V)$$

where `a` and `b` are **single pooled vectors** from SumPool. Then
`aW_Q (bW_K)ᵀ` is a 1×1 scalar, the softmax of a one-element vector is
identically 1, and the whole operator reduces to `bW_V` — a linear projection of
the second input that discards the query entirely.

This is implemented **as printed**, because a reproduction that silently repairs
the thing being reproduced measures something else. `--fix-attention` swaps in a
gated bilinear interaction that keeps both inputs, so the cost of the
degeneracy can be measured rather than argued about.

### 2. The protocol decides what the number means

The paper reports 10-fold cross-validation over drug–cell-line **pairs**. In that
protocol every test drug also appears in training, paired with hundreds of other
cell lines, so a model can score well by learning a per-drug mean and a
per-cell-line mean without consulting molecular structure at all — which means a
claim about *fragment representation* is not measurable there.

The default here is therefore `LDO` (leave-drug-out): the model must score a
molecule it has never been trained on. `--test-mode LPO` reproduces the paper's
setting, and running both is the informative comparison.

Under LDO, `NaiveDrugMeanPredictor` cannot score a held-out drug at all, and
`NaiveCellLineMeanPredictor` is the bar to clear. See `../EXPERIMENTS.md` §1.

---

## Files

```
config.yaml         Table 1 hyperparameters, each annotated with its source
fragment_vocab.py   BRICS vocabulary mining + coverage report  ← the substitution
data.py             atom graph + index-valued fragment graph, and batching
model.py            GIN encoders, the three fusion operators, the model
train.py            one configuration, all folds
run_all_fusions.py  all six configurations + paired significance tests
```

## Reference

Doan, D.-K. and Le, D.-H. *Dual-view Atomic and Fragment Graph Representation
Learning for Drug Response Prediction.* Proceedings of CIBB 2026.
Degen, J. et al. *On the Art of Compiling and Using "Drug-Like" Chemical Fragment
Spaces.* ChemMedChem 3(10):1503–1507, 2008.
