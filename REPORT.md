# FragDRP -- experimental report

*Generated 2026-09-25 - dataset `GDSC2` - primary protocol `LDO`*

## 0. The claim, and the verdict

> **Hypothesis.** Representing a drug at the level of chemical fragments and functional groups, rather than only as an atom-level graph, improves drug response prediction, because functional groups are the units that determine chemical activity.

This is DAF-DRP's hypothesis. What is added here is an architecture that can carry it across unseen drugs, and an experimental design in which it can fail.

**Verdict.** **Not supported on this evidence.** The fragment-aware variants are not reliably better than an atom-level graph encoder under leave-drug-out evaluation on this dataset.

1. **Fragments vs atoms.** The best fragment-aware variant (`abl_a5_fragdrp_full`) scores 0.3149 on partial_correlation_drug against 0.3020 for the atom-only encoder, a change of +0.0129 (Cohen's d = 0.51, Holm-adjusted p = 1.0000). This does **not** clear the significance bar.
2. **The naive floor.** The best fragment-aware variant reaches 0.3149 on partial_correlation_drug; the naive baseline for LDO (NaiveCellLineMeanPredictor) reaches 0.3057. The model clears the floor.

The decision rules were fixed before the numbers were in, and are stated in `EXPERIMENTS.md`: the fragment model must beat the atom-only encoder on the protocol's primary metric with a Holm-corrected paired p below 0.05; BRICS must beat the fragment-count-matched random-cut control; and the model must clear the naive baseline for the protocol.

---

## 1. Setup

- **Dataset** `GDSC2`, loaded through [drevalpy](https://github.com/daisybio/drevalpy), the Python package behind the [nf-core/drugresponseeval](https://nf-co.re/drugresponseeval) pipeline.
- **Primary protocol** `LDO` -- held-out **drugs**. The model must score a molecule it has never been trained on, so everything it knows about that molecule comes from structure.
- **Primary metric** `partial_correlation_drug` -- Pearson correlation computed *within each held-out drug*, then averaged. A global correlation under LDO mostly measures whether the model guessed each drug's average potency; the within-drug correlation asks the question that matters, namely which cell lines respond to this new molecule.
- **Baselines** are the drevalpy implementations, trained on byte-identical folds.
- Cell-line and response scalers are fit on the training fold only, and predictions are inverted back to the benchmark's own scale before any metric is computed.

```
GDSC2: 77,221 responses | 241 drugs | 939 cell lines | 270 genes | response mean 0.458 sd 2.577
```

Fold composition. Leakage is asserted rather than assumed: the splitter raises if a held-out entity turns up in training.

| fold | train | val | test | test drugs | unseen drugs | test cell lines | unseen cell lines |
|---|---|---|---|---|---|---|---|
| 0 | 59216 | 5115 | 12890 | 41 | 41 | 939 | 0 |
| 1 | 58075 | 6281 | 12865 | 40 | 40 | 939 | 0 |
| 2 | 57509 | 6848 | 12864 | 40 | 40 | 938 | 0 |
| 3 | 57393 | 6958 | 12870 | 40 | 40 | 939 | 0 |
| 4 | 59287 | 5066 | 12868 | 40 | 40 | 939 | 0 |
| 5 | 57917 | 6440 | 12864 | 40 | 40 | 938 | 0 |

## 2. Baselines from nf-core/drugresponseeval

Each model below is the drevalpy implementation with its own default hyperparameters and its own feature loaders, trained on the same folds as FragDRP. Mean +/- sd across folds.

| Model | RMSE (lower better) | Pearson | Spearman | R2 | per-drug r | per-cell-line r |
|---|---|---|---|---|---|---|
| NaiveCellLineMeanPredictor | 2.5339 ± 0.3016 | 0.1807 ± 0.0439 | 0.2040 ± 0.0337 | 0.0038 ± 0.0263 | 0.3057 ± 0.0138 | n/a |
| NaiveMeanEffectsPredictor | 2.5339 ± 0.3016 | 0.1807 ± 0.0439 | 0.2040 ± 0.0337 | 0.0038 ± 0.0263 | 0.3057 ± 0.0138 | n/a |
| ElasticNet | 4.4374 ± 0.6503 | 0.1670 ± 0.1825 | 0.1625 ± 0.1970 | -2.3930 ± 1.8785 | 0.2911 ± 0.0147 | 0.1593 ± 0.1966 |
| SimpleNeuralNetwork | 2.7508 ± 0.2603 | 0.1253 ± 0.1654 | 0.1384 ± 0.1385 | -0.2030 ± 0.2734 | 0.2202 ± 0.0217 | 0.0802 ± 0.1670 |
| RandomForest | 2.7565 ± 0.4107 | 0.0845 ± 0.1419 | 0.1172 ± 0.1964 | -0.1783 ± 0.1319 | 0.1677 ± 0.0369 | 0.0773 ± 0.1788 |
| NaiveDrugMeanPredictor | 2.5728 ± 0.2963 | n/a | n/a | -0.0282 ± 0.0365 | n/a | n/a |
| NaivePredictor | 2.5728 ± 0.2963 | n/a | n/a | -0.0282 ± 0.0365 | n/a | n/a |

**Read the naive predictors first.** Under `LDO` the reference baseline is `NaiveCellLineMeanPredictor`. `NaiveDrugMeanPredictor` and `NaivePredictor` report `n/a` for the correlation metrics because under LDO they emit a constant for a held-out drug, which has no correlation with anything. That is not a bug; it is what leave-drug-out means, and it is precisely why this protocol tests a drug encoder and a random pair split does not.

## 3. The protocol decides what the number means

_Not run._

## 4. Ablation: what does the fragment view add?

Each rung adds exactly one thing to the rung above it. Data, splits, cell-line encoder, optimiser and schedule are identical throughout, so a difference between two rows is that row's factor and nothing else.

| Rung | What it is | What it adds |
|---|---|---|
| A0 | cell line only | nothing -- the floor |
| A1 | + atom-level GIN | atom-level structure (the GraphDRP family) |
| A2 | fragment view only | side branch: priors without atom message passing |
| A3 | + fragment graph composed from atoms | hierarchy |
| A4 | + descriptors and functional groups | the chemical priors |
| A5 | + cell-conditioned fragment attention | conditioning |
| A6 | DAF-DRP-style learned vocabulary | side branch: the prior method's fragment representation |

| Variant | partial_correlation_drug | RMSE | Pearson | R2 | Params |
|---|---|---|---|---|---|
| A0  cell line only | 0.2788 ± 0.0319 | 2.5404 ± 0.3139 | 0.1603 ± 0.0329 | -0.0002 ± 0.0207 | 371585 |
| A1  atom GIN | 0.3020 ± 0.0210 | 2.5787 ± 0.3314 | 0.3521 ± 0.1531 | -0.0480 ± 0.2016 | 749445 |
| A2  fragment only | 0.3087 ± 0.0149 | 2.4071 ± 0.2468 | 0.4125 ± 0.1602 | 0.0845 ± 0.1761 | 752645 |
| A3  atom+frag, no priors | 0.3164 ± 0.0203 | 2.4530 ± 0.2709 | 0.4162 ± 0.1142 | 0.0557 ± 0.1554 | 1053834 |
| A4  atom+frag + priors | 0.3073 ± 0.0176 | 2.3962 ± 0.3107 | 0.4416 ± 0.1276 | 0.1063 ± 0.1079 | 1097738 |
| A5  FragDRP (full) | 0.3149 ± 0.0262 | 2.5031 ± 0.3593 | 0.3902 ± 0.1131 | 0.0212 ± 0.1674 | 1212938 |
| A6  DAF-style vocab | 0.3030 ± 0.0262 | 2.5172 ± 0.4063 | 0.3476 ± 0.2398 | -0.0026 ± 0.2712 | 1139337 |

![Ablation](results/figures/ablation.png)

### 4.1 Why a fragment vocabulary cannot carry a held-out drug

A6 uses DAF-DRP's representation: each fragment is an index into a mined vocabulary. Before training anything, it is worth measuring what such a vocabulary can even address. Mining one from the **training drugs of each fold** and applying it to that fold's held-out drugs gives:

| fold | vocabulary | train drugs | test drugs | fragments covered | atoms covered | drugs fully covered |
|---|---|---|---|---|---|---|
| 0 | 334 | 180 | 41 | 0.6631 | 0.4505 | 0.0732 |
| 1 | 336 | 180 | 40 | 0.6878 | 0.4992 | 0.1000 |
| 2 | 343 | 180 | 40 | 0.7240 | 0.5311 | 0.1500 |
| 3 | 338 | 180 | 40 | 0.6316 | 0.4183 | 0.0000 |
| 4 | 340 | 180 | 40 | 0.6882 | 0.5235 | 0.1500 |
| 5 | 341 | 180 | 40 | 0.7602 | 0.5279 | 0.1500 |

With a vocabulary mined from the training drugs only, 69.2% of a held-out drug's fragments and 49.2% of its atoms are covered on average, and only 10.4% of held-out drugs are covered completely. The rest of each molecule has no learned representation at all under a lookup-table scheme.

Roughly half of every held-out molecule is therefore invisible to a lookup-table encoder, which is the concrete reason FragDRP computes fragment features instead of looking them up. Note that A6 as run here is given a vocabulary mined over **all** drugs, not just the training drugs -- deliberately generous, so its score is an upper bound on what that representation achieves rather than a straw man.

Paired comparison against **A1 (atom-only)** on `partial_correlation_drug`. Because every model sees identical folds, the per-fold difference is the unit of analysis. The test is a Wilcoxon signed-rank over those differences, Holm-corrected across the family. A positive difference means the variant beats A1. With 6 folds the smallest attainable two-sided p is 0.0312.

| Variant | variant | A1 | difference | Cohen's d | p raw | p Holm |  |
|---|---|---|---|---|---|---|---|
| A3  atom+frag, no priors | 0.3164 | 0.3020 | 0.0144 | 0.6758 | 0.0938 | 0.5625 | n.s. |
| A5  FragDRP (full) | 0.3149 | 0.3020 | 0.0129 | 0.5063 | 0.3125 | 1.0000 | n.s. |
| A2  fragment only | 0.3087 | 0.3020 | 0.0067 | 0.2133 | 0.8438 | 1.0000 | n.s. |
| A4  atom+frag + priors | 0.3073 | 0.3020 | 0.0053 | 0.3067 | 0.6875 | 1.0000 | n.s. |
| A6  DAF-style vocab | 0.3030 | 0.3020 | 0.0010 | 0.0599 | 1.0000 | 1.0000 | n.s. |
| A0  cell line only | 0.2788 | 0.3020 | -0.0233 | -0.7184 | 0.2188 | 1.0000 | n.s. |

## 5. Does the chemistry of the cut matter?

_Not run._

## 6. Does the prior help most when drugs are scarce?

_Not run._

## 7. What does the model attend to?

_Not run._

## 8. Relation to DAF-DRP

This work keeps DAF-DRP's central idea, that a drug encoder should see chemical fragments and not only atoms, and changes what is needed to test it.

| | DAF-DRP | FragDRP (this work) |
|---|---|---|
| Evaluation | 10-fold CV over drug-cell-line **pairs** | LPO, LCO and **LDO**, LDO primary |
| Can the drug claim be tested? | No, test drugs appear in training | Yes, test drugs are unseen |
| Naive baselines | none reported | four, and they are the bar to clear |
| Atom/fragment ablation | listed as future work | the core experiment (section 4) |
| Fragment representation | learnable lookup over 798 mined substructures | pooled from atom states + physicochemical descriptors + functional-group SMARTS |
| An unseen fragment | falls back to singleton atoms | is represented normally |
| Fragment-atom coupling | none, two independent GINs | hierarchical, fragment states are pooled from atom states |
| Fusion attention | over two pooled vectors | over the set of fragment tokens |
| Controls for the mechanism | none | matched random cuts, whole-molecule, per-atom (section 5) |
| Statistics | mean +/- sd, winner by ordering | paired Wilcoxon, Holm correction, effect sizes |

### A note on the cross-attention in DAF-DRP Eq. 6

As written, that equation applies `softmax(a Wq (b Wk)^T / sqrt(d))` where `a` and `b` are single pooled vectors. With one key, the softmax is identically 1, so the operator reduces to `b Wv` -- a linear projection of `b` that discards the query entirely. Whatever the reported gain from the attention-fusion configurations is, it is not attention doing the work. FragDRP attends over the set of fragment tokens instead, where the softmax has something to choose between, which is also what makes the per-fragment attributions in section 7 possible at all.

## 9. Limitations

- **One dataset as primary.** The headline numbers come from `GDSC2`. Cross-dataset generalisation, training on one panel and testing on another, is not covered here and is the natural next experiment.
- **Gene expression only.** Mutation, copy number, methylation and proteomics views ship with the benchmark and are unused. The cell-line side is deliberately simple so that it stays constant while the drug side varies.
- **Shared hyperparameters.** Every variant uses the same settings rather than per-variant tuning. That keeps the ablation fair, but it means no single number here is the best achievable for that architecture.
- **Response values are what the benchmark ships.** No re-curation of dose-response curves was attempted.
- **Folds are the statistical unit.** Paired tests over a handful of folds detect only fairly large effects, and treat within-fold variation as noise.

---

## 10. Reproducing this report

```bash
conda activate drp
python scripts/01_download_data.py        --datasets GDSC2
python scripts/02_run_baselines.py        --dataset GDSC2 --test-mode LDO
python scripts/03_ablation.py             --dataset GDSC2 --test-mode LDO
python scripts/04_fragmenter_comparison.py --dataset GDSC2 --test-mode LDO
python scripts/05_protocol_comparison.py  --dataset GDSC2 --with-baselines
python scripts/06_drug_budget.py          --dataset GDSC2
python scripts/07_attribution.py          --dataset GDSC2
python scripts/08_make_report.py          --dataset GDSC2 --test-mode LDO
```

`MODEL.md` describes the architecture; `EXPERIMENTS.md` gives the reasoning behind each experiment and the decision rules above.