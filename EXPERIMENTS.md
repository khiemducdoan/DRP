# Experimental design

How the fragment hypothesis is tested here, and why each experiment is built the
way it is. The decision rules in §7 are fixed **before** the numbers, so
`REPORT.md` can only report what the data says rather than picking a framing
afterwards.

> **Hypothesis (H).** Representing a drug at the level of chemical fragments and
> functional groups, rather than only as an atom-level graph, improves drug
> response prediction — because functional groups are the units that determine
> chemical activity.

H has two parts, and they can come apart:

- **H1 (effect).** A fragment-aware encoder beats an atom-only encoder.
- **H2 (mechanism).** The improvement is due to the *chemistry* of the
  decomposition, not merely to coarse-graining the graph.

Most fragment-based papers test neither, report H1-shaped numbers in a protocol
that cannot measure them, and assert H2. The experiments below test both, and
H2 in particular is designed so it can come out negative.

---

## 1. The protocol problem, first

This is the single most consequential decision, and it is where DAF-DRP's
evidence does not support its claim.

DAF-DRP reports 10-fold cross-validation over drug–cell-line **pairs** and
obtains R² ≈ 0.80, PCC ≈ 0.90 on CCLE. In that protocol every test drug appears
in training alongside hundreds of *other* cell lines. A model can therefore
reach a high score by learning two lookup tables — a per-drug mean potency and a
per-cell-line mean sensitivity — and never consulting molecular structure at
all. The drug encoder is not what is being measured, so **a claim about drug
representation cannot be tested in that protocol**, no matter how large the
reported gap between fusion variants.

The three protocols, all from drevalpy, and what each can support:

| Protocol | Held out | Tests | Suitable for H? |
|---|---|---|---|
| **LPO** | random pairs | imputation of missing wells | No — drug seen in training |
| **LCO** | cell lines | the omics encoder | No — drug seen in training |
| **LDO** | **drugs** | **the drug encoder** | **Yes** |

Under LDO the model must score a molecule it has never been trained on, so every
bit of information it has about that molecule comes from structure. LDO is
therefore the primary protocol for every experiment here. LPO and LCO are run
too (experiment E1b), not to find a winner but to show how much of a headline
number is available without a model at all.

**The primary metric under LDO is the per-drug Pearson correlation** —
correlation computed *within* each held-out drug, then averaged. A global
correlation under LDO is dominated by whether the model guessed each drug's
average potency; the within-drug correlation asks the question that matters:
given this new molecule, which cell lines respond to it? Both are reported.

---

## 2. E1 — Benchmark against nf-core/drugresponseeval

**Question.** Where does the model sit against the community's baselines, on the
community's splits?

Baselines are the drevalpy implementations with their own defaults and feature
loaders, trained on byte-identical folds:
`NaivePredictor`, `NaiveDrugMeanPredictor`, `NaiveCellLineMeanPredictor`,
`NaiveMeanEffectsPredictor`, `ElasticNet`, `RandomForest`,
`SimpleNeuralNetwork`, `MultiViewNeuralNetwork`.

`SVR`, `GradientBoosting` and `KNNRegressor` are available through `--models`
but are not in the default set: kernel SVR scales quadratically in the number of
training pairs and gradient boosting is serial, so on a panel the size of GDSC2
both cost hours for a baseline nobody claims is competitive.

**The naive predictors are the point, not filler.** DAF-DRP reports none, and
without them an R² of 0.8 is uninterpretable. Each protocol has a reference
baseline that a model must clear before any claim is warranted:

| Protocol | Reference baseline | Why it is strong there |
|---|---|---|
| LPO | `NaiveMeanEffectsPredictor` | additive drug + cell-line effects explain most pair variance |
| LCO | `NaiveDrugMeanPredictor` | the drug is seen; only the cell line is new |
| LDO | `NaiveCellLineMeanPredictor` | some cell lines are broadly fragile, whatever the drug |

Under LDO, `NaiveDrugMeanPredictor` returns a constant for a held-out drug and
its correlation is undefined. That is not a bug — it is what leave-drug-out
means, and it is the clearest demonstration of why the protocol matters.

`python scripts/02_run_baselines.py --dataset GDSC2 --test-mode LDO`

### E1b — the same models across all three protocols

Run in advance of looking at the results, the expectation is: LPO scores high
for everything including the naive mean-effects model; LCO differences track the
omics encoder; LDO scores drop sharply and only then do drug encoders separate.
If that pattern does *not* appear, the framing in §1 is wrong and the report
says so.

`python scripts/05_protocol_comparison.py --dataset GDSC2 --with-baselines`

---

## 3. E2 — The ablation ladder (tests H1)

**Question.** What does the fragment view add, over an atom-level encoder?

Each rung adds exactly one thing. Data, splits, cell-line encoder, optimiser,
schedule and early stopping are identical throughout, so a difference between
two rungs is attributable to that rung's factor.

| Rung | Config | What it adds | What it isolates |
|---|---|---|---|
| **A0** | `a0_cell_only` | nothing — no drug information | the floor: what the cell line alone buys |
| **A1** | `a1_atom_only` | atom-level GIN | the GraphDRP/DeepCDR family — **the reference** |
| **A2** | `a2_frag_only` | fragment view, no atom message passing | what the chemical priors carry alone |
| **A3** | `a3_atom_frag_bare` | fragment graph composed from atoms, no priors | hierarchy, separated from chemistry |
| **A4** | `a4_atom_frag_priors` | descriptors + functional groups | **the chemical prior itself** |
| **A5** | `a5_fragdrp_full` | cell-conditioned fragment attention | conditioning |
| **A6** | `a6_daf_style_vocab` | DAF-DRP's learned 798-fragment lookup | what a vocabulary representation costs |

The informative contrasts:

- **A1 → A4/A5** is H1. This is the comparison the hypothesis lives or dies on.
- **A3 vs A4** separates *hierarchy* from *chemistry*. If A3 already matches A4,
  the benefit is an extra graph level and more parameters, not functional groups.
- **A6 vs A4** is the representation choice, with everything else held equal:
  a lookup table over mined fragments against computed chemical features.
- **A0** bounds everything. If A1 barely beats A0, no drug encoder on this
  dataset is doing much, and differences further up the ladder are noise.

`python scripts/03_ablation.py --dataset GDSC2 --test-mode LDO`

---

## 4. E3 — Fragmentation strategies, and the falsification test (tests H2)

**Question.** Does it matter *where* the molecule is cut, or only that it is cut?

H2 says BRICS helps because its cuts separate functional groups. The competing
explanation is that any decomposition into similarly-sized pieces would do just
as well, and the chemistry is incidental. These make different predictions, and
one control separates them.

### The control: `random_cut`

Cuts acyclic single bonds **uniformly at random**, with the number of cuts tuned
per molecule so the **final fragment count matches what BRICS produces on that
same molecule**. Fragment count and size distribution are therefore matched;
only the chemical meaning of the cut is removed.

Matching the number of *cut bonds* would not be enough — BRICS places its cuts
where they yield chemically substantial pieces, whereas random cuts often shave
off single atoms that the merging step reabsorbs. Matching *surviving fragments*
makes the control differ from BRICS in cut placement and nothing else.

`scripts/04_fragmenter_comparison.py` also writes the fragmentation statistics
per method, so the report can show the control really is matched rather than
asserting it.

**This is the experiment that can kill the hypothesis.** To the best of my
reading of the fragment-based DRP literature, including DAF-DRP, no prior paper
runs it. If `random_cut` matches BRICS, H2 is false — the architecture may still
be worth having, but not for the stated reason.

### The rest of the panel

| Method | Question it answers |
|---|---|
| `brics` | the default |
| `recap` | is it BRICS specifically, or any retrosynthetic rule set? |
| `murcko` | does coarse ring-vs-side-chain topology suffice? |
| `rotatable_bond` | conformational rather than reaction-based cuts |
| `random_cut` | **the falsification test** |
| `whole` | one fragment: the fragment branch degenerates to a second readout — lower bound |
| `atom` | one fragment per atom: depth without abstraction |

`python scripts/04_fragmenter_comparison.py --dataset GDSC2 --test-mode LDO`

---

## 5. E4 — The drug-budget curve (tests the mechanism behind H2)

**Question.** Does the prior help most when training drugs are scarce?

This turns H into a prediction rather than a post-hoc explanation. If chemical
priors work because they let a model reuse chemistry seen in *other* molecules,
then the advantage over an atom-only encoder should be **largest when few
training drugs are available and should shrink as drug coverage grows** — with
enough drugs, an atom-level encoder can learn the same regularities from data.

Only the number of training **drugs** is varied (20%, 40%, 70%, 100%), not the
number of rows: the resource under study is chemical diversity. Test drugs are
held fixed, so every budget is scored on the same molecules. Validation is
restricted to the available drugs too — early stopping must not peek at drugs
the budget says are unavailable.

A flat or inverted gap contradicts the mechanism even if the fragment model wins
overall, and the report reads the slope either way.

`python scripts/06_drug_budget.py --dataset GDSC2`

---

## 6. E5 — Attribution, from description to intervention

**Question.** What does the model attend to, and does it matter?

Attention weights are suggestive, not evidence. Three steps of increasing
strength:

1. **Description.** Which functional groups receive more attention than uniform,
   across held-out drugs?
2. **Conditioning.** Does a drug's attention pattern change across cell lines,
   or has it collapsed into a cell-line-independent saliency map? Both are
   reportable, and they support *different* claims — the report makes the weaker
   one if that is what the data shows.
3. **Intervention — the falsifiable step.** Mask the most-attended fragment and
   measure how far the prediction moves, against masking a random fragment of
   the same molecule. If top-masking does not move predictions more than
   random-masking, the attention is decorative and the enrichment table in step 1
   describes the weights, not the model's reasoning.

`python scripts/07_attribution.py --dataset GDSC2 --test-mode LDO`

---

## 7. Decision rules, fixed in advance

`REPORT.md` §0 applies these mechanically
([`hypothesis_verdict`](src/fragdrp/eval/report.py)):

1. **Effect (H1).** The best fragment-aware variant (A4 or A5) must beat A1 on
   the protocol's primary metric with a Holm-corrected paired p < 0.05.
2. **Mechanism (H2).** BRICS must beat the fragment-count-matched `random_cut`
   control, same test.
3. **Floor.** The model must beat the protocol's naive baseline. A model that
   loses to a per-cell-line mean has not earned a claim about molecules.

| Rule 1 | Rule 2 | Verdict |
|---|---|---|
| pass | pass | **Supported** |
| pass | fail | **Partially supported** — the architecture helps, but by coarse-graining, not chemistry |
| fail | pass | **Partially supported** — chemically-placed cuts beat random ones, but the overall effect is within fold noise |
| fail | — | **Not supported on this evidence** |

Rule 3 failing attaches a caveat to any verdict: all differences are then
differences between weak predictors.

### Statistics

- **Paired, always.** Every model sees identical folds, so the per-fold
  difference is the unit of analysis. Comparing unpaired means throws that away.
- **Wilcoxon signed-rank**, which assumes no normality — appropriate at this
  sample size.
- **Holm–Bonferroni** across each family of comparisons. A table of a dozen
  variants against one reference will otherwise produce a spurious p < 0.05.
- **Effect sizes reported alongside.** At these sample sizes a p-value says
  little on its own; the mean paired difference and Cohen's d say how much.
- **≥ 6 folds.** With *n* folds the smallest attainable two-sided Wilcoxon
  p-value is `2/2ⁿ`, so **at n = 5 the test can never reach p < 0.05 regardless
  of effect size**. Benchmarks reporting five folds and claiming significance
  trip on this. `compare_against` warns when a comparison is under-powered by
  construction.
- **Bootstrap over drugs, not rows**, for per-drug intervals: rows sharing a
  drug are the same molecule measured repeatedly and are not independent.

---

## 8. Datasets

| Dataset | Responses | Drugs | Cell lines | Role here |
|---|---|---|---|---|
| **GDSC2** | 90,666 | 287 | 969 | **primary** — best drug-count/density trade-off for LDO |
| **CTRPv2** | 235,974 | 545 | 886 | secondary — most drugs, strongest LDO evidence, largest |
| **CCLE** | 7,159 | 24 | 503 | consistency check only — 24 drugs cannot support an LDO claim |
| TOYv1 | 2,711 | 36 | 90 | pipeline smoke test; not evidence |

CCLE is one of the two datasets DAF-DRP reports on and is included for
comparability, but with 24 drugs an LDO fold holds out about 5 of them. Its
results are read as a consistency check, never as primary evidence.

---

## 9. Running everything

```bash
conda activate drp
python scripts/01_download_data.py         --datasets GDSC2 CCLE CTRPv2
python scripts/02_run_baselines.py         --dataset GDSC2 --test-mode LDO
python scripts/03_ablation.py              --dataset GDSC2 --test-mode LDO
python scripts/04_fragmenter_comparison.py --dataset GDSC2 --test-mode LDO
python scripts/05_protocol_comparison.py   --dataset GDSC2 --with-baselines
python scripts/06_drug_budget.py           --dataset GDSC2
python scripts/07_attribution.py           --dataset GDSC2 --test-mode LDO
python scripts/08_make_report.py           --dataset GDSC2 --test-mode LDO
```

Add `--quick` to any script for a 2-minute pipeline check (not results). Every
script takes `--set key.path=value` for one-off config overrides.

---

## 10. What would change the conclusion

Stated in advance, so the write-up is not free to move the goalposts:

- **H1 holds but H2 fails** → the fragment branch is a useful architecture, and
  the functional-group story is wrong. Report it that way; do not retreat to
  "fragments capture higher-order structure".
- **Both fail** → an atom-level GIN is sufficient on this data, and the
  fragment literature's premise is not supported at this scale.
- **Both hold on GDSC2 but not CTRPv2** → the effect is dataset-specific, which
  is a weaker and more interesting claim than either.
- **Everything sits at the naive floor** → the dataset or the protocol, not the
  model, is the binding constraint.
