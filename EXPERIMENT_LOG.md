# Experiment log

Everything actually run, in order, with the result and what it changed. Written
as work happened rather than reconstructed afterwards, so it includes the runs
that failed and the bugs they exposed.

Dataset unless stated: **GDSC2** via drevalpy — 77,221 responses, 241 drugs, 939
cell lines, 270 landmark genes. Protocol **LDO** (leave-drug-out), **6 folds**
(6 because the two-sided Wilcoxon floor is `2/2ⁿ`, so at n=5 no result can reach
p < 0.05). Primary metric: **per-drug Pearson** — correlation within each
held-out drug, averaged.

---

## Setup and validation

| # | What | Outcome |
|---|---|---|
| S1 | conda env `drp`, torch 2.6+cu124, RTX 3070 Ti | CUDA confirmed |
| S2 | Fragmenter partition test, 7 methods × 6 molecules | every atom in exactly one fragment |
| S3 | Functional-group SMARTS on textbook molecules | all 61 compile and fire correctly |
| S4 | 84 unit tests | pass |
| S5 | Throughput profile | collate 9–26%, model-bound; batch 512 → 2× speedup |
| S6 | Regularisation pilot, 2 folds, 4 settings | dropout 0.3 / wd 1e-4; model peaks at **epoch 3–8** (heavy overfitting under LDO) |

**Three bugs found by the checks, not by inspection:**

1. **PyYAML reads `1e-5` as a string.** YAML 1.1 needs a decimal point *and* a
   signed exponent. Fixed by coercing to each field's declared type at load.
2. **`strip_salts` returned unsanitised fragments** → BRICS raised
   `RingInfo not initialized` on the 8 salt SMILES in GDSC2. Verified the fix
   leaves all 241 existing graphs byte-identical, so runs in flight stayed valid.
3. **The landmark gene list hard-failed** on datasets with partial coverage.
   Now intersects and logs coverage.

A fourth, mine: `--fix-attention` runs **overwrote** the faithful DAF-DRP run
because the run name did not encode the variant. Fixed; the variant is now part
of the directory name.

---

## E1 — Baselines from nf-core/drugresponseeval

drevalpy implementations, own defaults, byte-identical folds.

| Model | per-drug r | RMSE | Pearson | R² |
|---|---|---|---|---|
| `NaiveCellLineMean` | **0.3057** | 2.534 | 0.181 | 0.004 |
| `NaiveMeanEffects` | 0.3057 | 2.534 | 0.181 | 0.004 |
| ElasticNet | 0.2911 | 4.437 | 0.167 | **−2.39** |
| SimpleNeuralNetwork | 0.2202 | 2.751 | 0.125 | −0.20 |
| RandomForest | 0.1677 | 2.757 | 0.084 | −0.18 |
| `NaiveDrugMean`, `NaivePredictor` | **n/a** | 2.573 | n/a | −0.03 |

**Every learned baseline loses to the naive cell-line mean, significantly**
(paired Wilcoxon, p = 0.031 for all three), and all have **R² ≤ 0**.
`NaiveDrugMean` returns `n/a` because it cannot score a drug it has never seen —
which is what LDO means, and why a random-pair split cannot test a drug encoder.

---

## E2 — Ablation ladder

| Rung | per-drug r | vs A1 | d | p (Holm) |
|---|---|---|---|---|
| A3 atom+frag, no priors | **0.3164** ± 0.0203 | +0.0144 | 0.68 | 0.563 |
| A5 FragDRP full | 0.3149 ± 0.0262 | +0.0129 | 0.51 | 1.000 |
| A2 fragment only | 0.3087 ± 0.0149 | +0.0067 | 0.21 | 1.000 |
| A4 atom+frag + priors | 0.3073 ± 0.0176 | +0.0053 | 0.31 | 1.000 |
| A6 DAF-style vocabulary | 0.3030 ± 0.0262 | **+0.0010** | **0.06** | 1.000 |
| A1 atom GIN (reference) | 0.3020 ± 0.0210 | — | — | — |
| A0 cell line only | 0.2788 ± 0.0319 | −0.0233 | −0.72 | 1.000 |

**Rule 1 fails**: nothing beats A1 significantly after Holm correction.

Two findings that do hold up:

- **A6 ≈ A1 (d = 0.06).** DAF-DRP's lookup-table fragment representation adds
  essentially nothing over a plain atom-level GIN.
- **Adding chemical priors *lowered* the primary metric** (A3 0.3164 → A4
  0.3073) while *raising* every global metric (RMSE 2.453 → 2.396, Pearson
  0.416 → 0.442, R² 0.056 → 0.106). The metrics disagree, and they disagree
  consistently.

**Power:** with d ≈ 0.68 at n = 6, power is ≈ 0.45. "Not significant" here is a
statement about the design, not about the data.

---

## E3 — Fragmentation strategy

| Method | per-drug r |
|---|---|
| **BRICS** | **0.3134** ± 0.0155 |
| Murcko ring/side-chain | 0.3100 ± 0.0188 |
| RECAP | 0.3098 ± 0.0082 |
| **random_cut (matched control)** | **0.3073** ± 0.0240 |
| rotatable bond | 0.3058 ± 0.0318 |
| atom (control) | 0.3040 ± 0.0142 |
| whole (control) | *running* |

BRICS − random_cut = **+0.0061**, the direction H2 predicts, but small against
sd ≈ 0.02. Paired test pending the final control.

The control is matched: `random_cut` cuts acyclic single bonds at random with
the number of cuts tuned per molecule so the *final fragment count* matches
BRICS on that molecule. `results/figures/molecules/Lapatinib_methods.png` shows
what that looks like — same piece count, but random cuts merge the quinazoline
core into the linker instead of isolating it.

---

## E5 — Attribution (validated on TOYv1; GDSC2 run queued)

Pipeline works end to end. Even on a throwaway 8-epoch toy model the chemistry
is recognisable: Gefitinib → quinazoline (its actual EGFR hinge binder),
Dasatinib → pyrimidine, Tretinoin → carboxylic acid.

Masking intervention: masking the top-attended fragment moves predictions
**1.33×** more than masking a random fragment of the same molecule
(p ≈ 5×10⁻³⁸, d = 0.48, n = 818). The absolute movement is small (0.0034), so
the fragment branch is not carrying much of the prediction.

---

## DAF-DRP reproduction with BRICS (`daf_drp_brics/`)

Faithful architecture — two independent GINs, lookup table, six fusion
configurations, Table 1 hyperparameters — with Principal Subgraph Mining
replaced by BRICS. 422 distinct BRICS fragments on GDSC2 (under the 798 cap, so
nothing truncated), but **302 of them occur exactly once**.

| | per-drug r | Pearson | R² | RMSE |
|---|---|---|---|---|
| WA, Eq. 6 **as printed** | 0.2998 | **0.1746** | 0.0023 | 2.536 |
| WA, Eq. 6 **repaired** | 0.3065 | **0.3807** | 0.0327 | 2.491 |
| *`NaiveCellLineMean`* | *0.3057* | *0.1807* | *0.0038* | *2.534* |

Repairing the fusion operator **more than doubles global Pearson**. The faithful
version sits exactly on the naive baseline — see E7/P1 for why.

---

## E7 — Theory checks (no training required)

### P1 — which fusion configurations can use the drug at all?

Same cell line, four different drugs, spread of predictions, averaged over 5
initialisations:

| Fusion | spread | uses the drug? |
|---|---|---|
| WC | 0.060 | yes |
| CC | 0.022 | yes |
| AC | 0.008 | yes |
| **CA** | **0.000000** | **no** |
| **WA** | **0.000000** | **no** |
| **AA** | **0.000000** | **no** |

All three blind configurations have `GlobalFusion = A`, exactly as
THEORY.md Proposition 1 predicts: with a single key the softmax is identically
1, so `O_attn(z_drug, h_c) = h_c W_V`, a function of the cell line alone.

**WA is the configuration DAF-DRP reports as its best.**

### P4 — lookup-table collisions

Vocabulary mined per fold from training drugs only, applied to held-out drugs:

| | mean over folds |
|---|---|
| fragments of a held-out drug in vocabulary | 69.2% |
| **atoms** inside a covered fragment | **49.2%** |
| held-out drugs fully covered | 10.4% |
| held-out drugs sharing a signature with another | 8.7% |
| held-out drugs with **no** in-vocabulary fragment | 10.8% |

By Proposition 4, drugs sharing a fragment signature receive *identical*
fragment representations for every parameter setting. About one drug in nine has
no in-vocabulary fragment at all, so its entire fragment representation is a
count of unknown pieces.

---

## E6 — Ring vs star functional groups *(running)*

The first experiment here that tests a *mechanism* rather than an effect.

Chen et al. (NeurIPS 2020) prove message-passing GNNs cannot induced-subgraph-count
any connected substructure of ≥ 3 nodes, but **can** subgraph-count star-shaped
ones. So explicit SMARTS features should help for groups defined by a cycle
(26 of 61: benzene, pyridine, quinazoline, morpholine, …) and add little for
tree-shaped groups around a central atom (35 of 61: hydroxyl, nitrile,
sulfonamide, trifluoromethyl, …).

Design: the indicator vector keeps its **full 61-dim width** in every arm;
entries outside the subset are masked to zero. Parameter count is therefore
identical across arms and cannot explain a difference.

- **E6a** ring groups only
- **E6b** star groups only
- compared against **A3** (no groups) and **A4** (all groups)

*Prediction:* E6a > E6b.

### Result — the prediction fails on the primary metric, holds on the others

| | per-drug r | Pearson | R² | RMSE |
|---|---|---|---|---|
| A3 no groups | **0.3164** | 0.4162 | 0.0557 | 2.453 |
| E6b star groups only | 0.3150 | 0.3484 | −0.0110 | 2.546 |
| E6a ring groups only | 0.3127 | **0.3912** | **0.0320** | **2.491** |
| A4 all groups | 0.3073 | **0.4416** | **0.1063** | **2.396** |

Ring vs star, paired over 6 folds:

| metric | ring | star | diff | d | p |
|---|---|---|---|---|---|
| **per-drug r** | 0.3127 | 0.3150 | **−0.0023** | **−0.11** | **0.84** |
| Pearson | 0.3912 | 0.3484 | +0.0428 | +0.72 | 0.16 |
| R² | 0.0320 | −0.0110 | +0.0431 | +0.49 | 0.22 |
| RMSE | 2.491 | 2.546 | +0.0550 | +0.49 | 0.22 |

**On the primary metric there is no difference whatsoever** (d = −0.11,
p = 0.84). Proposition 3's prediction is **not supported** for within-drug
ranking, and the expressivity account does not explain that part of the task.

On the three global metrics ring does beat star, with moderate effect sizes
(d ≈ 0.5–0.7) but not significantly at n = 6. Direction is as predicted;
magnitude is unresolved. The 12-fold rerun includes both arms.

### What E2 and E6 say together

A consistent pattern across two independent experiments:

- chemical priors **raise global metrics** (A4 best on Pearson 0.442 / R² 0.106;
  ring > star on all three), i.e. they help place a new molecule on the potency
  scale;
- chemical priors **do not raise — and slightly lower — per-drug ranking**
  (A3 with no groups at all is the best rung at 0.3164).

These are two different abilities, and DAF-DRP's framing conflates them.
Predicting *how potent a new drug is on average* is a molecule-level regression
that chemical descriptors are well suited to. Predicting *which cell lines
respond to it* is a drug × cell-line interaction, and nothing tested here
improves it beyond what the hierarchical structure already provides.

---

## Still to run

| | Experiment | Why |
|---|---|---|
| E1b | LPO / LCO / LDO on the same models | how much of a headline number the protocol supplies |
| E4 | Drug-budget curve | THEORY.md P2a — the advantage should shrink as drugs are added |
| E5 | Attribution on GDSC2 | the toy run only validated the pipeline |
| — | 12-fold rerun of the decisive comparisons | power: d ≈ 0.7 at n = 6 gives ≈ 0.45 |
| — | CTRPv2 (545 drugs) | the biggest lever on LDO signal; data already downloaded |

---

## Running notes

**Run-to-run noise.** `frag_brics` in E3 uses the identical config to `A5` and
scored 0.3134 vs 0.3149 — a 0.0015 gap from non-deterministic CUDA
`index_add_`. That is ~10× smaller than the A3−A1 difference of 0.0144, so the
measured effects are not GPU noise. Worth stating in limitations.

**Hyperparameters were chosen once**, on a 2-fold pilot, and are shared by every
variant. No per-variant tuning, which keeps the ablation fair at the cost of no
row being that architecture's best achievable number.
