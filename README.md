# FragDRP

Hierarchical fragment-aware drug response prediction, benchmarked against
[nf-core/drugresponseeval](https://nf-co.re/drugresponseeval).

A drug encoder that represents a molecule as atoms → chemical fragments →
functional groups, evaluated in the protocol where that representation actually
has to do something: **leave-drug-out**, scoring molecules the model has never
seen.

---

## Why this exists

The starting point is DAF-DRP's hypothesis — *functional groups, not individual
atoms, determine chemical activity, so a drug encoder should see them.* That is
a good idea. The problem is that the usual way of testing it cannot work:

- Results are reported on random drug–cell-line **pair** splits, where every
  test drug was seen in training. A model reaches R² ≈ 0.8 there by learning a
  per-drug mean and a per-cell-line mean, with no molecular structure involved.
  A claim about drug representation is not measurable in that protocol.
- No naive baselines are reported, so the headline numbers have no floor.
- The atom-vs-fragment ablation — the experiment the claim rests on — is listed
  as future work.
- Fragments are indices into a mined 798-entry vocabulary, so an unseen
  fragment has no representation at all. That is exactly the case a held-out
  drug produces.
- The "cross-attention" fusion softmaxes over a single key, which is
  identically 1, so the operator reduces to a linear projection that discards
  its query.

This repository keeps the hypothesis and rebuilds everything needed to test it:
a vocabulary-free fragment representation, leave-drug-out evaluation against
drevalpy's baselines, an ablation ladder, a random-cut control that can falsify
the mechanism, and paired statistics.

See [`MODEL.md`](MODEL.md) for the architecture and
[`EXPERIMENTS.md`](EXPERIMENTS.md) for the design and the pre-registered
decision rules. [`REPORT.md`](REPORT.md) is generated from results and is never
hand-edited.

---

## Install

```powershell
powershell -ExecutionPolicy Bypass -File scripts\00_setup_env.ps1
conda activate drp
```

Or by hand:

```bash
conda create -y -n drp python=3.11
conda activate drp
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "numpy<2.3" pandas scipy scikit-learn rdkit pyyaml tqdm matplotlib seaborn drevalpy
pip install --no-deps -e .
```

No PyTorch Geometric and no compiler toolchain: the message passing is built
from `torch.index_add_` / `scatter_reduce_` in
[`layers.py`](src/fragdrp/models/layers.py).

---

## Run

```bash
# 2-minute end-to-end check that nothing is broken
python scripts/03_ablation.py --dataset TOYv1 --quick

# the real thing
python scripts/01_download_data.py         --datasets GDSC2 CCLE CTRPv2
python scripts/02_run_baselines.py         --dataset GDSC2 --test-mode LDO
python scripts/03_ablation.py              --dataset GDSC2 --test-mode LDO
python scripts/04_fragmenter_comparison.py --dataset GDSC2 --test-mode LDO
python scripts/05_protocol_comparison.py   --dataset GDSC2 --with-baselines
python scripts/06_drug_budget.py           --dataset GDSC2
python scripts/07_attribution.py           --dataset GDSC2 --test-mode LDO
python scripts/08_make_report.py           --dataset GDSC2 --test-mode LDO
```

`scripts/run_all.py` chains these. Every script accepts
`--set key.path=value` for one-off overrides, e.g.
`--set train.max_epochs=40 --set model.hidden_dim=256`.

---

## Layout

```
configs/
  base.yaml                 every default, in one place
  model/a0…a6*.yaml         the ablation ladder (§4 of the report)
  fragmenter/*.yaml         BRICS, RECAP, Murcko, rotatable-bond + 3 controls
  data/*.yaml               GDSC2, CCLE, CTRPv2, TOYv1
  experiment/*.yaml         LPO / LCO / LDO protocols
src/fragdrp/
  config.py                 typed configs; YAML with a `defaults:` chain
  chem/
    fragmenters.py          "which bonds do we cut?" — one interface, 7 strategies
    functional_groups.py    61 curated pharmacophore SMARTS
    featurizer.py           SMILES → two-level graph
  data/
    loader.py               the only module that knows drevalpy's layout
    splits.py               LPO/LCO/LDO folds, with leakage assertions
    dataset.py              torch Dataset, graph batching, per-fold scalers
  models/
    layers.py               GINE, attention pooling, segment ops
    fragdrp.py              the model; ablations are flags on this class
  train/
    trainer.py, metrics.py  identical training loop for every variant
  eval/
    stats.py                paired Wilcoxon, Holm, effect sizes, drug bootstrap
    attribution.py          fragment attention + the masking intervention
    plots.py, markdown.py   figures and the generated report
  baselines.py              drives drevalpy's models on identical folds
  experiment.py             one config → all folds → metrics
scripts/                    01…08, one per experiment
```

Configs compose through a `defaults:` list, so a variant file is only its own
manipulation:

```yaml
# configs/model/a1_atom_only.yaml
defaults: [../base.yaml]
name: abl_a1_atom_only
model:
  name: A1-AtomOnly
  use_atom_view: true
  use_frag_view: false
```

---

## Fragmentation strategies

| Method | What it does |
|---|---|
| `brics` | **default** — 16 retrosynthetic rules (Degen 2008) |
| `recap` | 11 more conservative cleavage rules (Lewell 1998) |
| `murcko` | ring systems vs side chains (Bemis–Murcko) |
| `rotatable_bond` | every rotatable bond |
| `random_cut` | **the control** — random cuts, fragment count matched to BRICS |
| `whole` | one fragment per molecule (lower bound) |
| `atom` | one fragment per atom (depth without abstraction) |

`random_cut` is the experiment that can falsify the hypothesis: it matches
BRICS on fragment count and size and differs only in *where* the cuts land. If
it performs as well, the benefit is coarse-graining rather than functional-group
chemistry, and the report says so.

Adding a strategy means writing one function that returns a set of bonds and
registering it in `_FINDERS` — everything downstream is shared.

---

## Notes on correctness

- Splits come from drevalpy, so baselines and FragDRP see byte-identical rows.
- The splitter **raises** if a held-out drug, cell line or tissue appears in
  training, rather than letting a good-looking number through.
- Scalers are fit on training rows only, per fold; predictions are inverted back
  to the benchmark's scale before any metric.
- Every ablation shares one training loop and one hyperparameter set, so table
  differences are architecture, not tuning.
- With *n* folds the smallest attainable two-sided Wilcoxon p is `2/2ⁿ`, so at
  n = 5 significance is unreachable by construction. Runs use ≥ 6 folds and the
  stats module warns when a comparison is under-powered.
