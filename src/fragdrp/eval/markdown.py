"""Render the Markdown report from loaded results.

Separated from :mod:`fragdrp.eval.report`, which does the loading and the
verdict logic, so that "what do the numbers say" and "how is it worded" stay
apart.  Every section degrades to a short "not run" note when its experiment is
missing, so a partial run still produces a readable report.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..train.metrics import primary_metric
from .report import (
    ABLATION_LABELS,
    FRAGMENTER_LABELS,
    ReportInputs,
    Verdict,
    _fmt,
    _mean_sd,
    _sig,
    hypothesis_verdict,
    markdown_table,
    naive_baseline_name,
)

#: Path prefix the report uses for figure links, relative to REPORT.md itself.
#: :func:`build_report` overrides it so the report can live anywhere.
FIGURE_PREFIX = "results/figures"


def _section_setup(inputs: ReportInputs) -> str:
    ldo = inputs.test_mode == "LDO"
    lines = [
        "## 1. Setup",
        "",
        f"- **Dataset** `{inputs.dataset}`, loaded through "
        "[drevalpy](https://github.com/daisybio/drevalpy), the Python package behind the "
        "[nf-core/drugresponseeval](https://nf-co.re/drugresponseeval) pipeline.",
        f"- **Primary protocol** `{inputs.test_mode}`"
        + (" -- held-out **drugs**. The model must score a molecule it has never been "
           "trained on, so everything it knows about that molecule comes from structure."
           if ldo else ""),
        f"- **Primary metric** `{primary_metric(inputs.test_mode)}`"
        + (" -- Pearson correlation computed *within each held-out drug*, then averaged. "
           "A global correlation under LDO mostly measures whether the model guessed each "
           "drug's average potency; the within-drug correlation asks the question that "
           "matters, namely which cell lines respond to this new molecule."
           if ldo else ""),
        "- **Baselines** are the drevalpy implementations, trained on byte-identical folds.",
        "- Cell-line and response scalers are fit on the training fold only, and "
        "predictions are inverted back to the benchmark's own scale before any metric "
        "is computed.",
    ]
    if inputs.data_summary:
        lines += ["", "```", inputs.data_summary, "```"]
    if inputs.fold_summary is not None and not inputs.fold_summary.empty:
        lines += [
            "",
            "Fold composition. Leakage is asserted rather than assumed: the splitter "
            "raises if a held-out entity turns up in training.",
            "",
            markdown_table(
                inputs.fold_summary,
                ["fold", "n_train", "n_val", "n_test", "test_drugs", "unseen_drugs",
                 "test_cell_lines", "unseen_cell_lines"],
                ["fold", "train", "val", "test", "test drugs", "unseen drugs",
                 "test cell lines", "unseen cell lines"],
            ),
        ]
    return "\n".join(lines)


def _section_baselines(inputs: ReportInputs) -> str:
    if inputs.baseline_folds is None or inputs.baseline_folds.empty:
        return "## 2. Baselines from nf-core/drugresponseeval\n\n_Not run._"

    metric = primary_metric(inputs.test_mode)
    columns = [c for c in ["rmse", "pearson", "spearman", "r2",
                           "partial_correlation_drug", "partial_correlation_cell_line"]
               if c in inputs.baseline_folds.columns]
    grouped = inputs.baseline_folds.groupby("model")[columns].agg(["mean", "std"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    grouped = grouped.reset_index()
    if f"{metric}_mean" in grouped:
        grouped = grouped.sort_values(f"{metric}_mean", ascending=False)

    table = pd.DataFrame({"Model": grouped["model"].to_numpy()})
    for column in columns:
        table[column] = [_mean_sd(row, column) for _i, row in grouped.iterrows()]

    headers = {
        "Model": "Model", "rmse": "RMSE (lower better)", "pearson": "Pearson",
        "spearman": "Spearman", "r2": "R2", "partial_correlation_drug": "per-drug r",
        "partial_correlation_cell_line": "per-cell-line r",
    }
    naive = naive_baseline_name(inputs.test_mode)
    note = (
        "`NaiveDrugMeanPredictor` and `NaivePredictor` report `n/a` for the correlation "
        "metrics because under LDO they emit a constant for a held-out drug, which has no "
        "correlation with anything. That is not a bug; it is what leave-drug-out means, "
        "and it is precisely why this protocol tests a drug encoder and a random pair "
        "split does not."
        if inputs.test_mode == "LDO" else
        "Compare every learned model against the naive predictors before reading anything "
        "into the differences between learned models."
    )
    return "\n".join([
        "## 2. Baselines from nf-core/drugresponseeval",
        "",
        "Each model below is the drevalpy implementation with its own default "
        "hyperparameters and its own feature loaders, trained on the same folds as "
        "FragDRP. Mean +/- sd across folds.",
        "",
        markdown_table(table, list(table.columns), [headers[c] for c in table.columns]),
        "",
        f"**Read the naive predictors first.** Under `{inputs.test_mode}` the reference "
        f"baseline is `{naive}`. " + note,
    ])


def _section_protocols(inputs: ReportInputs) -> str:
    if inputs.protocol_summary is None or inputs.protocol_summary.empty:
        return "## 3. The protocol decides what the number means\n\n_Not run._"

    frame = inputs.protocol_summary.copy()
    base = frame["run"].str.replace(r"_(LPO|LCO|LDO|LTO)$", "", regex=True)
    frame["Model"] = base.map(ABLATION_LABELS).fillna(base)
    rows = [
        {
            "Protocol": row.get("test_mode", row.get("protocol", "")),
            "Model": row["Model"],
            "Pearson": _mean_sd(row, "pearson"),
            "RMSE": _mean_sd(row, "rmse"),
            "R2": _mean_sd(row, "r2"),
            "per-drug r": _mean_sd(row, "partial_correlation_drug"),
            "per-cell-line r": _mean_sd(row, "partial_correlation_cell_line"),
        }
        for _i, row in frame.iterrows()
    ]
    table = pd.DataFrame(rows)

    lines = [
        "## 3. The protocol decides what the number means",
        "",
        "The same two models, the same data, three split protocols:",
        "",
        markdown_table(table, list(table.columns), list(table.columns)),
        "",
        f"![Protocols]({FIGURE_PREFIX}/protocols.png)",
        "",
        "**LPO** holds out random drug-cell-line pairs, which is the protocol DAF-DRP "
        "reports. Both the drug and the cell line have been seen during training, so a "
        "model can score well by learning a per-drug mean and a per-cell-line mean and "
        "never consulting molecular structure at all. **LCO** holds out cell lines and "
        "tests the omics encoder. **LDO** holds out drugs, and is the only one of the "
        "three in which the drug representation has to do the work.",
    ]
    if inputs.protocol_baselines is not None and not inputs.protocol_baselines.empty:
        naive = (
            inputs.protocol_baselines.groupby(["protocol", "model"])["pearson"]
            .mean().reset_index()
            .pivot(index="model", columns="protocol", values="pearson")
            .reset_index().round(4)
        )
        lines += [
            "",
            "The naive predictors across the same protocols (global Pearson) make the "
            "point without any model at all:",
            "",
            markdown_table(naive, list(naive.columns), list(naive.columns)),
        ]
    return "\n".join(lines)


def _section_ablation(inputs: ReportInputs) -> str:
    if inputs.ablation_summary is None or inputs.ablation_summary.empty:
        return "## 4. Ablation: what does the fragment view add?\n\n_Not run._"

    metric = primary_metric(inputs.test_mode)
    frame = inputs.ablation_summary.copy()
    frame["order"] = frame["run"].map({k: i for i, k in enumerate(ABLATION_LABELS)})
    frame = frame.sort_values("order", na_position="last")

    rows = [
        {
            "Variant": ABLATION_LABELS.get(row["run"], row["run"]),
            metric: _mean_sd(row, metric),
            "RMSE": _mean_sd(row, "rmse"),
            "Pearson": _mean_sd(row, "pearson"),
            "R2": _mean_sd(row, "r2"),
            "Params": _fmt(row.get("n_parameters_mean", np.nan), 0),
        }
        for _i, row in frame.iterrows()
    ]
    table = pd.DataFrame(rows)

    lines = [
        "## 4. Ablation: what does the fragment view add?",
        "",
        "Each rung adds exactly one thing to the rung above it. Data, splits, cell-line "
        "encoder, optimiser and schedule are identical throughout, so a difference "
        "between two rows is that row's factor and nothing else.",
        "",
        "| Rung | What it is | What it adds |",
        "|---|---|---|",
        "| A0 | cell line only | nothing -- the floor |",
        "| A1 | + atom-level GIN | atom-level structure (the GraphDRP family) |",
        "| A2 | fragment view only | side branch: priors without atom message passing |",
        "| A3 | + fragment graph composed from atoms | hierarchy |",
        "| A4 | + descriptors and functional groups | the chemical priors |",
        "| A5 | + cell-conditioned fragment attention | conditioning |",
        "| A6 | DAF-DRP-style learned vocabulary | side branch: the prior method's "
        "fragment representation |",
        "",
        markdown_table(table, list(table.columns), list(table.columns)),
        "",
        f"![Ablation]({FIGURE_PREFIX}/ablation.png)",
    ]
    lines += _vocabulary_note(inputs)
    if inputs.ablation_significance is not None and not inputs.ablation_significance.empty:
        sig = inputs.ablation_significance.copy()
        sig["Variant"] = sig["model"].map(ABLATION_LABELS).fillna(sig["model"])
        sig["sig"] = sig.apply(_sig, axis=1)
        n_folds = int(sig["n_folds"].max())
        lines += [
            "",
            f"Paired comparison against **A1 (atom-only)** on `{metric}`. Because every "
            "model sees identical folds, the per-fold difference is the unit of analysis. "
            "The test is a Wilcoxon signed-rank over those differences, Holm-corrected "
            f"across the family. A positive difference means the variant beats A1. "
            f"With {n_folds} folds the smallest attainable two-sided p is "
            f"{2 / (2 ** n_folds):.4f}.",
            "",
            markdown_table(
                sig,
                ["Variant", "model_mean", "reference_mean", "improvement", "cohens_d",
                 "p_wilcoxon", "p_adjusted", "sig"],
                ["Variant", "variant", "A1", "difference", "Cohen's d",
                 "p raw", "p Holm", ""],
            ),
        ]
    return "\n".join(lines)


def _vocabulary_note(inputs: ReportInputs) -> list:
    """Why A6 is expected to struggle, measured rather than asserted."""
    frame = inputs.vocabulary_coverage
    if frame is None or frame.empty:
        return []
    from .vocabulary import summarise_coverage

    return [
        "",
        "### 4.1 Why a fragment vocabulary cannot carry a held-out drug",
        "",
        "A6 uses DAF-DRP's representation: each fragment is an index into a mined "
        "vocabulary. Before training anything, it is worth measuring what such a "
        "vocabulary can even address. Mining one from the **training drugs of each "
        "fold** and applying it to that fold's held-out drugs gives:",
        "",
        markdown_table(
            frame.round(4),
            ["fold", "vocab_size", "train_drugs", "test_drugs", "fragment_coverage",
             "atom_coverage", "fully_covered_drugs"],
            ["fold", "vocabulary", "train drugs", "test drugs", "fragments covered",
             "atoms covered", "drugs fully covered"],
        ),
        "",
        summarise_coverage(frame),
        "",
        "Roughly half of every held-out molecule is therefore invisible to a "
        "lookup-table encoder, which is the concrete reason FragDRP computes fragment "
        "features instead of looking them up. Note that A6 as run here is given a "
        "vocabulary mined over **all** drugs, not just the training drugs -- "
        "deliberately generous, so its score is an upper bound on what that "
        "representation achieves rather than a straw man.",
    ]


def _section_fragmenters(inputs: ReportInputs) -> str:
    if inputs.fragmenter_summary is None or inputs.fragmenter_summary.empty:
        return "## 5. Does the chemistry of the cut matter?\n\n_Not run._"

    metric = primary_metric(inputs.test_mode)
    frame = inputs.fragmenter_summary.copy()
    rows = [
        {
            "Fragmentation": FRAGMENTER_LABELS.get(row["run"], row["run"]),
            metric: _mean_sd(row, metric),
            "RMSE": _mean_sd(row, "rmse"),
            "Pearson": _mean_sd(row, "pearson"),
        }
        for _i, row in frame.iterrows()
    ]
    table = pd.DataFrame(rows)

    lines = [
        "## 5. Does the chemistry of the cut matter?",
        "",
        "BRICS cuts where a medicinal chemist would disconnect a molecule, and the "
        "hypothesis says that is *why* the fragment view helps. The competing explanation "
        "is that any coarse-graining would do and the chemistry is incidental.",
        "",
        "`random_cut` separates the two. It cuts acyclic single bonds uniformly at random, "
        "with the number of cuts tuned per molecule so that the **final fragment count "
        "matches what BRICS produces on that same molecule**. Fragment count and size "
        "distribution are therefore matched, and only the chemical meaning of the cut is "
        "removed. If this control matches BRICS, the functional-group account is wrong.",
        "",
        markdown_table(table, list(table.columns), list(table.columns)),
        "",
        f"![Fragmenters]({FIGURE_PREFIX}/fragmenters.png)",
    ]
    if inputs.fragment_stats is not None and not inputs.fragment_stats.empty:
        lines += [
            "",
            "The confound check. The comparison above is only meaningful if the control "
            "really is matched on the things that are not the manipulation:",
            "",
            markdown_table(
                inputs.fragment_stats,
                ["method", "fragments_per_molecule", "fragments_sd", "atoms_per_fragment",
                 "max_fragments", "singleton_fragments_pct"],
                ["method", "fragments/molecule", "sd", "atoms/fragment",
                 "max fragments", "% single-atom"],
            ),
            "",
            f"![Fragment statistics]({FIGURE_PREFIX}/fragment_stats.png)",
        ]
    if inputs.fragmenter_significance is not None and not inputs.fragmenter_significance.empty:
        sig = inputs.fragmenter_significance.copy()
        sig["Fragmentation"] = sig["model"].map(FRAGMENTER_LABELS).fillna(sig["model"])
        sig["sig"] = sig.apply(_sig, axis=1)
        lines += [
            "",
            f"Paired comparison against BRICS on `{metric}`. A negative difference means "
            "the method is worse than BRICS.",
            "",
            markdown_table(
                sig,
                ["Fragmentation", "model_mean", "reference_mean", "improvement",
                 "cohens_d", "p_adjusted", "sig"],
                ["Fragmentation", "method", "BRICS", "difference", "Cohen's d",
                 "p Holm", ""],
            ),
        ]
    return "\n".join(lines)


def _section_budget(inputs: ReportInputs) -> str:
    if inputs.budget_summary is None or inputs.budget_summary.empty:
        return "## 6. Does the prior help most when drugs are scarce?\n\n_Not run._"

    metric = primary_metric(inputs.test_mode)
    frame = inputs.budget_summary.copy()
    frame["budget %"] = (frame["budget"] * 100).round(0).astype(int)
    frame[metric] = [_mean_sd(r, metric) for _i, r in frame.iterrows()]

    lines = [
        "## 6. Does the prior help most when drugs are scarce?",
        "",
        "If chemical priors help because they let the model reuse structure it has seen "
        "in *other* molecules, then the advantage over an atom-level encoder should be "
        "largest when few training drugs are available and should shrink as drug coverage "
        "grows. That is a prediction the hypothesis makes in advance, and it can fail "
        "even if the fragment model wins overall.",
        "",
        "Only the number of training **drugs** is varied. The held-out test drugs are "
        "fixed, so every budget is scored on the same molecules.",
        "",
        markdown_table(
            frame,
            ["model", "budget %", "n_train_drugs", metric, "rmse_mean"],
            ["Model", "budget %", "train drugs", metric, "RMSE"],
        ),
        "",
        f"![Drug budget]({FIGURE_PREFIX}/budget.png)",
    ]
    pivot = frame.pivot_table(index="budget", columns="model", values=f"{metric}_mean")
    if pivot.shape[1] == 2:
        left, right = pivot.columns
        gap = (pivot[right] - pivot[left]).dropna()
        if len(gap) >= 2:
            slope = float(np.polyfit(gap.index.to_numpy(dtype=float), gap.to_numpy(), 1)[0])
            reading = (
                "A negative slope is what the mechanism predicts: the prior substitutes "
                "for training drugs, so its advantage shrinks as drugs are added."
                if slope < 0 else
                "A non-negative slope is **not** what the mechanism predicts. The "
                "advantage does not shrink as drug coverage grows, so 'the prior "
                "substitutes for data' is not the right explanation here."
            )
            lines += [
                "",
                f"The gap (`{right}` minus `{left}`) changes by **{slope:+.4f} per unit "
                f"of drug budget**. " + reading,
            ]
    return "\n".join(lines)


def _section_attribution(inputs: ReportInputs) -> str:
    if inputs.enrichment is None or inputs.enrichment.empty:
        return "## 7. What does the model attend to?\n\n_Not run._"

    lines = [
        "## 7. What does the model attend to?",
        "",
        "The cell-line-conditioned attention puts a weight on every fragment of every "
        "molecule, for every cell line. Attention weights are suggestive, not evidence, "
        "so this section moves from description to intervention.",
        "",
        "### 7.1 Which functional groups draw attention",
        "",
        "Attention relative to uniform, where 1.0 means a fragment receives exactly its "
        "share:",
        "",
        markdown_table(
            inputs.enrichment.head(15),
            ["functional_group", "n_drugs", "n_fragments", "enrichment", "enrichment_sd"],
            ["Functional group", "drugs", "fragments", "enrichment", "sd"],
        ),
        "",
        f"![Functional groups]({FIGURE_PREFIX}/functional_groups.png)",
    ]
    if inputs.top_fragments is not None and not inputs.top_fragments.empty:
        lines += [
            "",
            "### 7.2 The most-attended fragment of each held-out drug",
            "",
            "These are drugs the model never saw in training, so the fragment shown is "
            "one it had to represent from chemistry alone:",
            "",
            markdown_table(
                inputs.top_fragments.head(20),
                ["drug_name", "fragment_smiles", "functional_groups", "mean_attention"],
                ["Drug", "Top fragment (SMILES)", "Functional groups", "attention"],
            ),
        ]
    if inputs.specificity is not None and not inputs.specificity.empty:
        ratio = float(inputs.specificity["conditioning_ratio"].median())
        reading = (
            "The pattern shifts meaningfully from cell line to cell line, so this is a "
            "drug-by-cell-line interaction rather than a fixed saliency map."
            if ratio > 0.25 else
            "The pattern barely moves between cell lines, so in practice this is a "
            "drug-level saliency map. That is a weaker claim than 'the model learns which "
            "groups matter for which tumour', and this report does not make the stronger one."
        )
        lines += [
            "",
            "### 7.3 Is the attention really conditioned on the cell line?",
            "",
            f"Median ratio of between-cell-line spread to between-fragment spread: "
            f"**{ratio:.3f}**. " + reading,
        ]
    if inputs.masking:
        m = inputs.masking
        worked = m.get("mean_difference", 0) > 0 and m.get("p_wilcoxon", 1.0) < 0.05
        stats_line = (
            f", Wilcoxon p = {_fmt(m.get('p_wilcoxon'))}, d = {_fmt(m.get('cohens_d'), 2)}"
            if "p_wilcoxon" in m else ""
        )
        lines += [
            "",
            "### 7.4 The intervention test",
            "",
            "Mask the most-attended fragment and measure how far the prediction moves, "
            "against masking a random fragment of the same molecule. This test can fail, "
            "and failing would mean the attention is decorative.",
            "",
            f"- masking the top-attended fragment moves predictions by "
            f"**{_fmt(m.get('mean_delta_top'))}** on average",
            f"- masking a random fragment moves them by **{_fmt(m.get('mean_delta_random'))}**",
            f"- ratio **{_fmt(m.get('ratio'), 2)}x**, paired difference "
            f"{_fmt(m.get('mean_difference'))}{stats_line}, over n = {_fmt(m.get('n'), 0)} "
            "test pairs",
            "",
            ("**The attention identifies fragments the prediction depends on.**"
             if worked else
             "**No reliable difference, so the attention weights do not identify the "
             "fragments the prediction depends on.** The enrichment table above should be "
             "read as a description of the weights, not as a claim about the model's "
             "reasoning."),
        ]
    return "\n".join(lines)


def _section_relation_to_daf() -> str:
    return "\n".join([
        "## 8. Relation to DAF-DRP",
        "",
        "This work keeps DAF-DRP's central idea, that a drug encoder should see chemical "
        "fragments and not only atoms, and changes what is needed to test it.",
        "",
        "| | DAF-DRP | FragDRP (this work) |",
        "|---|---|---|",
        "| Evaluation | 10-fold CV over drug-cell-line **pairs** | LPO, LCO and **LDO**, "
        "LDO primary |",
        "| Can the drug claim be tested? | No, test drugs appear in training | Yes, test "
        "drugs are unseen |",
        "| Naive baselines | none reported | four, and they are the bar to clear |",
        "| Atom/fragment ablation | listed as future work | the core experiment (section 4) |",
        "| Fragment representation | learnable lookup over 798 mined substructures | "
        "pooled from atom states + physicochemical descriptors + functional-group SMARTS |",
        "| An unseen fragment | falls back to singleton atoms | is represented normally |",
        "| Fragment-atom coupling | none, two independent GINs | hierarchical, fragment "
        "states are pooled from atom states |",
        "| Fusion attention | over two pooled vectors | over the set of fragment tokens |",
        "| Controls for the mechanism | none | matched random cuts, whole-molecule, "
        "per-atom (section 5) |",
        "| Statistics | mean +/- sd, winner by ordering | paired Wilcoxon, Holm "
        "correction, effect sizes |",
        "",
        "### A note on the cross-attention in DAF-DRP Eq. 6",
        "",
        "As written, that equation applies `softmax(a Wq (b Wk)^T / sqrt(d))` where `a` "
        "and `b` are single pooled vectors. With one key, the softmax is identically 1, so "
        "the operator reduces to `b Wv` -- a linear projection of `b` that discards the "
        "query entirely. Whatever the reported gain from the attention-fusion "
        "configurations is, it is not attention doing the work. FragDRP attends over the "
        "set of fragment tokens instead, where the softmax has something to choose "
        "between, which is also what makes the per-fragment attributions in section 7 "
        "possible at all.",
    ])


def _section_limitations(inputs: ReportInputs, verdict: Verdict) -> str:
    lines = [
        "## 9. Limitations",
        "",
        f"- **One dataset as primary.** The headline numbers come from `{inputs.dataset}`. "
        "Cross-dataset generalisation, training on one panel and testing on another, is "
        "not covered here and is the natural next experiment.",
        "- **Gene expression only.** Mutation, copy number, methylation and proteomics "
        "views ship with the benchmark and are unused. The cell-line side is deliberately "
        "simple so that it stays constant while the drug side varies.",
        "- **Shared hyperparameters.** Every variant uses the same settings rather than "
        "per-variant tuning. That keeps the ablation fair, but it means no single number "
        "here is the best achievable for that architecture.",
        "- **Response values are what the benchmark ships.** No re-curation of "
        "dose-response curves was attempted.",
        "- **Folds are the statistical unit.** Paired tests over a handful of folds detect "
        "only fairly large effects, and treat within-fold variation as noise.",
    ]
    lines += [f"- {c}" for c in verdict.caveats]
    return "\n".join(lines)


def build_report(
    inputs: ReportInputs,
    figures: Optional[Dict[str, object]] = None,
    figure_prefix: str = FIGURE_PREFIX,
) -> str:
    """Assemble the full Markdown report.

    ``figure_prefix`` is the path figures are linked with, relative to wherever
    REPORT.md is written.
    """
    global FIGURE_PREFIX
    FIGURE_PREFIX = figure_prefix
    verdict = hypothesis_verdict(inputs)
    parts = [
        "# FragDRP -- experimental report",
        "",
        f"*Generated {date.today().isoformat()} - dataset `{inputs.dataset}` - "
        f"primary protocol `{inputs.test_mode}`*",
        "",
        "## 0. The claim, and the verdict",
        "",
        "> **Hypothesis.** Representing a drug at the level of chemical fragments and "
        "functional groups, rather than only as an atom-level graph, improves drug "
        "response prediction, because functional groups are the units that determine "
        "chemical activity.",
        "",
        "This is DAF-DRP's hypothesis. What is added here is an architecture that can "
        "carry it across unseen drugs, and an experimental design in which it can fail.",
        "",
        # The headline already carries its own emphasis, so it is not re-wrapped.
        f"**Verdict.** {verdict.headline}",
        "",
    ]
    parts += [f"{i}. {finding}" for i, finding in enumerate(verdict.findings, start=1)]
    parts += [
        "",
        "The decision rules were fixed before the numbers were in, and are stated in "
        "`EXPERIMENTS.md`: the fragment model must beat the atom-only encoder on the "
        "protocol's primary metric with a Holm-corrected paired p below 0.05; BRICS must "
        "beat the fragment-count-matched random-cut control; and the model must clear the "
        "naive baseline for the protocol.",
        "",
        "---",
        "",
        _section_setup(inputs),
        "",
        _section_baselines(inputs),
        "",
        _section_protocols(inputs),
        "",
        _section_ablation(inputs),
        "",
        _section_fragmenters(inputs),
        "",
        _section_budget(inputs),
        "",
        _section_attribution(inputs),
        "",
        _section_relation_to_daf(),
        "",
        _section_limitations(inputs, verdict),
        "",
        "---",
        "",
        "## 10. Reproducing this report",
        "",
        "```bash",
        "conda activate drp",
        f"python scripts/01_download_data.py        --datasets {inputs.dataset}",
        f"python scripts/02_run_baselines.py        --dataset {inputs.dataset} "
        f"--test-mode {inputs.test_mode}",
        f"python scripts/03_ablation.py             --dataset {inputs.dataset} "
        f"--test-mode {inputs.test_mode}",
        f"python scripts/04_fragmenter_comparison.py --dataset {inputs.dataset} "
        f"--test-mode {inputs.test_mode}",
        f"python scripts/05_protocol_comparison.py  --dataset {inputs.dataset} "
        "--with-baselines",
        f"python scripts/06_drug_budget.py          --dataset {inputs.dataset}",
        f"python scripts/07_attribution.py          --dataset {inputs.dataset}",
        f"python scripts/08_make_report.py          --dataset {inputs.dataset} "
        f"--test-mode {inputs.test_mode}",
        "```",
        "",
        "`MODEL.md` describes the architecture; `EXPERIMENTS.md` gives the reasoning "
        "behind each experiment and the decision rules above.",
    ]
    return "\n".join(parts)
