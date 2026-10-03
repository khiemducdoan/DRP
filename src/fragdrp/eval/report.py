"""Assemble the Markdown report from whatever results exist on disk.

The report is generated, never hand-written, so it cannot drift from the
numbers.  Each section degrades gracefully: if an experiment has not been run,
its section says so instead of the report failing.

The wording of the conclusions is *derived from the numbers*, not fixed in
advance.  :func:`hypothesis_verdict` reads the ablation and control results and
picks between "supported", "partially supported" and "not supported" according
to stated decision rules.  That is the honest way to write up a hypothesis test:
the criteria are fixed before the data, and the text follows the data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..train.metrics import primary_metric

LOGGER = logging.getLogger(__name__)

# Human-readable names for the ablation runs, in ladder order.
ABLATION_LABELS: Dict[str, str] = {
    "abl_a0_cell_only": "A0  cell line only",
    "abl_a1_atom_only": "A1  atom GIN",
    "abl_a2_frag_only": "A2  fragment only",
    "abl_a3_atom_frag_bare": "A3  atom+frag, no priors",
    "abl_a4_atom_frag_priors": "A4  atom+frag + priors",
    "abl_a5_fragdrp_full": "A5  FragDRP (full)",
    "abl_a6_daf_style_vocab": "A6  DAF-style vocab",
    "abl_e6_ring_groups": "E6a  ring groups only",
    "abl_e6_star_groups": "E6b  star groups only",
}
ABLATION_ROLES: Dict[str, str] = {
    "abl_a0_cell_only": "control",
    "abl_a1_atom_only": "reference",
    "abl_a2_frag_only": "control",
    "abl_a3_atom_frag_bare": "proposed",
    "abl_a4_atom_frag_priors": "proposed",
    "abl_a5_fragdrp_full": "proposed",
    "abl_a6_daf_style_vocab": "control",
    "abl_e6_ring_groups": "proposed",
    "abl_e6_star_groups": "control",
}
FRAGMENTER_LABELS: Dict[str, str] = {
    "frag_brics": "BRICS",
    "frag_recap": "RECAP",
    "frag_murcko": "Murcko ring/side-chain",
    "frag_rotatable_bond": "rotatable bonds",
    "frag_random_cut": "random cuts (control)",
    "frag_atom": "one atom per fragment (control)",
    "frag_whole": "whole molecule (control)",
}
FRAGMENTER_ROLES: Dict[str, str] = {
    "frag_brics": "proposed",
    "frag_recap": "proposed",
    "frag_murcko": "proposed",
    "frag_rotatable_bond": "proposed",
    "frag_random_cut": "control",
    "frag_atom": "control",
    "frag_whole": "control",
}


@dataclass
class ReportInputs:
    """Result tables loaded from a results directory; any may be absent."""

    results_dir: Path
    dataset: str
    test_mode: str
    ablation_summary: Optional[pd.DataFrame] = None
    ablation_folds: Optional[pd.DataFrame] = None
    ablation_significance: Optional[pd.DataFrame] = None
    fragmenter_summary: Optional[pd.DataFrame] = None
    fragmenter_folds: Optional[pd.DataFrame] = None
    fragmenter_significance: Optional[pd.DataFrame] = None
    fragment_stats: Optional[pd.DataFrame] = None
    baseline_folds: Optional[pd.DataFrame] = None
    protocol_summary: Optional[pd.DataFrame] = None
    protocol_baselines: Optional[pd.DataFrame] = None
    budget_summary: Optional[pd.DataFrame] = None
    enrichment: Optional[pd.DataFrame] = None
    top_fragments: Optional[pd.DataFrame] = None
    masking: Optional[Dict[str, float]] = None
    specificity: Optional[pd.DataFrame] = None
    vocabulary_coverage: Optional[pd.DataFrame] = None
    data_summary: str = ""
    fold_summary: Optional[pd.DataFrame] = None


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("Could not read %s: %s", path, error)
        return None


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except Exception:  # noqa: BLE001
        return None


def load_inputs(results_dir: Path, dataset: str, test_mode: str) -> ReportInputs:
    """Collect every result table the scripts may have written."""
    d = Path(results_dir)
    suffix = f"{dataset}_{test_mode}"
    metric = primary_metric(test_mode)
    return ReportInputs(
        results_dir=d,
        dataset=dataset,
        test_mode=test_mode,
        ablation_summary=_read_csv(d / "ablation" / f"summary_{suffix}.csv"),
        ablation_folds=_read_csv(d / "ablation" / f"per_fold_{suffix}.csv"),
        ablation_significance=_read_csv(d / "ablation" / f"significance_{metric}_{suffix}.csv"),
        fragmenter_summary=_read_csv(d / "fragmenters" / f"summary_{suffix}.csv"),
        fragmenter_folds=_read_csv(d / "fragmenters" / f"per_fold_{suffix}.csv"),
        fragmenter_significance=_read_csv(d / "fragmenters" / f"significance_{suffix}.csv"),
        fragment_stats=_read_csv(d / "fragmenters" / f"fragment_stats_{dataset}.csv"),
        baseline_folds=_read_csv(d / "baselines" / f"per_fold_{suffix}.csv"),
        protocol_summary=_read_csv(d / "protocols" / f"summary_{dataset}.csv"),
        protocol_baselines=_read_csv(d / "protocols" / f"baselines_{dataset}.csv"),
        budget_summary=_read_csv(d / "budget" / f"summary_{suffix}.csv"),
        enrichment=_read_csv(d / "attribution" / f"functional_group_enrichment_{dataset}.csv"),
        top_fragments=_read_csv(d / "attribution" / f"top_fragment_per_drug_{dataset}.csv"),
        masking=_read_json(d / "attribution" / f"masking_summary_{dataset}.json"),
        specificity=_read_csv(d / "attribution" / f"attention_specificity_{dataset}.csv"),
        vocabulary_coverage=_read_csv(
            d / "ablation" / f"vocabulary_coverage_{suffix}.csv"
        ),
        fold_summary=_read_csv(d / "baselines" / f"folds_{suffix}.csv"),
    )


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def _mean_sd(row: pd.Series, metric: str, digits: int = 4) -> str:
    mean, sd = row.get(f"{metric}_mean"), row.get(f"{metric}_std")
    if mean is None or (isinstance(mean, float) and not np.isfinite(mean)):
        return "n/a"
    if sd is None or not np.isfinite(sd):
        return _fmt(mean, digits)
    return f"{float(mean):.{digits}f} ± {float(sd):.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], headers: Sequence[str]) -> str:
    """Render selected columns of a frame as a GitHub Markdown table."""
    present = [c for c in columns if c in frame.columns]
    labels = [h for c, h in zip(columns, headers) if c in frame.columns]
    lines = ["| " + " | ".join(labels) + " |",
             "|" + "|".join(["---"] * len(labels)) + "|"]
    for _i, row in frame.iterrows():
        cells = []
        for column in present:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append(_fmt(value))
            elif isinstance(value, (bool, np.bool_)):
                cells.append("yes" if value else "no")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _sig(row: pd.Series) -> str:
    """Compact significance marker."""
    if not np.isfinite(row.get("p_adjusted", np.nan)):
        return ""
    p = float(row["p_adjusted"])
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #


@dataclass
class Verdict:
    """The decision, plus the specific findings it rests on."""

    headline: str
    findings: List[str]
    caveats: List[str]


def hypothesis_verdict(inputs: ReportInputs) -> Verdict:
    """Decide whether the fragment hypothesis survived, by fixed rules.

    The rules, stated before looking at the numbers:

    1. **Necessary:** the fragment-aware model (A4 or A5) must beat the
       atom-only encoder (A1) on the protocol's primary metric, with a
       Holm-corrected paired p < 0.05.  Without this there is no effect.
    2. **Mechanism:** BRICS must beat the fragment-count-matched random-cut
       control.  If it does not, any effect is coarse-graining, not chemistry,
       and the hypothesis *as stated* is wrong even though the model may still
       be useful.
    3. **Floor:** the model must beat the naive baseline appropriate to the
       protocol.  A model that loses to a per-cell-line mean has not earned a
       claim about molecules.
    """
    findings: List[str] = []
    caveats: List[str] = []
    metric = primary_metric(inputs.test_mode)

    # --- rule 1: fragments beat atoms ------------------------------------- #
    beats_atoms: Optional[bool] = None
    if inputs.ablation_significance is not None and not inputs.ablation_significance.empty:
        frame = inputs.ablation_significance
        candidates = frame[frame["model"].isin(["abl_a4_atom_frag_priors", "abl_a5_fragdrp_full"])]
        if not candidates.empty:
            best = candidates.loc[candidates["improvement"].idxmax()]
            beats_atoms = bool(best["significant"] and best["improvement"] > 0)
            findings.append(
                f"**Fragments vs atoms.** The best fragment-aware variant "
                f"(`{best['model']}`) scores {_fmt(best['model_mean'])} on {metric} "
                f"against {_fmt(best['reference_mean'])} for the atom-only encoder, "
                f"a change of {best['improvement']:+.4f} "
                f"(Cohen's d = {_fmt(best['cohens_d'], 2)}, "
                f"Holm-adjusted p = {_fmt(best['p_adjusted'])}). "
                + ("This clears the significance bar."
                   if beats_atoms else
                   "This does **not** clear the significance bar.")
            )
            under_powered = candidates["n_folds"].max() < 6
            if under_powered:
                caveats.append(
                    f"Only {int(candidates['n_folds'].max())} folds: the Wilcoxon "
                    "signed-rank test cannot reach p < 0.05 at that sample size "
                    "regardless of effect size, so significance here is not "
                    "attainable by construction."
                )

    # --- rule 2: chemistry vs matched random cuts -------------------------- #
    chemistry_matters: Optional[bool] = None
    if inputs.fragmenter_significance is not None and not inputs.fragmenter_significance.empty:
        frame = inputs.fragmenter_significance
        control = frame[frame["model"] == "frag_random_cut"]
        if not control.empty:
            row = control.iloc[0]
            # 'improvement' is oriented as random_cut minus BRICS.
            gap = -float(row["improvement"])
            chemistry_matters = bool(row["significant"] and row["improvement"] < 0)
            findings.append(
                f"**Chemistry of the cut.** BRICS scores {_fmt(row['reference_mean'])} "
                f"against {_fmt(row['model_mean'])} for random cuts matched on fragment "
                f"count, a gap of {gap:+.4f} (Holm-adjusted p = {_fmt(row['p_adjusted'])}). "
                + ("The placement of the cut matters, which is what the "
                   "functional-group account predicts."
                   if chemistry_matters else
                   "**The control is not distinguishable from BRICS.** Whatever the "
                   "fragment branch contributes, it is not explained by the chemical "
                   "meaning of the cut.")
            )

    # --- rule 3: the naive floor ------------------------------------------ #
    beats_naive: Optional[bool] = None
    naive_value = naive_reference_value(inputs)
    if naive_value is not None and inputs.ablation_summary is not None:
        best_row = _best_fragment_row(inputs.ablation_summary, metric)
        if best_row is not None:
            model_value = float(best_row[f"{metric}_mean"])
            beats_naive = model_value > naive_value
            findings.append(
                f"**The naive floor.** The best fragment-aware variant reaches "
                f"{_fmt(model_value)} on {metric}; the naive baseline for "
                f"{inputs.test_mode} ({naive_baseline_name(inputs.test_mode)}) reaches "
                f"{_fmt(naive_value)}. "
                + ("The model clears the floor."
                   if beats_naive else
                   "**The model does not clear the floor**, so its score is not "
                   "evidence that it uses molecular structure.")
            )

    # --- combine ----------------------------------------------------------- #
    if beats_atoms and chemistry_matters and (beats_naive is not False):
        headline = (
            "**Supported.** Fragment-level representation improves leave-drug-out "
            "prediction over an atom-level encoder, and the improvement depends on "
            "cutting the molecule at chemically meaningful bonds rather than merely "
            "cutting it into pieces."
        )
    elif beats_atoms and chemistry_matters is False:
        headline = (
            "**Partially supported.** The fragment view helps, but a random-cut "
            "control matched on fragment count does just as well, so the gain comes "
            "from hierarchical coarse-graining rather than from functional-group "
            "semantics. The architecture is useful; the stated mechanism is not "
            "the reason it works."
        )
    elif beats_atoms is False and chemistry_matters:
        headline = (
            "**Partially supported.** Chemically-placed cuts beat random ones, but "
            "the fragment-aware model does not significantly beat the atom-only "
            "encoder overall, so the effect is real but small relative to fold "
            "variance."
        )
    elif beats_atoms is False:
        headline = (
            "**Not supported on this evidence.** The fragment-aware variants are not "
            "reliably better than an atom-level graph encoder under leave-drug-out "
            "evaluation on this dataset."
        )
    else:
        headline = (
            "**Inconclusive.** Not all of the planned experiments produced results; "
            "the sections below report what was run."
        )

    if beats_naive is False:
        caveats.append(
            "Every model in this table, including the baselines, sits near the naive "
            "floor for this protocol. Read all differences here as differences "
            "between weak predictors."
        )
    return Verdict(headline=headline, findings=findings, caveats=caveats)


def _best_fragment_row(summary: pd.DataFrame, metric: str) -> Optional[pd.Series]:
    column = f"{metric}_mean"
    if column not in summary.columns:
        return None
    fragment_runs = summary[summary["run"].isin(["abl_a4_atom_frag_priors", "abl_a5_fragdrp_full"])]
    if fragment_runs.empty or fragment_runs[column].isna().all():
        return None
    return fragment_runs.loc[fragment_runs[column].idxmax()]


def naive_baseline_name(test_mode: str) -> str:
    from ..data.splits import REFERENCE_BASELINE

    return REFERENCE_BASELINE.get(test_mode, "NaivePredictor")


def naive_reference_value(inputs: ReportInputs) -> Optional[float]:
    """The naive baseline's score on the protocol's primary metric."""
    if inputs.baseline_folds is None or inputs.baseline_folds.empty:
        return None
    metric = primary_metric(inputs.test_mode)
    if metric not in inputs.baseline_folds.columns:
        return None
    naive = inputs.baseline_folds[
        inputs.baseline_folds["model"] == naive_baseline_name(inputs.test_mode)
    ]
    if naive.empty or naive[metric].isna().all():
        return None
    return float(naive[metric].mean())
