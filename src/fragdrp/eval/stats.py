"""Statistical comparison between runs.

DAF-DRP reports ``mean +/- sd`` across folds and declares a winner from the
ordering of the means.  With five or ten folds and differences in the third
decimal place, that is not evidence: fold-to-fold variance in drug-response
benchmarks is large, and the folds are *paired* (every model sees the same
splits), which a naive comparison throws away.

This module does three things instead:

1. **Paired tests.**  Because every model is evaluated on identical folds, the
   per-fold difference is the right unit of analysis.  The Wilcoxon signed-rank
   test makes no normality assumption, which matters at n=5.
2. **Effect sizes.**  A p-value at n=5 says little; the mean paired difference
   and Cohen's d say how much.
3. **Multiple-comparison control.**  A table of a dozen ablations against one
   baseline will produce a spurious p<0.05 by chance; Holm-Bonferroni fixes the
   family-wise error rate without the conservatism of plain Bonferroni.

For per-drug analyses there is also a bootstrap over *drugs* rather than over
rows, since rows within a drug are not independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..train.metrics import HIGHER_IS_BETTER


@dataclass
class PairedComparison:
    """One model compared against a reference, over paired folds."""

    model: str
    reference: str
    metric: str
    n_folds: int
    model_mean: float
    reference_mean: float
    difference: float
    """Signed so that a positive number always means the model is better."""
    cohens_d: float
    wilcoxon_p: float
    ttest_p: float
    p_adjusted: float = float("nan")
    significant: bool = False

    def as_row(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "reference": self.reference,
            "metric": self.metric,
            "n_folds": self.n_folds,
            "model_mean": self.model_mean,
            "reference_mean": self.reference_mean,
            "improvement": self.difference,
            "cohens_d": self.cohens_d,
            "p_wilcoxon": self.wilcoxon_p,
            "p_ttest": self.ttest_p,
            "p_adjusted": self.p_adjusted,
            "significant": self.significant,
        }


def _paired_d(differences: np.ndarray) -> float:
    """Cohen's d for paired samples: mean difference over its own sd."""
    sd = float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
    return float(np.mean(differences) / sd) if sd > 1e-12 else float("nan")


def paired_compare(
    per_fold: pd.DataFrame,
    model: str,
    reference: str,
    metric: str,
    model_column: str = "model",
    fold_column: str = "fold",
) -> Optional[PairedComparison]:
    """Compare two runs on one metric, pairing by fold.

    Returns ``None`` when the two runs share fewer than two folds, which is the
    honest answer rather than a p-value computed from one pair.
    """
    subset = per_fold[per_fold[model_column].isin([model, reference])]
    wide = subset.pivot_table(index=fold_column, columns=model_column, values=metric)
    if model not in wide.columns or reference not in wide.columns:
        return None
    wide = wide[[model, reference]].dropna()
    if len(wide) < 2:
        return None

    a = wide[model].to_numpy(dtype=float)
    b = wide[reference].to_numpy(dtype=float)
    # Orient so that positive always means "model beats reference".
    sign = 1.0 if metric in HIGHER_IS_BETTER else -1.0
    differences = sign * (a - b)

    if np.allclose(differences, 0):
        wilcoxon_p = 1.0
    else:
        try:
            wilcoxon_p = float(stats.wilcoxon(differences).pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
    ttest_p = float(stats.ttest_1samp(differences, 0.0).pvalue) if len(differences) > 1 else float("nan")

    return PairedComparison(
        model=model,
        reference=reference,
        metric=metric,
        n_folds=len(wide),
        model_mean=float(a.mean()),
        reference_mean=float(b.mean()),
        difference=float(differences.mean()),
        cohens_d=_paired_d(differences),
        wilcoxon_p=wilcoxon_p,
        ttest_p=ttest_p,
    )


def min_attainable_wilcoxon_p(n_folds: int) -> float:
    """Smallest two-sided p the signed-rank test can produce with ``n_folds`` pairs.

    This is a hard floor of the test, not a property of the data: with all
    differences pointing the same way the p-value is ``2 / 2**n``.  At n=5 that
    is 0.0625, so **a five-fold Wilcoxon test can never reach p<0.05 no matter
    how large the effect**.  Benchmarks that report five or ten folds and quote
    significance often trip on this.  The experiments here use six or more folds
    for exactly this reason, and :func:`compare_against` warns when a comparison
    is under-powered by construction.
    """
    return min(1.0, 2.0 / (2.0**n_folds)) if n_folds > 0 else float("nan")


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values.

    Uniformly more powerful than Bonferroni at the same family-wise error rate.
    NaN inputs (a test that could not be computed) pass through as NaN rather
    than being counted in the family.
    """
    p = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p)
    adjusted = np.full_like(p, np.nan)
    if not finite.any():
        return adjusted

    values = p[finite]
    order = np.argsort(values)
    m = values.size
    running = 0.0
    out = np.empty(m)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        out[index] = min(running, 1.0)
    adjusted[finite] = out
    return adjusted


def compare_against(
    per_fold: pd.DataFrame,
    reference: str,
    metric: str,
    models: Optional[Sequence[str]] = None,
    model_column: str = "model",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare every model against one reference, with Holm correction.

    This is the table that decides whether the fragment hypothesis survives:
    ``reference`` is the atom-only ablation (or the naive baseline), the models
    are the fragment-aware variants, and the metric is the protocol's primary
    metric.
    """
    candidates = models or [m for m in per_fold[model_column].unique() if m != reference]
    comparisons = [
        c
        for model in candidates
        if (c := paired_compare(per_fold, model, reference, metric, model_column)) is not None
    ]
    if not comparisons:
        return pd.DataFrame()

    frame = pd.DataFrame([c.as_row() for c in comparisons])
    frame["p_adjusted"] = holm_bonferroni(frame["p_wilcoxon"].to_numpy())
    frame["significant"] = frame["p_adjusted"] < alpha

    n_folds = int(frame["n_folds"].max())
    floor = min_attainable_wilcoxon_p(n_folds)
    if floor > alpha:
        import warnings

        warnings.warn(
            f"{n_folds} folds: the Wilcoxon test cannot produce p < {alpha} "
            f"(its floor is {floor:.4f}), so no comparison can be marked significant. "
            "Use >= 6 folds, or read the effect sizes and the t-test column instead.",
            stacklevel=2,
        )
    frame.attrs["min_attainable_p"] = floor
    return frame.sort_values("improvement", ascending=False).reset_index(drop=True)


def bootstrap_per_drug_ci(
    predictions: pd.DataFrame,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap confidence interval, resampling *drugs* rather than rows.

    Rows sharing a drug are not independent -- they are the same molecule
    measured on different cell lines -- so a row-level bootstrap understates the
    interval badly.  Resampling whole drugs respects the dependence structure
    and is the right unit under LDO, where a drug is what gets held out.
    """
    rng = np.random.default_rng(seed)
    drugs = predictions["drug_id"].to_numpy()
    unique = np.unique(drugs)
    by_drug = {d: predictions[drugs == d] for d in unique}

    point = metric_fn(predictions)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        resampled = pd.concat([by_drug[d] for d in chosen], ignore_index=True)
        samples[i] = metric_fn(resampled)

    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return {"point": point, "lower": float("nan"), "upper": float("nan")}
    return {
        "point": float(point),
        "lower": float(np.quantile(finite, alpha / 2)),
        "upper": float(np.quantile(finite, 1 - alpha / 2)),
        "n_boot": int(finite.size),
    }


def summarise_significance(frame: pd.DataFrame) -> str:
    """A one-paragraph, plain-language reading of a comparison table."""
    if frame.empty:
        return "No comparisons could be computed."
    wins = frame[frame["significant"] & (frame["improvement"] > 0)]
    losses = frame[frame["significant"] & (frame["improvement"] < 0)]
    lines = [
        f"{len(frame)} comparisons against {frame['reference'].iloc[0]} "
        f"on {frame['metric'].iloc[0]} (Wilcoxon signed-rank over paired folds, "
        f"Holm-corrected, alpha=0.05):",
        f"- {len(wins)} significantly better, {len(losses)} significantly worse, "
        f"{len(frame) - len(wins) - len(losses)} not distinguishable.",
    ]
    if not wins.empty:
        best = wins.iloc[0]
        lines.append(
            f"- Largest confirmed gain: {best['model']} "
            f"({best['improvement']:+.4f}, d={best['cohens_d']:.2f}, "
            f"p_adj={best['p_adjusted']:.4f})."
        )
    return "\n".join(lines)
