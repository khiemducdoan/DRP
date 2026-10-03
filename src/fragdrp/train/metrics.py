"""Evaluation metrics.

Two families, and the distinction is the whole point:

**Global metrics** (RMSE, Pearson, Spearman, R^2) pool every test pair.  They
are what DAF-DRP reports, and they are dominated by between-drug and
between-cell-line variance.  A model that knows only "this drug is potent on
average" scores well on them.

**Partial / per-entity metrics** compute a correlation *within* each drug or
each cell line and then average.  They ask the question a clinician would:
given this cell line, does the model rank drugs correctly?  Given this drug,
does it identify the sensitive cell lines?  A model that has learned only the
marginal means scores ~0 here.

The report shows both, always, so the reader can see how much of a headline
number is structure and how much is marginals.  This mirrors what drevalpy
does, for the same reason.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats

#: Metrics where a larger value is better.
HIGHER_IS_BETTER = {
    "pearson", "spearman", "kendall", "r2",
    "partial_correlation_drug", "partial_correlation_cell_line",
    "spearman_per_drug", "spearman_per_cell_line",
}


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson r, returning nan rather than raising on degenerate input."""
    if y_true.size < 2 or np.std(y_true) < 1e-9 or np.std(y_pred) < 1e-9:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rho, nan on degenerate input.

    The constant-input check is done here rather than left to scipy: a model
    that emits a constant for a held-out drug is the expected behaviour of the
    naive predictors under LDO, and scipy would warn once per drug per fold.
    The returned value is the same nan either way.
    """
    if y_true.size < 2 or np.std(y_true) < 1e-9 or np.std(y_pred) < 1e-9:
        return float("nan")
    with np.errstate(invalid="ignore"):
        rho = stats.spearmanr(y_true, y_pred).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def _grouped_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    method: str = "pearson",
    min_group_size: int = 5,
) -> float:
    """Mean within-group correlation.

    Groups smaller than ``min_group_size`` are skipped: a correlation over three
    points is noise, and averaging it in would make the metric jitter between
    folds for reasons unrelated to the model.
    """
    correlate = _safe_pearson if method == "pearson" else _safe_spearman
    values = []
    for group in np.unique(groups):
        mask = groups == group
        if int(mask.sum()) < min_group_size:
            continue
        value = correlate(y_true[mask], y_pred[mask])
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drug_ids: Optional[Sequence[str]] = None,
    cell_line_ids: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """All metrics for one set of predictions, on the benchmark's own scale."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[finite], y_pred[finite]
    if y_true.size == 0:
        return {"n": 0}

    residual = y_true - y_pred
    total_variance = float(np.var(y_true))
    metrics: Dict[str, float] = {
        "n": int(y_true.size),
        "mse": float(np.mean(residual**2)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "pearson": _safe_pearson(y_true, y_pred),
        "spearman": _safe_spearman(y_true, y_pred),
        # R^2 against the test mean, i.e. the fraction of test variance
        # explained.  This can go negative, and should be allowed to: a model
        # worse than predicting the mean deserves a negative number.
        "r2": float(1.0 - np.mean(residual**2) / total_variance) if total_variance > 1e-12 else float("nan"),
    }

    if drug_ids is not None:
        drugs = np.asarray(drug_ids)[finite]
        metrics["partial_correlation_drug"] = _grouped_correlation(y_true, y_pred, drugs, "pearson")
        metrics["spearman_per_drug"] = _grouped_correlation(y_true, y_pred, drugs, "spearman")
    if cell_line_ids is not None:
        cells = np.asarray(cell_line_ids)[finite]
        metrics["partial_correlation_cell_line"] = _grouped_correlation(
            y_true, y_pred, cells, "pearson"
        )
        metrics["spearman_per_cell_line"] = _grouped_correlation(
            y_true, y_pred, cells, "spearman"
        )
    return metrics


def primary_metric(test_mode: str) -> str:
    """The metric a given protocol should be judged on.

    Under LDO the model must rank cell lines *within* a held-out drug, so the
    per-drug correlation is the quantity of interest; a global correlation there
    mostly measures whether the model guessed the drug's average potency.
    Under LCO the mirror image holds.
    """
    return {
        "LDO": "partial_correlation_drug",
        "LCO": "partial_correlation_cell_line",
        "LPO": "pearson",
        "LTO": "partial_correlation_cell_line",
    }.get(test_mode, "pearson")


def format_metrics(metrics: Dict[str, float], keys: Optional[Sequence[str]] = None) -> str:
    keys = keys or ("rmse", "pearson", "spearman", "r2",
                    "partial_correlation_drug", "partial_correlation_cell_line")
    return "  ".join(
        f"{k}={metrics[k]:.4f}" for k in keys if k in metrics and np.isfinite(metrics.get(k, np.nan))
    )
