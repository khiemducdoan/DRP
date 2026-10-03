"""The one figure that puts every model in this repository on a single axis.

Individual experiments each answer a narrow question; this collects their
results so a reader can see the whole landscape at once -- naive predictors, the
nf-core/drugresponseeval baselines, the DAF-DRP reproduction, and the FragDRP
ablation ladder -- measured on identical folds.

Grouping is by *what kind of claim a model supports*, not by who wrote it:

``naive``      no learning; the bar every other row has to clear
``baseline``   published feature-based models from drevalpy
``daf``        the DAF-DRP reproduction
``ours``       the FragDRP ablation ladder

The vertical rule is the protocol's naive reference. Rows whose interval crosses
it are not distinguishable from predicting a per-cell-line mean, and the figure
is drawn so that this is the first thing a reader notices.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..train.metrics import primary_metric
from .plots import AQUA, BLUE, GRID, INK, MUTED, NEUTRAL, ORANGE, apply_style

LOGGER = logging.getLogger(__name__)

GROUP_COLOURS: Dict[str, str] = {
    "naive": NEUTRAL,
    "baseline": AQUA,
    "daf": ORANGE,
    "ours": BLUE,
}
GROUP_LABELS: Dict[str, str] = {
    "naive": "naive (no learning)",
    "baseline": "nf-core/drugresponseeval",
    "daf": "DAF-DRP reproduction",
    "ours": "FragDRP ablation",
}

DISPLAY_NAMES: Dict[str, str] = {
    "NaivePredictor": "global mean",
    "NaiveDrugMeanPredictor": "drug mean",
    "NaiveCellLineMeanPredictor": "cell-line mean",
    "NaiveMeanEffectsPredictor": "drug + cell-line effects",
    "ElasticNet": "ElasticNet",
    "RandomForest": "RandomForest",
    "SimpleNeuralNetwork": "SimpleNeuralNetwork",
    "MultiViewNeuralNetwork": "MultiViewNeuralNetwork",
    "abl_a0_cell_only": "A0  cell line only",
    "abl_a1_atom_only": "A1  atom GIN",
    "abl_a2_frag_only": "A2  fragment only",
    "abl_a3_atom_frag_bare": "A3  atom+frag, no priors",
    "abl_a4_atom_frag_priors": "A4  atom+frag + priors",
    "abl_a5_fragdrp_full": "A5  FragDRP (full)",
    "abl_a6_daf_style_vocab": "A6  DAF-style vocabulary",
    "abl_e6_ring_groups": "E6a  ring groups only",
    "abl_e6_star_groups": "E6b  star groups only",
}


def _pretty_name(names: pd.Series) -> pd.Series:
    """Readable labels for runs that have no entry in DISPLAY_NAMES.

    Run directories are named ``dafbrics_<fusion>_<dataset>_<protocol>``; the
    dataset and protocol are already in the figure title, so they are stripped.
    """
    cleaned = names.str.replace(
        r"_(GDSC2|CCLE|CTRPv2|GDSC1|CTRPv1|TOYv1|TOYv2)_(LPO|LCO|LDO|LTO)$", "", regex=True
    )
    cleaned = cleaned.str.replace(r"^dafbrics_", "DAF-DRP(BRICS) ", regex=True)
    cleaned = cleaned.str.replace(r"^frag_", "FragDRP / ", regex=True)
    return cleaned


def _group_of(name: str) -> str:
    if name.startswith("Naive"):
        return "naive"
    if name.startswith("dafbrics"):
        return "daf"
    if name.startswith("abl_") or name.startswith("frag_"):
        return "ours"
    return "baseline"


def collect_results(
    results_dir: Path,
    dataset: str,
    test_mode: str,
    daf_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Gather per-fold metrics from every experiment that has run.

    Anything missing is simply absent from the table rather than an error, so
    this is safe to call part-way through a sweep.
    """
    results_dir = Path(results_dir)
    frames: List[pd.DataFrame] = []

    def add(name_series, frame: pd.DataFrame) -> None:
        keep = [c for c in ("partial_correlation_drug", "rmse", "pearson", "r2",
                            "partial_correlation_cell_line") if c in frame.columns]
        frames.append(pd.DataFrame({"name": name_series, "fold": frame["fold"],
                                    **{c: frame[c] for c in keep}}))

    baselines = results_dir / "baselines" / f"per_fold_{dataset}_{test_mode}.csv"
    if baselines.exists():
        frame = pd.read_csv(baselines)
        add(frame["model"], frame)

    for path in sorted(glob.glob(str(results_dir / "abl_a*" / "per_fold_metrics.csv"))):
        frame = pd.read_csv(path)
        add(frame["run"], frame)

    if daf_dir is not None:
        pattern = str(Path(daf_dir) / f"dafbrics_*_{dataset}_{test_mode}" / "per_fold_metrics.csv")
        for path in sorted(glob.glob(pattern)):
            frame = pd.read_csv(path)
            add(frame["run"], frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_overview(
    per_fold: pd.DataFrame,
    metric: str,
    path: Path,
    title: str = "",
    naive_reference_model: str = "NaiveCellLineMeanPredictor",
    subtitle: str = "",
) -> Optional[Path]:
    """Dot-and-interval across every model, grouped and sorted by score."""
    if per_fold.empty or metric not in per_fold.columns:
        return None

    apply_style()
    stats = (
        per_fold.groupby("name")[metric]
        .agg(["mean", "std", "count"]).reset_index().dropna(subset=["mean"])
    )
    if stats.empty:
        return None
    stats["group"] = stats["name"].map(_group_of)
    stats["label"] = stats["name"].map(DISPLAY_NAMES).fillna(_pretty_name(stats["name"]))
    stats = stats.sort_values("mean", ascending=True).reset_index(drop=True)

    reference = stats.loc[stats["name"] == naive_reference_model, "mean"]
    reference_value = float(reference.iloc[0]) if not reference.empty else None

    y = np.arange(len(stats))
    fig, ax = plt.subplots(figsize=(8.4, max(3.4, 0.40 * len(stats) + 1.8)))

    # Fix the limits from the data *before* shading, so the span cannot drag the
    # autoscale out to its own left edge.
    spread = stats["std"].fillna(0.0)
    low = float((stats["mean"] - spread).min())
    high = float((stats["mean"] + spread).max())
    pad = 0.16 * max(high - low, 1e-6)
    ax.set_xlim(low - pad, high + 2.2 * pad)

    if reference_value is not None:
        ax.axvspan(low - pad, reference_value, color=GRID, alpha=0.55,
                   zorder=0, linewidth=0)
        ax.axvline(reference_value, color=NEUTRAL, linestyle="--", linewidth=1.5, zorder=1)
        ax.annotate(
            f"← not better than a per-cell-line mean ({reference_value:.3f})",
            xy=(reference_value, -0.5), xycoords=("data", "data"),
            textcoords="offset points", xytext=(-6, 0),
            ha="right", va="center", fontsize=8, color=MUTED,
        )

    for yi, row in zip(y, stats.itertuples()):
        colour = GROUP_COLOURS[row.group]
        error = row.std if np.isfinite(row.std) else 0.0
        ax.plot([row.mean - error, row.mean + error], [yi, yi],
                color=colour, linewidth=2.2, solid_capstyle="round", zorder=2)
        ax.plot(row.mean, yi, "o", markersize=7.5, color=colour,
                markeredgecolor="white", markeredgewidth=1.6, zorder=3)
        ax.annotate(f"{row.mean:.3f}", (row.mean + error, yi),
                    textcoords="offset points", xytext=(7, 0),
                    va="center", fontsize=8, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(stats["label"], fontsize=8.6)
    ax.set_xlabel(f"{metric.replace('_', ' ')}  (mean ± sd across folds)")
    ax.set_title(title or "All models on identical folds", color=INK, loc="left")
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1.0), xycoords="axes fraction",
                    textcoords="offset points", xytext=(0, 14),
                    fontsize=8.5, color=MUTED, ha="left")
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.04)

    present = list(dict.fromkeys(stats["group"]))
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", linestyle="-", color=GROUP_COLOURS[g],
                       markersize=7, markeredgecolor="white", label=GROUP_LABELS[g])
            for g in ("naive", "baseline", "daf", "ours") if g in present
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.10 - 0.3 / len(stats)),
        ncol=2, handletextpad=0.5, columnspacing=1.6,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_grid(
    per_fold: pd.DataFrame,
    path: Path,
    metrics: tuple = ("partial_correlation_drug", "pearson", "rmse", "r2"),
    title: str = "",
) -> Optional[Path]:
    """The same models across four metrics.

    Worth its own panel because the metrics disagree here: a model can predict a
    new drug's overall potency well (global Pearson, R²) while ranking cell
    lines within that drug no better than chance (per-drug r).  One number would
    hide that.
    """
    if per_fold.empty:
        return None
    apply_style()
    available = [m for m in metrics if m in per_fold.columns]
    stats = per_fold.groupby("name")[available].agg(["mean", "std"])
    stats.columns = [f"{a}__{b}" for a, b in stats.columns]
    stats = stats.reset_index()
    stats["group"] = stats["name"].map(_group_of)
    stats["label"] = stats["name"].map(DISPLAY_NAMES).fillna(_pretty_name(stats["name"]))
    order = stats.sort_values(f"{available[0]}__mean")["name"].tolist()
    stats = stats.set_index("name").loc[order].reset_index()

    fig, axes = plt.subplots(
        1, len(available), figsize=(3.3 * len(available), max(3.4, 0.36 * len(stats) + 1.6)),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    y = np.arange(len(stats))

    for ax, metric in zip(axes, available):
        lower_better = metric in ("rmse", "mse", "mae")
        for yi, row in zip(y, stats.itertuples()):
            mean = getattr(row, f"{metric}__mean")
            sd = getattr(row, f"{metric}__std")
            sd = sd if np.isfinite(sd) else 0.0
            colour = GROUP_COLOURS[row.group]
            ax.plot([mean - sd, mean + sd], [yi, yi], color=colour, linewidth=2.0,
                    solid_capstyle="round", zorder=2)
            ax.plot(mean, yi, "o", markersize=6.5, color=colour,
                    markeredgecolor="white", markeredgewidth=1.4, zorder=3)
        ax.set_title(metric.replace("_", " ") + (" ↓" if lower_better else " ↑"),
                     color=INK, loc="left", fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.16, y=0.03)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(stats["label"], fontsize=8.2)
    if title:
        fig.suptitle(title, color=INK, x=0.005, ha="left", fontsize=10.5,
                     fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
