"""Figures for the report.

House rules, applied once in :func:`apply_style` so every figure obeys them:

* recessive axes -- no top/right spine, a faint horizontal grid only;
* a fixed three-hue categorical palette, assigned by *role* (fragment-aware /
  atom-only reference / control) and never cycled by rank, so a colour means the
  same thing in every figure;
* naive baselines are drawn as a neutral reference rule rather than as a series,
  because they are the bar to clear, not a competitor;
* one y-axis per plot, always;
* a legend whenever more than one series is present.

The palette is the validated default from the dataviz reference: slots 1-3,
which clear the all-pairs colour-vision separation gates in both light and dark
mode.  Three is the cap for scatter-type forms, so no figure here uses more.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated categorical slots 1-3.
BLUE = "#2a78d6"     # the proposed model / fragment-aware variants
ORANGE = "#eb6834"   # the reference to beat (atom-only, or the prior method)
AQUA = "#1baf7a"     # controls (random cuts, degenerate fragmenters)
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9892"  # naive baselines: a rule, not a series

ROLE_COLOURS: Dict[str, str] = {
    "proposed": BLUE,
    "reference": ORANGE,
    "control": AQUA,
    "naive": NEUTRAL,
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _bar_labels(ax, bars, values, errors=None, fmt: str = "{:.3f}") -> None:
    """Direct value labels: with fewer than a dozen bars they beat an axis read.

    Labels clear the error-bar cap rather than the bar top, otherwise they
    collide with it exactly where the reader is trying to compare.
    """
    errors = list(errors) if errors is not None else [0.0] * len(values)
    for bar, value, error in zip(bars, values, errors):
        if not np.isfinite(value):
            continue
        error = error if np.isfinite(error) else 0.0
        top = value + (error if value >= 0 else -error)
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, top),
            textcoords="offset points",
            xytext=(0, 5 if value >= 0 else -13),
            ha="center",
            fontsize=8,
            color=MUTED,
        )


# --------------------------------------------------------------------------- #


def plot_ablation(
    summary: pd.DataFrame,
    metric: str,
    roles: Dict[str, str],
    title: str,
    path: Path,
    naive_reference: Optional[float] = None,
    naive_label: str = "naive baseline",
    ylabel: Optional[str] = None,
) -> Path:
    """Bar chart of one metric across model variants, with fold error bars.

    ``roles`` maps each model name to ``proposed`` / ``reference`` / ``control``
    so the colour carries meaning rather than position.
    """
    apply_style()
    frame = summary.dropna(subset=[f"{metric}_mean"]).copy()
    labels = frame["label"].tolist()
    means = frame[f"{metric}_mean"].to_numpy(dtype=float)
    errors = frame.get(f"{metric}_std", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
    colours = [ROLE_COLOURS.get(roles.get(name, "proposed"), BLUE) for name in frame["run"]]

    fig, ax = plt.subplots(figsize=(max(5.5, 0.85 * len(labels) + 1.8), 3.7))
    bars = ax.bar(
        np.arange(len(labels)), means, yerr=errors, width=0.62, color=colours,
        capsize=3, error_kw={"elinewidth": 1.0, "ecolor": MUTED},
        edgecolor="white", linewidth=2,  # 2px surface gap between adjacent fills
    )
    _bar_labels(ax, bars, means, errors)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel(ylabel or metric.replace("_", " "))
    ax.set_title(title, color=INK, loc="left")
    ax.grid(axis="x", visible=False)
    # Headroom so the value labels are never clipped by the axes box.
    ax.set_ylim(top=float(np.nanmax(means + errors)) * 1.16)

    if naive_reference is not None and np.isfinite(naive_reference):
        ax.axhline(naive_reference, color=NEUTRAL, linestyle="--", linewidth=1.4, zorder=1)
        # Parked outside the plotting area on the right, so it cannot collide
        # with a bar or its value label whatever the data does.
        ax.annotate(
            f"{naive_label}\n{naive_reference:.3f}",
            xy=(1.0, naive_reference), xycoords=("axes fraction", "data"),
            textcoords="offset points", xytext=(6, 0),
            ha="left", va="center", fontsize=7.8, color=MUTED,
        )

    present = {roles.get(name, "proposed") for name in frame["run"]}
    if len(present) > 1:
        ax.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, color=ROLE_COLOURS[r], label=r)
                for r in ("proposed", "reference", "control")
                if r in present
            ],
            loc="best",
        )
    return _save(fig, path)


def plot_ablation_dots(
    summary: pd.DataFrame,
    metric: str,
    roles: Dict[str, str],
    title: str,
    path: Path,
    naive_reference: Optional[float] = None,
    naive_label: str = "naive baseline",
    xlabel: Optional[str] = None,
) -> Path:
    """Dot-and-interval version of :func:`plot_ablation`.

    Bars encode magnitude by length, so their axis must start at zero -- which
    for a correlation in the 0.27-0.32 range squashes every difference into
    invisibility.  A dot plot encodes *position*, so the axis may span just the
    range of interest, and the error bars can be read against each other.  That
    is the honest form when the question is "are these different?" rather than
    "how big are these?", and it lets the overlap with the naive baseline be
    seen rather than asserted.
    """
    apply_style()
    frame = summary.dropna(subset=[f"{metric}_mean"]).copy()
    labels = frame["label"].tolist()
    means = frame[f"{metric}_mean"].to_numpy(dtype=float)
    errors = frame.get(f"{metric}_std", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=float)
    colours = [ROLE_COLOURS.get(roles.get(name, "proposed"), BLUE) for name in frame["run"]]

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, max(3.0, 0.46 * len(labels) + 1.5)))

    if naive_reference is not None and np.isfinite(naive_reference):
        ax.axvline(naive_reference, color=NEUTRAL, linestyle="--", linewidth=1.4, zorder=1)
        ax.annotate(
            f"{naive_label}  {naive_reference:.3f}",
            xy=(naive_reference, 1.0), xycoords=("data", "axes fraction"),
            textcoords="offset points", xytext=(4, -10),
            ha="left", va="top", fontsize=8, color=MUTED,
        )

    for yi, mean, error, colour in zip(y, means, errors, colours):
        error = error if np.isfinite(error) else 0.0
        ax.plot([mean - error, mean + error], [yi, yi],
                color=colour, linewidth=2.2, solid_capstyle="round", zorder=2)
        ax.plot(mean, yi, "o", markersize=8, color=colour,
                markeredgecolor="white", markeredgewidth=1.8, zorder=3)
        ax.annotate(f"{mean:.3f}", (mean + error, yi), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=8, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel or f"{metric.replace('_', ' ')}  (mean ± sd across folds)")
    ax.set_title(title, color=INK, loc="left")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.13, y=0.10)

    present = {roles.get(name, "proposed") for name in frame["run"]}
    if len(present) > 1:
        # Below the axes: any in-axes corner collides with either a value label
        # or the baseline annotation, depending on the data.
        ax.legend(
            handles=[
                plt.Line2D([], [], marker="o", linestyle="-", color=ROLE_COLOURS[r],
                           markersize=7, markeredgecolor="white", label=r)
                for r in ("proposed", "reference", "control") if r in present
            ],
            loc="upper center", bbox_to_anchor=(0.5, -0.16),
            ncol=3, handletextpad=0.5, columnspacing=1.8,
        )
    return _save(fig, path)


def plot_protocol_comparison(
    frame: pd.DataFrame, metric: str, title: str, path: Path
) -> Path:
    """Grouped bars: the same models across LPO / LCO / LDO.

    The figure's job is to show how much of a headline score survives when the
    protocol stops letting the model memorise marginals.
    """
    apply_style()
    protocols = [p for p in ("LPO", "LCO", "LDO") if p in set(frame["test_mode"])]
    models = list(dict.fromkeys(frame["label"]))
    palette = [BLUE, ORANGE, AQUA]

    x = np.arange(len(protocols))
    width = min(0.8 / max(len(models), 1), 0.3)

    fig, ax = plt.subplots(figsize=(1.9 * len(protocols) + 2.6, 3.7))
    for i, model in enumerate(models[:3]):
        block = frame[frame["label"] == model].set_index("test_mode")
        means = [block[f"{metric}_mean"].get(p, np.nan) for p in protocols]
        errors = [block.get(f"{metric}_std", pd.Series(dtype=float)).get(p, 0.0) for p in protocols]
        offset = (i - (len(models[:3]) - 1) / 2) * width
        bars = ax.bar(
            x + offset, means, width * 0.92, yerr=errors, label=model,
            color=palette[i], capsize=3,
            error_kw={"elinewidth": 1.0, "ecolor": MUTED},
            edgecolor="white", linewidth=2,
        )
        _bar_labels(ax, bars, means, errors)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{p}\n{ {'LPO': 'unseen pairs', 'LCO': 'unseen cell lines', 'LDO': 'unseen drugs'}[p] }"
         for p in protocols]
    )
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title, color=INK, loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="best")
    return _save(fig, path)


def plot_budget_curve(
    frame: pd.DataFrame, metric: str, title: str, path: Path
) -> Path:
    """How performance scales with the number of training drugs.

    The hypothesis predicts the fragment-aware model's advantage is largest when
    training drug coverage is thin, because chemical priors substitute for
    examples.  A converging pair of curves would contradict that.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    palette = [BLUE, ORANGE, AQUA]

    for i, (label, block) in enumerate(frame.groupby("label", sort=False)):
        block = block.sort_values("n_train_drugs")
        x = block["n_train_drugs"].to_numpy()
        y = block[f"{metric}_mean"].to_numpy(dtype=float)
        e = block.get(f"{metric}_std", pd.Series(np.zeros(len(block)))).to_numpy(dtype=float)
        colour = palette[i % 3]
        ax.plot(x, y, marker="o", markersize=5.5, linewidth=2, color=colour, label=label,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.fill_between(x, y - e, y + e, color=colour, alpha=0.13, linewidth=0)
        # Direct-label the line's end rather than every point.
        ax.annotate(label, (x[-1], y[-1]), textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=8.5, color=colour)

    ax.set_xlabel("number of drugs available for training")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title, color=INK, loc="left")
    ax.margins(x=0.16)
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_functional_group_enrichment(
    frame: pd.DataFrame, path: Path, top_n: int = 18, title: str = ""
) -> Path:
    """Which functional groups the cell-conditioned attention favours.

    One measure across categories, so a single hue: colour here would encode
    rank, which it must never do.  The dashed rule at 1.0 is uniform attention.
    """
    apply_style()
    frame = frame.nlargest(top_n, "enrichment").iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.0, max(3.2, 0.26 * len(frame) + 1.1)))

    y = np.arange(len(frame))
    ax.barh(y, frame["enrichment"], height=0.68, color=BLUE,
            edgecolor="white", linewidth=1.6)
    ax.axvline(1.0, color=NEUTRAL, linestyle="--", linewidth=1.3)
    ax.annotate("uniform attention", (1.0, len(frame) - 0.3),
                textcoords="offset points", xytext=(4, 0), fontsize=8, color=MUTED)

    for i, (value, n) in enumerate(zip(frame["enrichment"], frame["n_drugs"])):
        ax.annotate(f"{value:.2f}  (n={n})", (value, i), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=7.6, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels([g.replace("_", " ") for g in frame["functional_group"]], fontsize=8)
    ax.set_xlabel("attention relative to uniform")
    ax.set_title(title or "Functional groups the model attends to", color=INK, loc="left")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.22)
    return _save(fig, path)


def plot_predictions(
    predictions: pd.DataFrame, title: str, path: Path, sample: int = 6000
) -> Path:
    """Predicted vs measured, plus the per-drug residual spread.

    One series, so no legend: the title names it.  The left panel shows the
    global picture that global metrics score; the right shows what is left once
    each drug's mean is removed, which is what the per-drug metric scores.
    """
    apply_style()
    frame = predictions.sample(min(sample, len(predictions)), random_state=0)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

    ax = axes[0]
    ax.scatter(frame["target"], frame["prediction"], s=7, alpha=0.22, color=BLUE,
               linewidths=0, rasterized=True)
    lo = float(min(frame["target"].min(), frame["prediction"].min()))
    hi = float(max(frame["target"].max(), frame["prediction"].max()))
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.2, linestyle="--")
    ax.set_xlabel("measured response")
    ax.set_ylabel("predicted response")
    ax.set_title("All test pairs", color=INK, loc="left")

    ax = axes[1]
    centred = predictions.copy()
    for column in ("target", "prediction"):
        centred[column] = centred[column] - centred.groupby("drug_id")[column].transform("mean")
    centred = centred.sample(min(sample, len(centred)), random_state=0)
    ax.scatter(centred["target"], centred["prediction"], s=7, alpha=0.22, color=BLUE,
               linewidths=0, rasterized=True)
    lo = float(min(centred["target"].min(), centred["prediction"].min()))
    hi = float(max(centred["target"].max(), centred["prediction"].max()))
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.2, linestyle="--")
    ax.set_xlabel("measured, drug mean removed")
    ax.set_ylabel("predicted, drug mean removed")
    ax.set_title("Within-drug signal only", color=INK, loc="left")

    fig.suptitle(title, color=INK, x=0.005, ha="left", fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


def plot_fragment_statistics(frame: pd.DataFrame, path: Path, title: str = "") -> Path:
    """Fragment count and size by method -- the confound check for E3.

    Two panels on separate axes (never two scales on one), showing that the
    random-cut control really is matched to BRICS on both statistics, so any
    performance gap is about where the cuts land.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    colours = [AQUA if m in ("random_cut", "whole", "atom") else BLUE for m in frame["method"]]

    for ax, column, label in (
        (axes[0], "fragments_per_molecule", "fragments per molecule"),
        (axes[1], "atoms_per_fragment", "atoms per fragment"),
    ):
        bars = ax.bar(np.arange(len(frame)), frame[column], width=0.62, color=colours,
                      edgecolor="white", linewidth=2)
        _bar_labels(ax, bars, frame[column].tolist(), fmt="{:.1f}")
        ax.set_xticks(np.arange(len(frame)))
        ax.set_xticklabels(frame["method"], rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)

    axes[0].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=BLUE, label="chemistry-aware"),
            plt.Rectangle((0, 0), 1, 1, color=AQUA, label="control"),
        ],
        loc="best",
    )
    fig.suptitle(title or "Fragmentation statistics", color=INK, x=0.005, ha="left",
                 fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)
