"""Generate REPORT.md and its figures from whatever results exist on disk.

The report is never hand-edited, so it cannot drift from the numbers.  Sections
whose experiment has not been run say so rather than failing, which means this
script is safe to run at any point during a sweep to see where things stand.

    python scripts/08_make_report.py --dataset GDSC2 --test-mode LDO
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from _common import PROJECT_ROOT, banner, setup_logging

LOGGER = logging.getLogger("report")


def make_figures(inputs, figure_dir: Path) -> dict:
    """Render every figure the report references; skip those with no data."""
    from fragdrp.eval.plots import (
        plot_ablation,
        plot_ablation_dots,
        plot_budget_curve,
        plot_fragment_statistics,
        plot_functional_group_enrichment,
        plot_predictions,
        plot_protocol_comparison,
    )
    from fragdrp.eval.report import (
        ABLATION_LABELS,
        ABLATION_ROLES,
        FRAGMENTER_LABELS,
        FRAGMENTER_ROLES,
        naive_reference_value,
    )
    from fragdrp.train.metrics import primary_metric

    figure_dir.mkdir(parents=True, exist_ok=True)
    metric = primary_metric(inputs.test_mode)
    figures: dict = {}

    if inputs.ablation_summary is not None and not inputs.ablation_summary.empty:
        frame = inputs.ablation_summary.copy()
        frame["label"] = frame["run"].map(ABLATION_LABELS).fillna(frame["run"])
        frame["order"] = frame["run"].map({k: i for i, k in enumerate(ABLATION_LABELS)})
        frame = frame.sort_values("order", na_position="last")
        # Dot-and-interval is the primary form: the differences here are small
        # relative to fold variance, and a bar chart anchored at zero hides that.
        figures["ablation"] = plot_ablation_dots(
            frame, metric, ABLATION_ROLES,
            f"Ablation on {inputs.dataset} ({inputs.test_mode})",
            figure_dir / "ablation.png",
            naive_reference=naive_reference_value(inputs),
        )
        figures["ablation_bars"] = plot_ablation(
            frame, metric, ABLATION_ROLES,
            f"Ablation on {inputs.dataset} ({inputs.test_mode})",
            figure_dir / "ablation_bars.png",
            naive_reference=naive_reference_value(inputs),
            ylabel=metric.replace("_", " "),
        )

    if inputs.fragmenter_summary is not None and not inputs.fragmenter_summary.empty:
        frame = inputs.fragmenter_summary.copy()
        frame["label"] = frame["run"].map(FRAGMENTER_LABELS).fillna(frame["run"])
        figures["fragmenters"] = plot_ablation_dots(
            frame, metric, FRAGMENTER_ROLES,
            f"Fragmentation strategy on {inputs.dataset} ({inputs.test_mode})",
            figure_dir / "fragmenters.png",
            naive_reference=naive_reference_value(inputs),
        )

    if inputs.fragment_stats is not None and not inputs.fragment_stats.empty:
        figures["fragment_stats"] = plot_fragment_statistics(
            inputs.fragment_stats, figure_dir / "fragment_stats.png",
            title=f"Fragmentation statistics on {inputs.dataset} "
                  "(controls matched to BRICS)",
        )

    if inputs.protocol_summary is not None and not inputs.protocol_summary.empty:
        frame = inputs.protocol_summary.copy()
        base = frame["run"].str.replace(r"_(LPO|LCO|LDO|LTO)$", "", regex=True)
        frame["label"] = base.map(ABLATION_LABELS).fillna(base)
        if "test_mode" not in frame.columns and "protocol" in frame.columns:
            frame["test_mode"] = frame["protocol"]
        figures["protocols"] = plot_protocol_comparison(
            frame, "pearson",
            f"The same models under three protocols ({inputs.dataset})",
            figure_dir / "protocols.png",
        )

    if inputs.budget_summary is not None and not inputs.budget_summary.empty:
        frame = inputs.budget_summary.copy().rename(columns={"model": "label"})
        figures["budget"] = plot_budget_curve(
            frame, metric,
            f"Performance against training-drug budget ({inputs.dataset}, "
            f"{inputs.test_mode})",
            figure_dir / "budget.png",
        )

    if inputs.enrichment is not None and not inputs.enrichment.empty:
        figures["functional_groups"] = plot_functional_group_enrichment(
            inputs.enrichment, figure_dir / "functional_groups.png",
            title=f"Attention by functional group ({inputs.dataset}, {inputs.test_mode})",
        )

    predictions = (
        inputs.results_dir / "abl_a5_fragdrp_full" / "predictions.csv"
    )
    if predictions.exists():
        frame = pd.read_csv(predictions)
        if not frame.empty:
            figures["predictions"] = plot_predictions(
                frame, f"FragDRP predictions ({inputs.dataset}, {inputs.test_mode})",
                figure_dir / "predictions.png",
            )
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="GDSC2")
    parser.add_argument("--test-mode", default="LDO")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="REPORT.md")
    args = parser.parse_args()

    setup_logging()
    from fragdrp.eval.markdown import build_report
    from fragdrp.eval.report import load_inputs

    banner(f"Building report for {args.dataset} / {args.test_mode}")
    results_dir = Path(args.results_dir)
    inputs = load_inputs(results_dir, args.dataset, args.test_mode)

    # A one-line description of the loaded dataset, for the setup section.
    try:
        from fragdrp.config import load_config
        from fragdrp.data.loader import load_response_data

        cfg = load_config(
            PROJECT_ROOT / "configs" / "base.yaml",
            [f"data.dataset_name={args.dataset}", f"data.test_mode={args.test_mode}"],
        )
        inputs.data_summary = load_response_data(cfg.data).summary()
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("Could not summarise the dataset: %s", error)

    figures = make_figures(inputs, results_dir / "figures")
    LOGGER.info("Rendered %d figures", len(figures))

    text = build_report(inputs, figures)
    output = Path(args.output)
    output.write_text(text, encoding="utf-8")

    # The report references figures relative to itself.
    LOGGER.info("Wrote %s (%d lines)", output, text.count("\n") + 1)
    for name, path in figures.items():
        LOGGER.info("  figure %-20s -> %s", name, path)


if __name__ == "__main__":
    main()
