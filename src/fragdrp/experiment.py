"""Run one :class:`~fragdrp.config.ExperimentConfig` across all CV folds.

This is the single entry point every script uses, so a FragDRP run, an ablation
and a fragmenter comparison all go through identical data loading, splitting,
scaling and evaluation code.  Only the config differs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ExperimentConfig, to_dict
from .data.dataset import (
    DrugResponseTorchDataset,
    build_drug_graphs,
    fit_scalers,
)
from .data.loader import ResponseData, load_response_data
from .data.splits import Fold, fold_report, make_folds
from .models.fragdrp import build_model
from .train.metrics import compute_metrics, primary_metric
from .train.trainer import Trainer, set_seed

LOGGER = logging.getLogger(__name__)


def _release_cuda() -> None:
    """Drop the finished fold's tensors and return their memory to the driver."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ExperimentContext:
    """Dataset, folds and drug graphs, loaded once and reused.

    Loading GDSC2 and featurising its drugs takes long enough that repeating it
    for each of a dozen ablations would dominate the runtime.  A context is
    built once per (dataset, test_mode) and shared; the drug-graph cache is
    additionally keyed by fragmenter, so the fragmenter comparison reuses the
    response data while correctly rebuilding the graphs.
    """

    def __init__(self, cfg: ExperimentConfig, data: Optional[ResponseData] = None):
        self.cfg = cfg
        self.data = data if data is not None else load_response_data(cfg.data)
        self.folds: List[Fold] = make_folds(self.data, cfg.data, seed=cfg.train.seed)
        self._graph_cache: Dict[str, Dict] = {}

    def drug_graphs(self, cfg: ExperimentConfig) -> Dict:
        key = f"{cfg.fragmenter.method}|{cfg.featurizer.use_descriptors}|" \
              f"{cfg.featurizer.use_functional_groups}|{cfg.featurizer.use_fragment_vocab}"
        if key not in self._graph_cache:
            self._graph_cache[key] = build_drug_graphs(
                self.data.smiles, cfg.fragmenter, cfg.featurizer
            )
        return self._graph_cache[key]

    def fold_summary(self) -> pd.DataFrame:
        return fold_report(self.data, self.folds, self.cfg.data.test_mode)


def run_fold(
    cfg: ExperimentConfig,
    context: ExperimentContext,
    fold: Fold,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train FragDRP on one fold and evaluate on its test part."""
    data = context.data
    graphs = context.drug_graphs(cfg)

    # Seeded per fold so folds are independent but the whole run is reproducible.
    set_seed(cfg.train.seed + fold.index)

    cell_scaler, response_scaler = fit_scalers(data, fold.train, cfg)
    train_set = DrugResponseTorchDataset(data, fold.train, graphs, cell_scaler, response_scaler)
    val_set = DrugResponseTorchDataset(data, fold.validation, graphs, cell_scaler, response_scaler)
    test_set = DrugResponseTorchDataset(data, fold.test, graphs, cell_scaler, response_scaler)

    model = build_model(data.n_genes, cfg.model, cfg.featurizer)
    if verbose and fold.index == 0:
        LOGGER.info(model.describe())

    trainer = Trainer(model, cfg, response_scaler)
    result = trainer.fit(train_set, val_set, test_set, verbose=verbose)

    return {
        "fold": fold.index,
        "metrics": result.test_metrics,
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "seconds": result.train_seconds,
        "n_parameters": model.count_parameters(),
        "predictions": result.predictions,
        "targets": result.targets,
        "drug_ids": result.drug_ids,
        "cell_line_ids": result.cell_line_ids,
        "model": model,
        "cell_scaler": cell_scaler,
        "response_scaler": response_scaler,
    }


def run_experiment(
    cfg: ExperimentConfig,
    context: Optional[ExperimentContext] = None,
    folds: Optional[Sequence[int]] = None,
    save: bool = True,
    verbose: bool = True,
    keep_models: bool = False,
) -> Dict[str, object]:
    """Run every fold and aggregate.

    Returns per-fold metrics, the aggregate, and (optionally) the trained models
    for downstream attribution analysis.
    """
    context = context or ExperimentContext(cfg)
    selected = [f for f in context.folds if folds is None or f.index in folds]

    per_fold: List[Dict[str, float]] = []
    predictions: List[pd.DataFrame] = []
    models = []

    for fold in selected:
        LOGGER.info("--- %s | fold %d/%d ---", cfg.name, fold.index + 1, len(selected))
        outcome = run_fold(cfg, context, fold, verbose=verbose)
        per_fold.append(
            {
                "run": cfg.name,
                "model": cfg.model.name,
                "dataset": cfg.data.dataset_name,
                "test_mode": cfg.data.test_mode,
                "fragmenter": cfg.fragmenter.method,
                "fold": fold.index,
                "best_epoch": outcome["best_epoch"],
                "epochs_run": outcome["epochs_run"],
                "seconds": outcome["seconds"],
                "n_parameters": outcome["n_parameters"],
                **outcome["metrics"],
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "fold": fold.index,
                    "drug_id": outcome["drug_ids"],
                    "cell_line_id": outcome["cell_line_ids"],
                    "target": outcome["targets"],
                    "prediction": outcome["predictions"],
                }
            )
        )
        if keep_models:
            models.append(
                {
                    "fold": fold.index,
                    "model": outcome["model"],
                    "cell_scaler": outcome["cell_scaler"],
                    "response_scaler": outcome["response_scaler"],
                }
            )
        else:
            # A sweep runs dozens of folds back to back. Without this, the
            # caching allocator holds every fold's activations and an 8 GB card
            # runs out partway through a long ablation.
            outcome["model"] = None
            _release_cuda()

    fold_frame = pd.DataFrame(per_fold)
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()

    summary = aggregate_folds(fold_frame, cfg)
    LOGGER.info(
        "%s | %s %s | %s = %.4f +/- %.4f",
        cfg.name, cfg.data.dataset_name, cfg.data.test_mode,
        primary_metric(cfg.data.test_mode),
        summary.get(f"{primary_metric(cfg.data.test_mode)}_mean", float("nan")),
        summary.get(f"{primary_metric(cfg.data.test_mode)}_std", float("nan")),
    )

    if save:
        run_dir = cfg.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.save(run_dir / "config.yaml")
        fold_frame.to_csv(run_dir / "per_fold_metrics.csv", index=False)
        prediction_frame.to_csv(run_dir / "predictions.csv", index=False)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.info("Wrote results to %s", run_dir)

    return {
        "config": cfg,
        "per_fold": fold_frame,
        "predictions": prediction_frame,
        "summary": summary,
        "models": models,
        "context": context,
    }


def aggregate_folds(fold_frame: pd.DataFrame, cfg: ExperimentConfig) -> Dict[str, object]:
    """Mean and standard deviation of every metric across folds."""
    if fold_frame.empty:
        return {}
    numeric = fold_frame.select_dtypes("number").drop(columns=["fold"], errors="ignore")
    summary: Dict[str, object] = {
        "run": cfg.name,
        "model": cfg.model.name,
        "dataset": cfg.data.dataset_name,
        "test_mode": cfg.data.test_mode,
        "fragmenter": cfg.fragmenter.method,
        "n_folds": int(len(fold_frame)),
        "primary_metric": primary_metric(cfg.data.test_mode),
    }
    for column in numeric.columns:
        summary[f"{column}_mean"] = float(numeric[column].mean())
        summary[f"{column}_std"] = float(numeric[column].std(ddof=1)) if len(numeric) > 1 else 0.0
    return summary


def collect_summaries(results: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """One row per run, for the comparison tables in the report."""
    return pd.DataFrame([r["summary"] for r in results if r.get("summary")])
