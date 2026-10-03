"""Run nf-core/drugresponseeval (drevalpy) models as baselines.

The baselines are not reimplemented here.  Each one is instantiated from
``drevalpy.models.MODEL_FACTORY``, given its own feature loaders and its own
default hyperparameters, and trained on exactly the fold indices FragDRP sees.
That is the point: the comparison is against the community's implementations on
the community's data, so a difference in the results table is a difference
between models rather than between preprocessing pipelines.

Naive baselines deserve special attention, and DAF-DRP reports none of them.

``NaivePredictor``
    Predicts the global training mean.  The absolute floor.
``NaiveDrugMeanPredictor``
    Predicts each drug's mean training response, ignoring the cell line.
    Strong under LCO; by construction useless under LDO.
``NaiveCellLineMeanPredictor``
    Predicts each cell line's mean training response, ignoring the drug.
    **Strong under LDO** -- some cell lines are simply fragile.  This is the bar
    a drug encoder has to clear, because it is what you get from knowing
    nothing about the molecule at all.
``NaiveMeanEffectsPredictor``
    Additive drug effect + cell-line effect.  The right reference under LPO,
    and the number that shows how much of a headline LPO score is marginals.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data.loader import ResponseData
from .data.splits import Fold
from .train.metrics import compute_metrics

LOGGER = logging.getLogger(__name__)

#: Baselines run by default.  The four naive predictors establish the floor for
#: each protocol; the rest are the standard feature-based models.
DEFAULT_BASELINES: Sequence[str] = (
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "NaiveCellLineMeanPredictor",
    "NaiveMeanEffectsPredictor",
    "ElasticNet",
    "RandomForest",
    "SimpleNeuralNetwork",
    "MultiViewNeuralNetwork",
)

#: Available but not run by default: kernel SVR scales quadratically in the
#: number of training pairs and gradient boosting is serial, so on a panel the
#: size of GDSC2 both cost hours for a baseline nobody claims is competitive.
#: Request them explicitly with ``--models`` if you want them.
SLOW_BASELINES: Sequence[str] = ("SVR", "GradientBoosting", "KNNRegressor")

#: Models needing omics views beyond expression.  They are kept in the default
#: set because they are legitimate competitors where the data supports them,
#: but they fail on datasets whose views do not cover every cell line; the
#: runner logs and skips those rather than aborting.
MULTI_OMICS_MODELS = {"MultiViewNeuralNetwork", "MultiViewRandomForest", "MultiViewXGBoost"}

#: Models that only ever fit one drug at a time; meaningless under LDO, where
#: the test drug has no training rows at all.
SINGLE_DRUG_MODELS = {"SingleDrugRandomForest", "SingleDrugElasticNet", "MOLIR", "SuperFELTR"}


def _to_drevalpy(data: ResponseData, rows: np.ndarray):
    """Build a drevalpy ``DrugResponseDataset`` for a subset of rows."""
    from drevalpy.datasets.dataset import DrugResponseDataset

    return DrugResponseDataset(
        response=data.response[rows],
        cell_line_ids=data.cell_line_ids[rows],
        drug_ids=data.drug_ids[rows],
        tissues=data.tissues[rows],
        dataset_name=data.dataset_name,
    )


def run_baseline_fold(
    model_name: str,
    data: ResponseData,
    fold: Fold,
    cfg: ExperimentConfig,
    hyperparameters: Optional[Dict] = None,
) -> Dict[str, object]:
    """Train and evaluate one drevalpy baseline on one fold."""
    from drevalpy.models import MODEL_FACTORY

    if model_name not in MODEL_FACTORY:
        raise KeyError(f"Unknown baseline {model_name!r}. Available: {sorted(MODEL_FACTORY)}")

    model = MODEL_FACTORY[model_name]()
    if hyperparameters is None:
        candidates = model.get_hyperparameter_set()
        hyperparameters = candidates[0] if candidates else {}
    model.build_model(hyperparameters)

    cell_input = model.load_cell_line_features(
        data_path=cfg.data.path_data, dataset_name=cfg.data.dataset_name
    )
    drug_input = model.load_drug_features(
        data_path=cfg.data.path_data, dataset_name=cfg.data.dataset_name
    )

    train = _to_drevalpy(data, fold.train)
    validation = _to_drevalpy(data, fold.validation)

    started = time.time()
    model.train(
        output=train,
        cell_line_input=cell_input,
        drug_input=drug_input,
        output_earlystopping=validation,
    )
    predictions = model.predict(
        cell_line_ids=data.cell_line_ids[fold.test],
        drug_ids=data.drug_ids[fold.test],
        cell_line_input=cell_input,
        drug_input=drug_input,
    )
    elapsed = time.time() - started

    targets = data.response[fold.test]
    drug_ids = [str(x) for x in data.drug_ids[fold.test]]
    cell_ids = [str(x) for x in data.cell_line_ids[fold.test]]

    # A baseline that cannot score a held-out entity returns NaN; treat those
    # rows as the training mean rather than dropping them, so every model is
    # scored on the same rows.
    predictions = np.asarray(predictions, dtype=np.float64)
    missing = ~np.isfinite(predictions)
    if missing.any():
        LOGGER.warning(
            "%s produced %d/%d non-finite predictions; filled with the training mean",
            model_name, int(missing.sum()), predictions.size,
        )
        predictions[missing] = float(np.mean(data.response[fold.train]))

    return {
        "model": model_name,
        "fold": fold.index,
        "seconds": elapsed,
        "predictions": predictions,
        "targets": targets,
        "drug_ids": drug_ids,
        "cell_line_ids": cell_ids,
        "metrics": compute_metrics(targets, predictions, drug_ids, cell_ids),
    }


def run_baselines(
    data: ResponseData,
    folds: List[Fold],
    cfg: ExperimentConfig,
    model_names: Sequence[str] = DEFAULT_BASELINES,
) -> pd.DataFrame:
    """Run every baseline on every fold.

    A baseline that raises is logged and skipped rather than aborting the run:
    some drevalpy models require omics views a given dataset does not ship, and
    losing one baseline should not cost the whole table.
    """
    rows: List[Dict[str, object]] = []
    for model_name in model_names:
        if cfg.data.test_mode == "LDO" and model_name in SINGLE_DRUG_MODELS:
            LOGGER.info("Skipping %s: single-drug models cannot score unseen drugs", model_name)
            continue
        for fold in folds:
            try:
                result = run_baseline_fold(model_name, data, fold, cfg)
            except Exception as error:  # noqa: BLE001 - one bad baseline must not kill the table
                LOGGER.warning("%s failed on fold %d: %s", model_name, fold.index, error)
                continue
            LOGGER.info(
                "%-28s fold %d | rmse=%.4f pearson=%.4f pcorr_drug=%.4f (%.1fs)",
                model_name, fold.index,
                result["metrics"].get("rmse", float("nan")),
                result["metrics"].get("pearson", float("nan")),
                result["metrics"].get("partial_correlation_drug", float("nan")),
                result["seconds"],
            )
            rows.append(
                {
                    "model": model_name,
                    "fold": fold.index,
                    "dataset": cfg.data.dataset_name,
                    "test_mode": cfg.data.test_mode,
                    "seconds": result["seconds"],
                    **result["metrics"],
                }
            )
    return pd.DataFrame(rows)
