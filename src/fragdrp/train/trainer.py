"""Training loop: one model, one fold.

Kept deliberately plain -- AdamW, cosine schedule with warmup, early stopping on
validation MSE, mixed precision, and the best checkpoint restored before the
test pass.  The interesting claims in this project are about representation,
not about optimisation, so the training procedure is held identical across
every model and ablation.  Any difference in the results table therefore comes
from the architecture.
"""

from __future__ import annotations

import copy
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from ..config import ExperimentConfig
from ..data.dataset import Batch, DrugResponseTorchDataset, ResponseScaler, make_loader
from .metrics import compute_metrics, format_metrics

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed every RNG that can affect a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN costs a little speed and removes a real source of
    # fold-to-fold variance that would otherwise be mistaken for a model effect.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


@dataclass
class TrainingResult:
    """Everything one fold produces."""

    best_epoch: int
    best_val_mse: float
    epochs_run: int
    train_seconds: float
    history: List[Dict[str, float]] = field(default_factory=list)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    drug_ids: Optional[List[str]] = None
    cell_line_ids: Optional[List[str]] = None


class Trainer:
    """Train and evaluate one model on one fold."""

    def __init__(self, model: nn.Module, cfg: ExperimentConfig, response_scaler: ResponseScaler):
        self.cfg = cfg
        self.device = resolve_device(cfg.train.device)
        self.model = model.to(self.device)
        self.response_scaler = response_scaler
        self.criterion = nn.MSELoss()
        self.optimizer = AdamW(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # ------------------------------------------------------------------ #

    def _lr_at(self, epoch: int) -> float:
        """Linear warmup then cosine decay to 1% of the peak."""
        cfg = self.cfg.train
        if cfg.scheduler == "none":
            return cfg.lr
        if epoch < cfg.warmup_epochs:
            return cfg.lr * (epoch + 1) / max(cfg.warmup_epochs, 1)
        if cfg.scheduler == "cosine":
            progress = (epoch - cfg.warmup_epochs) / max(cfg.max_epochs - cfg.warmup_epochs, 1)
            return cfg.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))
        return cfg.lr

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # ------------------------------------------------------------------ #

    def train_epoch(self, loader) -> float:
        self.model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                prediction = self.model(batch)["prediction"]
                loss = self.criterion(prediction, batch.target)
            self.scaler.scale(loss).backward()
            if self.cfg.train.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.detach()) * batch.n_graphs
            count += batch.n_graphs
        return total / max(count, 1)

    @torch.no_grad()
    def predict(self, loader) -> Dict[str, object]:
        """Predictions on the benchmark's original response scale."""
        self.model.eval()
        predictions, targets, drugs, cells = [], [], [], []
        for batch in loader:
            batch = batch.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(batch)["prediction"]
            predictions.append(out.float().cpu().numpy())
            targets.append(batch.raw_response.cpu().numpy())
            drugs += batch.drug_ids
            cells += batch.cell_line_ids
        return {
            "predictions": self.response_scaler.inverse_transform(np.concatenate(predictions)),
            "targets": np.concatenate(targets),
            "drug_ids": drugs,
            "cell_line_ids": cells,
        }

    def evaluate(self, loader) -> Dict[str, float]:
        out = self.predict(loader)
        return compute_metrics(
            out["targets"], out["predictions"], out["drug_ids"], out["cell_line_ids"]
        )

    # ------------------------------------------------------------------ #

    def fit(
        self,
        train_set: DrugResponseTorchDataset,
        val_set: DrugResponseTorchDataset,
        test_set: Optional[DrugResponseTorchDataset] = None,
        verbose: bool = True,
    ) -> TrainingResult:
        cfg = self.cfg.train
        train_loader = make_loader(train_set, cfg.batch_size, True, cfg.num_workers)
        val_loader = make_loader(val_set, cfg.batch_size * 2, False, cfg.num_workers)

        best_state = copy.deepcopy(self.model.state_dict())
        best_val, best_epoch, stale = float("inf"), 0, 0
        history: List[Dict[str, float]] = []
        started = time.time()

        for epoch in range(cfg.max_epochs):
            self._set_lr(self._lr_at(epoch))
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_mse = val_metrics.get("mse", float("inf"))

            history.append(
                {"epoch": epoch, "lr": self._lr_at(epoch), "train_loss": train_loss, **{
                    f"val_{k}": v for k, v in val_metrics.items()
                }}
            )

            if val_mse < best_val - 1e-6:
                best_val, best_epoch, stale = val_mse, epoch, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                stale += 1

            if verbose and (epoch % 10 == 0 or stale == 0):
                LOGGER.info(
                    "epoch %3d | train %.4f | val mse %.4f | %s%s",
                    epoch, train_loss, val_mse,
                    format_metrics(val_metrics, ("pearson", "partial_correlation_drug")),
                    "  *" if stale == 0 else "",
                )

            if stale >= cfg.patience:
                LOGGER.info("Early stopping at epoch %d (best %d)", epoch, best_epoch)
                break

        self.model.load_state_dict(best_state)
        elapsed = time.time() - started

        result = TrainingResult(
            best_epoch=best_epoch,
            best_val_mse=best_val,
            epochs_run=len(history),
            train_seconds=elapsed,
            history=history,
        )
        if test_set is not None:
            test_loader = make_loader(test_set, cfg.batch_size * 2, False, cfg.num_workers)
            out = self.predict(test_loader)
            result.predictions = out["predictions"]
            result.targets = out["targets"]
            result.drug_ids = out["drug_ids"]
            result.cell_line_ids = out["cell_line_ids"]
            result.test_metrics = compute_metrics(
                out["targets"], out["predictions"], out["drug_ids"], out["cell_line_ids"]
            )
            LOGGER.info("test | %s", format_metrics(result.test_metrics))
        return result
