"""Train and evaluate DAF-DRP(BRICS).

    python train.py                                  # WA, the paper's best config
    python train.py --drug-fusion C --global-fusion C
    python train.py --test-mode LPO                  # the paper's protocol
    python train.py --set train.max_epochs=40

Data loading, cross-validation splits and metrics come from ``fragdrp`` so the
numbers are directly comparable to the other models in this repository — the
folds are the same rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from data import ATOM_DIM, DualDataset, build_drug_graphs, collate  # noqa: E402
from fragment_vocab import build_vocabulary, coverage_report  # noqa: E402
from model import FUSION_CONFIGS, DAFDRPBrics  # noqa: E402

from fragdrp.config import DataConfig, FragmenterConfig, _deep_merge, parse_override  # noqa: E402
from fragdrp.data.dataset import ResponseScaler, StandardScaler  # noqa: E402
from fragdrp.data.loader import load_response_data  # noqa: E402
from fragdrp.data.splits import make_folds  # noqa: E402
from fragdrp.train.metrics import compute_metrics, format_metrics, primary_metric  # noqa: E402
from fragdrp.train.trainer import resolve_device, set_seed  # noqa: E402

LOGGER = logging.getLogger("daf_drp_brics")


def load_config(path: Path, overrides: Optional[List[str]] = None) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for expr in overrides or []:
        cfg = _deep_merge(cfg, parse_override(expr))
    return cfg


def make_loader(dataset: DualDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate, pin_memory=torch.cuda.is_available(),
    )


def run_fold(cfg: dict, data, fold, drug_graphs, vocab_size: int) -> Dict[str, object]:
    """Train one fold and evaluate on its held-out part."""
    tcfg, mcfg = cfg["train"], cfg["model"]
    device = resolve_device(tcfg["device"])
    set_seed(int(tcfg["seed"]) + fold.index)

    # Scalers fit on the training fold only.
    train_cells = sorted({str(c) for c in data.cell_line_ids[fold.train]})
    cell_scaler = StandardScaler().fit(
        np.stack([data.cell_line_features[c] for c in train_cells])
    )
    response_scaler = ResponseScaler(
        enabled=cfg["data"]["response_transform"] == "standard"
    ).fit(data.response[fold.train])

    sets = {
        name: DualDataset(data, rows, drug_graphs, cell_scaler, response_scaler)
        for name, rows in (("train", fold.train), ("val", fold.validation), ("test", fold.test))
    }
    loaders = {
        "train": make_loader(sets["train"], int(tcfg["batch_size"]), True),
        "val": make_loader(sets["val"], int(tcfg["batch_size"]) * 2, False),
        "test": make_loader(sets["test"], int(tcfg["batch_size"]) * 2, False),
    }

    model = DAFDRPBrics(
        n_genes=data.n_genes,
        atom_feature_dim=ATOM_DIM,
        vocab_size=vocab_size,
        dim=int(mcfg["dim"]),
        atom_layers=int(mcfg["atom_layers"]),
        frag_layers=int(mcfg["frag_layers"]),
        dropout=float(mcfg["dropout"]),
        drug_fusion=mcfg["drug_fusion"],
        global_fusion=mcfg["global_fusion"],
        head_dims=list(mcfg["head_dims"]),
        fix_degenerate_attention=bool(mcfg.get("fix_degenerate_attention", False)),
    ).to(device)
    if fold.index == 0:
        LOGGER.info(model.describe())

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )
    criterion = nn.MSELoss()
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    @torch.no_grad()
    def predict(loader) -> Dict[str, object]:
        model.eval()
        preds, targets, drugs, cells = [], [], [], []
        for batch in loader:
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch)["prediction"]
            preds.append(out.float().cpu().numpy())
            targets.append(batch.raw_response.cpu().numpy())
            drugs += batch.drug_ids
            cells += batch.cell_line_ids
        return {
            "predictions": response_scaler.inverse_transform(np.concatenate(preds)),
            "targets": np.concatenate(targets),
            "drug_ids": drugs,
            "cell_line_ids": cells,
        }

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val, best_epoch, stale = float("inf"), 0, 0
    started = time.time()

    for epoch in range(int(tcfg["max_epochs"])):
        model.train()
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(batch)["prediction"], batch.target)
            scaler.scale(loss).backward()
            if float(tcfg.get("grad_clip", 0.0)) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

        out = predict(loaders["val"])
        val_metrics = compute_metrics(
            out["targets"], out["predictions"], out["drug_ids"], out["cell_line_ids"]
        )
        val_mse = val_metrics.get("mse", float("inf"))

        if val_mse < best_val - 1e-6:
            best_val, best_epoch, stale = val_mse, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if epoch % 10 == 0 or stale == 0:
            LOGGER.info("  epoch %3d | val mse %.4f | %s%s", epoch, val_mse,
                        format_metrics(val_metrics, ("pearson", "partial_correlation_drug")),
                        "  *" if stale == 0 else "")
        if stale >= int(tcfg["patience"]):
            LOGGER.info("  early stop at epoch %d (best %d)", epoch, best_epoch)
            break

    model.load_state_dict(best_state)
    out = predict(loaders["test"])
    metrics = compute_metrics(
        out["targets"], out["predictions"], out["drug_ids"], out["cell_line_ids"]
    )
    LOGGER.info("  fold %d test | %s", fold.index, format_metrics(metrics))
    return {
        "fold": fold.index,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "seconds": time.time() - started,
        "n_parameters": model.count_parameters(),
        "predictions": out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--drug-fusion", choices=["C", "W", "A"], default=None)
    parser.add_argument("--global-fusion", choices=["C", "W", "A"], default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--test-mode", default=None, choices=["LPO", "LCO", "LDO", "LTO"])
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--fix-attention", action="store_true",
                        help="Use the non-degenerate cross-attention instead of Eq. 6 as printed")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout, force=True,
    )

    cfg = load_config(Path(args.config), args.overrides)
    if args.drug_fusion:
        cfg["model"]["drug_fusion"] = args.drug_fusion
    if args.global_fusion:
        cfg["model"]["global_fusion"] = args.global_fusion
    if args.fix_attention:
        cfg["model"]["fix_degenerate_attention"] = True
    if args.dataset:
        cfg["data"]["dataset_name"] = args.dataset
    if args.test_mode:
        cfg["data"]["test_mode"] = args.test_mode
    if args.folds:
        cfg["data"]["n_cv_splits"] = args.folds
    if args.quick:
        cfg["train"]["max_epochs"] = 5
        cfg["train"]["patience"] = 3
        cfg["data"]["n_cv_splits"] = 2

    code = f"{cfg['model']['drug_fusion']}{cfg['model']['global_fusion']}"
    # The attention variant must be part of the run name: it changes the model,
    # and without it a --fix-attention run silently overwrites the faithful one.
    variant = "-fixattn" if cfg["model"].get("fix_degenerate_attention") else ""
    run_name = (
        f"dafbrics_{code}{variant}_"
        f"{cfg['data']['dataset_name']}_{cfg['data']['test_mode']}"
    )
    out_dir = Path(cfg["output_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=" * 74)
    LOGGER.info("DAF-DRP (BRICS)  config %s  |  %s  |  %s",
                code, cfg["data"]["dataset_name"], cfg["data"]["test_mode"])
    LOGGER.info("=" * 74)

    data_cfg = DataConfig(**{k: v for k, v in cfg["data"].items()})
    data = load_response_data(data_cfg)
    LOGGER.info(data.summary())
    folds = make_folds(data, data_cfg, seed=int(cfg["train"]["seed"]))

    frag_cfg = FragmenterConfig(
        method=cfg["fragmenter"]["method"],
        max_fragments=int(cfg["fragmenter"]["max_fragments"]),
        min_fragment_atoms=int(cfg["fragmenter"]["min_fragment_atoms"]),
    )
    vocabulary = build_vocabulary(
        list(data.smiles.values()), frag_cfg,
        max_size=int(cfg["model"]["vocab_size"]),
        min_count=int(cfg["fragmenter"].get("min_vocab_count", 1)),
    )
    coverage = coverage_report(vocabulary, data.smiles, frag_cfg)
    LOGGER.info(
        "Vocabulary coverage over all drugs: %.1f%% of fragments, %.1f%% of atoms, "
        "%.1f%% of drugs fully covered",
        100 * coverage["fragment_coverage"], 100 * coverage["atom_coverage"],
        100 * coverage["fully_covered_drugs"],
    )

    drug_graphs = build_drug_graphs(data.smiles, vocabulary, frag_cfg)

    rows, predictions = [], []
    for fold in folds:
        LOGGER.info("--- fold %d/%d ---", fold.index + 1, len(folds))
        outcome = run_fold(cfg, data, fold, drug_graphs, int(cfg["model"]["vocab_size"]))
        rows.append({
            "run": run_name, "model": f"DAF-DRP(BRICS)_{code}",
            "dataset": cfg["data"]["dataset_name"], "test_mode": cfg["data"]["test_mode"],
            "fusion": code, "fold": fold.index,
            "best_epoch": outcome["best_epoch"], "seconds": outcome["seconds"],
            "n_parameters": outcome["n_parameters"], **outcome["metrics"],
        })
        p = outcome["predictions"]
        predictions.append(pd.DataFrame({
            "fold": fold.index, "drug_id": p["drug_ids"], "cell_line_id": p["cell_line_ids"],
            "target": p["targets"], "prediction": p["predictions"],
        }))

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "per_fold_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    metric = primary_metric(cfg["data"]["test_mode"])
    numeric = frame.select_dtypes("number").drop(columns=["fold"], errors="ignore")
    summary = {"run": run_name, "fusion": code, "vocabulary_size": len(vocabulary.index_of),
               **{f"{c}_mean": float(numeric[c].mean()) for c in numeric.columns},
               **{f"{c}_std": float(numeric[c].std(ddof=1)) for c in numeric.columns},
               **{f"coverage_{k}": v for k, v in coverage.items()}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOGGER.info("=" * 74)
    LOGGER.info("%s | %s = %.4f +/- %.4f", run_name, metric,
                summary[f"{metric}_mean"], summary[f"{metric}_std"])
    LOGGER.info("RMSE %.4f | Pearson %.4f | R2 %.4f",
                summary["rmse_mean"], summary["pearson_mean"], summary["r2_mean"])
    LOGGER.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
