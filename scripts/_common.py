"""Shared helpers for the experiment scripts: logging, paths, argument parsing."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = PROJECT_ROOT / "configs"

# Let the scripts run from a checkout without installing the package.
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # drevalpy and matplotlib are chatty at INFO and drown out the run log.
    for noisy in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def base_parser(description: str) -> argparse.ArgumentParser:
    """Arguments every experiment script accepts."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset", default="GDSC2",
        help="Dataset name, e.g. GDSC2, CCLE, CTRPv2, TOYv1",
    )
    parser.add_argument(
        "--test-mode", default="LDO", choices=["LPO", "LCO", "LDO", "LTO"],
        help="Split protocol. LDO is the one that tests the drug encoder.",
    )
    parser.add_argument("--folds", type=int, default=None, help="Override n_cv_splits")
    parser.add_argument("--output-dir", default="results", help="Where to write results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Config override, e.g. --set train.max_epochs=40 (repeatable)",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Short run for pipeline checks, not for results")
    return parser


def common_overrides(args: argparse.Namespace) -> list[str]:
    """Translate the shared CLI flags into config overrides."""
    overrides = [
        f"data.dataset_name={args.dataset}",
        f"data.test_mode={args.test_mode}",
        f"output_dir={args.output_dir}",
        f"train.device={args.device}",
        f"train.seed={args.seed}",
    ]
    if args.folds:
        overrides.append(f"data.n_cv_splits={args.folds}")
    if args.quick:
        overrides += ["train.max_epochs=5", "train.patience=3", "data.n_cv_splits=2"]
    return overrides + list(args.overrides)


def results_dir(args: argparse.Namespace, name: str) -> Path:
    path = Path(args.output_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def banner(text: str) -> None:
    logging.getLogger("run").info("=" * 78)
    logging.getLogger("run").info(text)
    logging.getLogger("run").info("=" * 78)
