"""Run the whole experimental programme end to end, then build the report.

    python scripts/run_all.py --dataset GDSC2                # everything
    python scripts/run_all.py --dataset TOYv1 --quick        # 5-minute check
    python scripts/run_all.py --dataset GDSC2 --only ablation fragmenters

Each stage is a subprocess, so one failure does not lose the stages that already
finished; the summary at the end says what ran and what did not. The report is
built regardless, because a partial report is more useful than none -- sections
whose experiment is missing say so.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from _common import PROJECT_ROOT, banner, setup_logging

LOGGER = logging.getLogger("run_all")

#: name -> (script, extra arguments).  Order is the running order.
STAGES = {
    "download": ("01_download_data.py", []),
    "baselines": ("02_run_baselines.py", []),
    "ablation": ("03_ablation.py", []),
    "fragmenters": ("04_fragmenter_comparison.py", []),
    "protocols": ("05_protocol_comparison.py", ["--with-baselines"]),
    "budget": ("06_drug_budget.py", []),
    "attribution": ("07_attribution.py", []),
}

#: Stages that take a --test-mode; `download` and `protocols` do not.
TAKES_TEST_MODE = {"baselines", "ablation", "fragmenters", "budget", "attribution"}
TAKES_DATASET_FLAG = {"baselines", "ablation", "fragmenters", "protocols", "budget",
                      "attribution"}


def run_stage(name: str, args: argparse.Namespace) -> tuple[bool, float]:
    script, extra = STAGES[name]
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script)]

    if name == "download":
        command += ["--datasets", args.dataset]
    else:
        if name in TAKES_DATASET_FLAG:
            command += ["--dataset", args.dataset]
        if name in TAKES_TEST_MODE:
            command += ["--test-mode", args.test_mode]
        command += ["--output-dir", args.output_dir, "--device", args.device,
                    "--seed", str(args.seed)]
        if args.folds:
            command += ["--folds", str(args.folds)]
        if args.quick:
            command.append("--quick")
        for override in args.overrides:
            command += ["--set", override]
    command += extra

    banner(f"STAGE {name}: {' '.join(command[1:])}")
    started = time.time()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.time() - started

    if result.returncode == 0:
        LOGGER.info("stage %s finished in %.1f min", name, elapsed / 60)
    else:
        LOGGER.error("stage %s FAILED (exit %d) after %.1f min",
                     name, result.returncode, elapsed / 60)
    return result.returncode == 0, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="GDSC2")
    parser.add_argument("--test-mode", default="LDO")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", nargs="+", default=None, choices=list(STAGES),
                        help="Run only these stages")
    parser.add_argument("--skip", nargs="+", default=[], choices=list(STAGES))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    setup_logging(log_file=Path(args.output_dir) / "run_all.log")
    selected = [s for s in (args.only or STAGES) if s not in args.skip]

    banner(f"FragDRP | dataset={args.dataset} | protocol={args.test_mode} | "
           f"stages: {', '.join(selected)}")

    outcomes = {}
    total = time.time()
    for name in selected:
        outcomes[name] = run_stage(name, args)

    if not args.no_report:
        command = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "08_make_report.py"),
            "--dataset", args.dataset, "--test-mode", args.test_mode,
            "--results-dir", args.output_dir,
        ]
        banner("STAGE report")
        subprocess.run(command, cwd=PROJECT_ROOT)

    banner("Summary")
    for name, (ok, elapsed) in outcomes.items():
        LOGGER.info("  %-14s %-8s %6.1f min", name, "ok" if ok else "FAILED", elapsed / 60)
    LOGGER.info("  %-14s %-8s %6.1f min", "TOTAL", "", (time.time() - total) / 60)

    if any(not ok for ok, _t in outcomes.values()):
        LOGGER.warning(
            "Some stages failed; REPORT.md marks their sections as not run."
        )


if __name__ == "__main__":
    main()
