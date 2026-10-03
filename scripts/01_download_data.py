"""Download benchmark datasets from nf-core/drugresponseeval and report their shape.

The table this prints is the first thing to read before designing an
experiment: a leave-drug-out claim needs many *drugs*, and the public panels
differ by more than an order of magnitude on that axis.

    python scripts/01_download_data.py --datasets GDSC2 CCLE CTRPv2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from _common import PROJECT_ROOT, setup_logging

LOGGER = logging.getLogger("download")


def describe(dataset_name: str, path_data: str) -> dict:
    from drevalpy.datasets import AVAILABLE_DATASETS

    dataset = AVAILABLE_DATASETS[dataset_name](path_data=path_data)
    dataset.remove_nan_responses()

    n_drugs = len(set(map(str, dataset.drug_ids)))
    n_cells = len(set(map(str, dataset.cell_line_ids)))
    n_pairs = len(dataset.response)

    smiles_csv = Path(path_data) / dataset_name / "drug_smiles.csv"
    n_smiles = len(pd.read_csv(smiles_csv)) if smiles_csv.exists() else 0

    return {
        "dataset": dataset_name,
        "responses": n_pairs,
        "drugs": n_drugs,
        "cell_lines": n_cells,
        "drugs_with_smiles": n_smiles,
        "density": round(n_pairs / max(n_drugs * n_cells, 1), 3),
        # With k folds, each LDO fold holds out roughly this many drugs.  Below
        # ~10 the per-fold estimate is too noisy to support a claim.
        "drugs_held_out_per_LDO_fold_k5": round(n_drugs / 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["GDSC2", "CCLE", "CTRPv2"])
    parser.add_argument("--path-data", default="data")
    args = parser.parse_args()

    setup_logging()
    rows = []
    for name in args.datasets:
        LOGGER.info("Fetching %s (downloads on first use, then cached)", name)
        try:
            rows.append(describe(name, args.path_data))
        except Exception as error:  # noqa: BLE001
            LOGGER.error("%s failed: %s", name, error)

    frame = pd.DataFrame(rows)
    out = Path(args.path_data) / "dataset_overview.csv"
    frame.to_csv(out, index=False)
    LOGGER.info("\n%s", frame.to_string(index=False))
    LOGGER.info("Wrote %s", out)


if __name__ == "__main__":
    main()
