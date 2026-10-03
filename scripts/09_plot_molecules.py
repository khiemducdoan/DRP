"""Draw drugs with their fragmentation.

    python scripts/09_plot_molecules.py --dataset GDSC2 --drugs Axitinib Gefitinib
    python scripts/09_plot_molecules.py --dataset GDSC2 --top 8

Produces, per drug, a structure with colour-coded fragments and a table of the
functional groups in each, plus a side-by-side of the same molecule under
several fragmentation strategies -- the picture behind experiment E3.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from _common import PROJECT_ROOT, banner, setup_logging

LOGGER = logging.getLogger("molecules")

#: Well-known drugs whose pharmacophore a reader can check by eye.
DEFAULT_DRUGS = [
    "Axitinib", "Gefitinib", "Imatinib", "Dasatinib",
    "Erlotinib", "Sorafenib", "Lapatinib", "Palbociclib",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="GDSC2")
    parser.add_argument("--path-data", default="data")
    parser.add_argument("--drugs", nargs="+", default=None,
                        help="Drug names to draw; default is a set of familiar inhibitors")
    parser.add_argument("--top", type=int, default=None,
                        help="Instead of names, draw the first N drugs in the dataset")
    parser.add_argument("--method", default="brics")
    parser.add_argument("--compare", nargs="+",
                        default=["brics", "recap", "murcko", "random_cut"])
    parser.add_argument("--output-dir", default="results/figures/molecules")
    args = parser.parse_args()

    setup_logging()
    from fragdrp.config import FragmenterConfig
    from fragdrp.data.loader import drug_names
    from fragdrp.eval.molecule_plots import plot_drug_fragments, plot_fragmenter_comparison

    import pandas as pd

    csv = Path(args.path_data) / args.dataset / "drug_smiles.csv"
    frame = pd.read_csv(csv).dropna(subset=["canonical_smiles"])
    by_name = {str(r["drug_name"]): str(r["canonical_smiles"]) for _i, r in frame.iterrows()}

    if args.top:
        chosen = list(by_name.items())[: args.top]
    else:
        wanted = args.drugs or DEFAULT_DRUGS
        chosen = [(n, by_name[n]) for n in wanted if n in by_name]
        missing = [n for n in wanted if n not in by_name]
        if missing:
            LOGGER.warning("Not in %s: %s", args.dataset, ", ".join(missing))

    if not chosen:
        raise SystemExit(f"No matching drugs found in {csv}")

    out = Path(args.output_dir)
    banner(f"Drawing {len(chosen)} drugs from {args.dataset}")

    cfg = FragmenterConfig(method=args.method)
    for name, smiles in chosen:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        path = plot_drug_fragments(smiles, out / f"{safe}_{args.method}.png", name, cfg)
        LOGGER.info("  %-14s %s", name, path)

    # The E3 picture: one representative molecule under every strategy.
    name, smiles = chosen[0]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = plot_fragmenter_comparison(
        smiles, out / f"{safe}_methods.png", args.compare, name
    )
    LOGGER.info("Fragmenter comparison: %s", path)


if __name__ == "__main__":
    main()
