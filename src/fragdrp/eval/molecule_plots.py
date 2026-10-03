"""Draw molecules with their fragmentation, for the report and for sanity.

Two figures live here, and the second is the one that earns its place:

``plot_drug_fragments``
    One drug, its BRICS fragments colour-coded on the structure, with each
    fragment's SMILES and detected functional groups listed beside it.  This is
    how you check that the decomposition is chemically sensible rather than
    trusting a fragment count.

``plot_fragmenter_comparison``
    The *same* molecule cut by every strategy, side by side.  This is the
    picture that makes experiment E3 legible: BRICS and RECAP cut at recognisable
    disconnections, while ``random_cut`` -- matched on fragment count -- slices
    through the middle of groups.  A reader can see in one glance what the
    control removes.

Colour follows the dataviz reference palette in slot order, and never cycles.
Fragments are additionally labelled by index on the structure and in the table,
so identity never depends on colour alone.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from ..chem import functional_groups as fg
from ..chem.featurizer import strip_salts
from ..chem.fragmenters import Fragmentation, fragment_molecule
from ..config import FragmenterConfig
from .plots import INK, MUTED, apply_style

RDLogger.DisableLog("rdApp.*")

#: The eight validated categorical slots, in their fixed order.  Fragments of a
#: molecule are spatially adjacent, which is exactly the pairlist this ordering
#: was validated against.
FRAGMENT_COLOURS: Sequence[str] = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)


def _hex_to_rgb(colour: str, alpha: float = 0.45) -> tuple:
    """RDKit wants 0-1 RGBA; the fill is light so bond lines stay readable."""
    colour = colour.lstrip("#")
    r, g, b = (int(colour[i: i + 2], 16) / 255 for i in (0, 2, 4))
    # Lighten towards white rather than using true alpha, which RDKit renders
    # inconsistently across backends.
    return (1 - alpha * (1 - r), 1 - alpha * (1 - g), 1 - alpha * (1 - b))


def render_fragmentation(
    mol: Chem.Mol,
    fragmentation: Fragmentation,
    width: int = 520,
    height: int = 380,
    label_fragments: bool = True,
) -> bytes:
    """PNG of ``mol`` with each fragment shaded in its own colour."""
    highlight_atoms: List[int] = []
    atom_colours: Dict[int, tuple] = {}
    for k, atoms in enumerate(fragmentation.fragments):
        colour = _hex_to_rgb(FRAGMENT_COLOURS[k % len(FRAGMENT_COLOURS)])
        for atom in atoms:
            highlight_atoms.append(int(atom))
            atom_colours[int(atom)] = colour

    # A bond is shaded only when both ends sit in the same fragment, so the cut
    # bonds stay visibly uncoloured -- that is where the fragmenter acted.
    highlight_bonds: List[int] = []
    bond_colours: Dict[int, tuple] = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        k = fragmentation.atom_to_fragment[i]
        if k == fragmentation.atom_to_fragment[j]:
            highlight_bonds.append(bond.GetIdx())
            bond_colours[bond.GetIdx()] = _hex_to_rgb(
                FRAGMENT_COLOURS[k % len(FRAGMENT_COLOURS)]
            )

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.addStereoAnnotation = False
    options.bondLineWidth = 2
    options.highlightBondWidthMultiplier = 16
    options.fillHighlights = True
    if label_fragments:
        # Number one atom per fragment so the structure ties to the table.
        for k, atoms in enumerate(fragmentation.fragments):
            anchor = int(atoms[len(atoms) // 2])
            mol.GetAtomWithIdx(anchor).SetProp("atomNote", str(k))

    target = Chem.Mol(mol)
    AllChem.Compute2DCoords(target)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, target,
        highlightAtoms=highlight_atoms, highlightAtomColors=atom_colours,
        highlightBonds=highlight_bonds, highlightBondColors=bond_colours,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _fragment_rows(mol: Chem.Mol, fragmentation: Fragmentation) -> List[Dict[str, object]]:
    rows = []
    for k, atoms in enumerate(fragmentation.fragments):
        smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), canonical=True)
        sub = Chem.MolFromSmiles(smiles, sanitize=True)
        groups = fg.describe(sub) if sub is not None else []
        rows.append(
            {
                "index": k,
                "smiles": smiles,
                "n_atoms": len(atoms),
                "groups": groups,
                "colour": FRAGMENT_COLOURS[k % len(FRAGMENT_COLOURS)],
            }
        )
    return rows


def plot_drug_fragments(
    smiles: str,
    path: Path,
    name: str = "",
    frag_cfg: Optional[FragmenterConfig] = None,
) -> Optional[Path]:
    """One drug: structure with coloured fragments, plus a fragment table."""
    apply_style()
    frag_cfg = frag_cfg or FragmenterConfig()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = strip_salts(mol)
    fragmentation = fragment_molecule(mol, frag_cfg, mol_key=smiles)
    rows = _fragment_rows(mol, fragmentation)

    png = render_fragmentation(mol, fragmentation, width=620, height=440)
    image = mpimg.imread(io.BytesIO(png), format="png")

    fig = plt.figure(figsize=(11.2, max(4.4, 0.34 * len(rows) + 3.0)))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.02)

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(
        f"{name or smiles}  —  {frag_cfg.method.upper()}: "
        f"{mol.GetNumAtoms()} atoms → {fragmentation.n_fragments} fragments",
        color=INK, loc="left", fontsize=10.5,
    )

    ax = fig.add_subplot(grid[0, 1])
    ax.axis("off")
    y = 1.0
    step = 1.0 / (len(rows) + 1.5)
    for row in rows:
        ax.add_patch(
            plt.Rectangle((0.0, y - step * 0.62), 0.030, step * 0.5,
                          color=row["colour"], transform=ax.transAxes, clip_on=False)
        )
        ax.text(0.055, y - step * 0.42, f"{row['index']}  {row['smiles']}",
                transform=ax.transAxes, fontsize=9, color=INK, va="center",
                family="DejaVu Sans Mono")
        detail = ", ".join(row["groups"]) if row["groups"] else "no listed group"
        ax.text(0.055, y - step * 0.80, f"    {row['n_atoms']} atoms · {detail}",
                transform=ax.transAxes, fontsize=8, color=MUTED, va="center")
        y -= step
    ax.set_title("Fragments and detected functional groups", color=INK,
                 loc="left", fontsize=10)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def plot_fragmenter_comparison(
    smiles: str,
    path: Path,
    methods: Sequence[str] = ("brics", "recap", "murcko", "random_cut"),
    name: str = "",
) -> Optional[Path]:
    """The same molecule under several strategies -- the picture behind E3.

    Put BRICS next to the count-matched ``random_cut`` control and the argument
    becomes visual: both produce a similar number of pieces, but only one of
    them cuts where a chemist would.
    """
    apply_style()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = strip_salts(mol)

    columns = min(len(methods), 4)
    rows_n = (len(methods) + columns - 1) // columns
    fig, axes = plt.subplots(rows_n, columns, figsize=(3.5 * columns, 3.3 * rows_n))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, method in zip(axes, methods):
        cfg = FragmenterConfig(method=method)
        if method in ("atom", "whole"):
            cfg.min_fragment_atoms = 1
        fragmentation = fragment_molecule(mol, cfg, mol_key=smiles)
        png = render_fragmentation(mol, fragmentation, 420, 330, label_fragments=False)
        ax.imshow(mpimg.imread(io.BytesIO(png), format="png"))
        ax.axis("off")
        control = method in ("random_cut", "whole", "atom")
        ax.set_title(
            f"{method}  ({fragmentation.n_fragments} fragments)"
            + ("  — control" if control else ""),
            color=INK if not control else MUTED, fontsize=10, loc="left",
        )
    for ax in axes[len(methods):]:
        ax.axis("off")

    fig.suptitle(
        f"{name or smiles} — same molecule, different fragmentation",
        color=INK, x=0.01, ha="left", fontsize=11, fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path
