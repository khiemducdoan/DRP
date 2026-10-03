"""Correctness checks for the parts where a silent bug would produce a
plausible-looking but wrong number.

Run with:  python -m pytest tests/ -v   (or just: python tests/test_fragdrp.py)

These are not coverage tests. Each one guards a specific way this kind of
pipeline fails quietly: a fragmentation that loses atoms, a batch whose edge
indices point at the wrong molecule, an attention that does not sum to one, a
scaler fit on the test fold, a split that leaks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fragdrp.chem import functional_groups as fg
from fragdrp.chem.featurizer import featurize_drug, strip_salts
from fragdrp.chem.fragmenters import AVAILABLE_METHODS, fragment_smiles
from fragdrp.config import FeaturizerConfig, FragmenterConfig, ModelConfig, load_config, parse_override
from fragdrp.data.dataset import collate
from fragdrp.models.fragdrp import build_model
from fragdrp.models.layers import scatter_softmax, scatter_sum

# A spread of real drugs: kinase inhibitors, a salt, a macrocycle, a tiny one.
SMILES = [
    "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=NN3)C=CC4=CC=CC=N4",          # axitinib
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",  # imatinib
    "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",        # gefitinib
    "CC(C)(C)NC(=O)C1CN(CCN1)Cc1ccccc1.Cl",                              # a salt
    "CCO",                                                                # ethanol
    "c1ccccc1",                                                           # benzene
]


# --------------------------------------------------------------------------- #
# Fragmentation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", AVAILABLE_METHODS)
@pytest.mark.parametrize("smiles", SMILES)
def test_fragmentation_is_a_partition(method: str, smiles: str) -> None:
    """Every atom belongs to exactly one fragment, and no atom is invented.

    A fragmenter that drops atoms would silently shrink molecules, and the
    resulting model would still train and still report a number.
    """
    cfg = FragmenterConfig(method=method)
    if method in ("atom", "whole"):
        cfg.min_fragment_atoms = 1
    if method == "atom":
        cfg.max_fragments = 128

    result = fragment_smiles(smiles, cfg)
    assert result is not None

    mol = Chem.MolFromSmiles(smiles)
    covered = sorted(atom for frag in result.fragments for atom in frag)
    assert covered == list(range(mol.GetNumAtoms())), "fragments are not a partition"
    assert len(result.atom_to_fragment) == mol.GetNumAtoms()
    assert max(result.atom_to_fragment) < result.n_fragments


@pytest.mark.parametrize("smiles", SMILES)
def test_junction_edges_connect_different_fragments(smiles: str) -> None:
    """A junction edge must cross a fragment boundary, or the graph is wrong."""
    result = fragment_smiles(smiles, FragmenterConfig(method="brics"))
    assert result is not None
    for atom_i, atom_j, frag_i, frag_j in result.junction_bonds:
        assert frag_i != frag_j
        assert result.atom_to_fragment[atom_i] == frag_i
        assert result.atom_to_fragment[atom_j] == frag_j


def test_random_cut_is_matched_to_brics() -> None:
    """The control must match BRICS on fragment count, or E3 is confounded.

    This is the assumption the falsification test rests on: if random_cut
    produced systematically fewer or larger fragments, a performance gap would
    be explained by granularity rather than by chemistry.
    """
    brics_counts, random_counts = [], []
    for smiles in SMILES:
        b = fragment_smiles(smiles, FragmenterConfig(method="brics"))
        r = fragment_smiles(smiles, FragmenterConfig(method="random_cut"))
        assert b is not None and r is not None
        brics_counts.append(b.n_fragments)
        random_counts.append(r.n_fragments)

    mean_brics = float(np.mean(brics_counts))
    mean_random = float(np.mean(random_counts))
    assert abs(mean_random - mean_brics) < 0.25 * max(mean_brics, 1.0), (
        f"random_cut averages {mean_random:.2f} fragments vs BRICS {mean_brics:.2f}; "
        "the control is not matched"
    )


def test_random_cut_is_deterministic() -> None:
    """Same molecule, same seed, same cuts -- results must be reproducible."""
    cfg = FragmenterConfig(method="random_cut", random_cut_seed=7)
    first = fragment_smiles(SMILES[0], cfg)
    second = fragment_smiles(SMILES[0], cfg)
    assert first is not None and second is not None
    assert first.fragments == second.fragments


def test_random_cut_differs_from_brics() -> None:
    """A control that reproduced BRICS would test nothing."""
    brics = fragment_smiles(SMILES[1], FragmenterConfig(method="brics"))
    random_ = fragment_smiles(SMILES[1], FragmenterConfig(method="random_cut"))
    assert brics is not None and random_ is not None
    assert brics.fragments != random_.fragments


def test_salts_are_stripped() -> None:
    """A counter-ion is not a pharmacophore and must not become a fragment."""
    mol = Chem.MolFromSmiles(SMILES[3])
    assert len(Chem.GetMolFrags(mol)) > 1, "test molecule should be a salt"
    assert len(Chem.GetMolFrags(strip_salts(mol))) == 1


# --------------------------------------------------------------------------- #
# Functional groups
# --------------------------------------------------------------------------- #


def test_functional_group_patterns_all_compile() -> None:
    assert fg.N_FUNCTIONAL_GROUPS == len(fg.GROUP_NAMES) > 40
    for name, smarts in fg.FUNCTIONAL_GROUPS:
        assert Chem.MolFromSmarts(smarts) is not None, f"bad SMARTS for {name}"


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # Acetanilide: the nitrogen is an amide N on an aryl -- an anilide, not
        # an aniline, and the `aniline` pattern rightly excludes it.
        ("CC(=O)Nc1ccccc1", {"amide", "benzene"}),
        # An aryl amine classifies as `aniline`, not `primary_amine`: the
        # aliphatic-amine patterns exclude N bonded to an aromatic atom so the
        # two categories stay distinct.
        ("Nc1ccccc1", {"aniline", "benzene"}),
        ("c1ccc2ncncc2c1", {"quinazoline"}),                        # EGFR hinge binder
        ("CC(=O)O", {"carboxylic_acid"}),
        ("C1CNCCN1", {"piperazine"}),
        ("FC(F)(F)c1ccccc1", {"trifluoromethyl", "benzene"}),
        ("c1cnc2[nH]cnc2c1", set()),                                # only checks no crash
    ],
)
def test_functional_groups_detect_the_obvious(smiles: str, expected: set) -> None:
    """The SMARTS must fire on textbook examples, or the prior is noise."""
    found = set(fg.describe(Chem.MolFromSmiles(smiles)))
    assert expected.issubset(found), f"missing {expected - found} in {smiles}"


def test_functional_groups_are_computed_per_fragment() -> None:
    """A fragment's vector describes that fragment, not the whole molecule.

    If this leaked, every fragment of a molecule would carry the same groups and
    the fragment view would collapse to a molecule-level fingerprint.
    """
    graph = featurize_drug("x", SMILES[0], FragmenterConfig(), FeaturizerConfig())
    assert graph is not None and graph.n_fragments > 2
    rows = [tuple(r) for r in graph.frag_functional_groups]
    assert len(set(rows)) > 1, "all fragments share a functional-group vector"


# --------------------------------------------------------------------------- #
# Featurisation and batching
# --------------------------------------------------------------------------- #


def _sample(smiles: str, n_genes: int = 32) -> dict:
    graph = featurize_drug(smiles, smiles, FragmenterConfig(), FeaturizerConfig())
    assert graph is not None
    return {
        "graph": graph,
        "cell": np.random.randn(n_genes).astype(np.float32),
        "target": np.float32(0.5),
        "raw_response": np.float32(1.5),
        "drug_id": smiles,
        "cell_line_id": "cl",
    }


def test_collate_offsets_indices_correctly() -> None:
    """Edge indices must point inside their own molecule after batching.

    An off-by-one in the offset would wire molecules together and produce a
    model that trains fine and is quietly wrong.
    """
    items = [_sample(s) for s in SMILES[:4]]
    batch = collate(items)

    assert batch.n_graphs == 4
    assert batch.atom_x.shape[0] == sum(i["graph"].n_atoms for i in items)
    assert batch.frag_descriptors.shape[0] == sum(i["graph"].n_fragments for i in items)

    # Every atom edge stays within one sample.
    src, dst = batch.atom_edge_index
    assert torch.equal(batch.atom_batch[src], batch.atom_batch[dst])

    # Every junction edge stays within one sample.
    if batch.frag_edge_index.numel():
        fsrc, fdst = batch.frag_edge_index
        assert torch.equal(batch.frag_batch[fsrc], batch.frag_batch[fdst])

    # Each atom maps to a fragment belonging to the same sample.
    assert torch.equal(batch.frag_batch[batch.atom_to_frag], batch.atom_batch)


def test_collate_handles_a_single_atom_molecule() -> None:
    """Methane has no bonds; the batch must still be well formed."""
    batch = collate([_sample("C"), _sample(SMILES[0])])
    assert batch.n_graphs == 2
    assert batch.atom_x.shape[0] >= 2


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #


def test_scatter_sum_matches_numpy() -> None:
    src = torch.randn(10, 3)
    index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3])
    out = scatter_sum(src, index, 4)
    for segment in range(4):
        assert torch.allclose(out[segment], src[index == segment].sum(0), atol=1e-5)


def test_scatter_softmax_sums_to_one_per_segment() -> None:
    """Attention over a molecule must be a distribution over its fragments."""
    scores = torch.randn(12, 2)
    index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    weights = scatter_softmax(scores, index, 4)
    totals = scatter_sum(weights, index, 4)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)
    assert (weights >= 0).all()


def test_context_attention_is_not_an_identity() -> None:
    """The regression test for DAF-DRP Eq. 6.

    Attention over a single pooled vector softmaxes over one key, which is
    identically 1, so the operator ignores its query. Here the query must
    change the output -- otherwise the cell line is not conditioning anything.
    """
    from fragdrp.models.layers import ContextFragmentAttention

    torch.manual_seed(0)
    layer = ContextFragmentAttention(frag_dim=16, cell_dim=8, n_heads=2).eval()
    frag_h = torch.randn(7, 16)
    frag_batch = torch.tensor([0, 0, 0, 1, 1, 1, 1])

    out_a, weights_a = layer(frag_h, frag_batch, torch.randn(2, 8), return_attention=True)
    out_b, weights_b = layer(frag_h, frag_batch, torch.randn(2, 8), return_attention=True)

    assert not torch.allclose(out_a, out_b, atol=1e-4), "output ignores the cell line"
    assert not torch.allclose(weights_a, weights_b, atol=1e-4), "weights ignore the cell line"
    # And the weights really are per-molecule distributions.
    totals = scatter_sum(weights_a, frag_batch, 2)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "flags",
    [
        {},                                                                  # full
        {"use_frag_view": False},                                            # A1
        {"use_atom_view": False},                                            # A2-ish
        {"use_atom_view": False, "use_frag_view": False},                    # A0
        {"use_cell_conditioned_attention": False},                           # A4
        {"compose_from_atoms": False},                                       # A6-ish
        {"interaction": "concat"},
        {"interaction": "hadamard"},
    ],
)
def test_every_ablation_runs_forward_and_backward(flags: dict) -> None:
    """Each config in the ablation ladder must actually train.

    An ablation that silently fails to build would drop out of the results table
    and quietly change what the comparison means.
    """
    batch = collate([_sample(s) for s in SMILES[:3]])
    model_cfg = ModelConfig(hidden_dim=32, atom_layers=2, frag_layers=2,
                            attention_heads=2, cell_hidden_dims=[32, 16],
                            head_dims=[16], **flags)
    model = build_model(n_genes=32, model_cfg=model_cfg, feat_cfg=FeaturizerConfig())

    out = model(batch)["prediction"]
    assert out.shape == (3,)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


def test_cell_only_model_ignores_the_drug() -> None:
    """A0 must genuinely see no drug information.

    If it did, the floor would be too high and every gain measured against it
    would be understated.
    """
    model_cfg = ModelConfig(hidden_dim=32, use_atom_view=False, use_frag_view=False,
                            use_cell_conditioned_attention=False,
                            cell_hidden_dims=[32, 16], head_dims=[16],
                            interaction="concat", dropout=0.0)
    model = build_model(32, model_cfg, FeaturizerConfig()).eval()

    cell = np.random.randn(32).astype(np.float32)
    def one(smiles: str) -> dict:
        item = _sample(smiles)
        item["cell"] = cell
        return item

    # Same cell line, different drugs -> identical predictions.
    batch = collate([one(SMILES[0]), one(SMILES[1]), one(SMILES[2])])
    with torch.no_grad():
        predictions = model(batch)["prediction"]
    assert torch.allclose(predictions, predictions[0].expand(3), atol=1e-5)


def test_attention_weights_are_returned_and_normalised() -> None:
    batch = collate([_sample(s) for s in SMILES[:3]])
    model_cfg = ModelConfig(hidden_dim=32, atom_layers=2, frag_layers=2,
                            attention_heads=2, cell_hidden_dims=[32, 16], head_dims=[16])
    model = build_model(32, model_cfg, FeaturizerConfig()).eval()

    with torch.no_grad():
        out = model(batch, return_attention=True)
    assert "fragment_attention" in out
    weights = out["fragment_attention"]
    assert weights.shape[0] == batch.frag_descriptors.shape[0]
    totals = scatter_sum(weights, batch.frag_batch, batch.n_graphs)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-4)


# --------------------------------------------------------------------------- #
# Scalers and statistics
# --------------------------------------------------------------------------- #


def test_scalers_use_training_statistics_only() -> None:
    """A scaler fit on all rows leaks the test distribution into training."""
    from fragdrp.data.dataset import ResponseScaler, StandardScaler

    train = np.random.randn(50, 8).astype(np.float32) * 3 + 10
    scaler = StandardScaler().fit(train)
    transformed = scaler.transform(train.mean(axis=0))
    assert np.allclose(transformed, 0.0, atol=1e-4)

    # A constant gene must not explode.
    constant = np.concatenate([train, np.ones((50, 1), dtype=np.float32)], axis=1)
    out = StandardScaler().fit(constant).transform(constant[0])
    assert np.isfinite(out).all()

    response = ResponseScaler(enabled=True).fit(np.array([1.0, 2.0, 3.0, 4.0]))
    values = np.array([1.0, 2.5, 4.0], dtype=np.float32)
    assert np.allclose(response.inverse_transform(response.transform(values)), values, atol=1e-5)


def test_wilcoxon_floor_is_reported() -> None:
    """At n=5 the signed-rank test cannot reach p<0.05; the code must know it."""
    from fragdrp.eval.stats import min_attainable_wilcoxon_p

    assert min_attainable_wilcoxon_p(5) > 0.05
    assert min_attainable_wilcoxon_p(6) < 0.05


def test_paired_comparison_orients_metrics_correctly() -> None:
    """Lower RMSE is better, higher Pearson is better; 'improvement' must know."""
    import pandas as pd

    from fragdrp.eval.stats import paired_compare

    frame = pd.DataFrame(
        {
            "model": ["good"] * 6 + ["bad"] * 6,
            "fold": list(range(6)) * 2,
            "rmse": [1.0] * 6 + [2.0] * 6,
            "pearson": [0.8] * 6 + [0.4] * 6,
        }
    )
    for metric in ("rmse", "pearson"):
        result = paired_compare(frame, "good", "bad", metric)
        assert result is not None
        assert result.difference > 0, f"{metric}: 'good' should show as an improvement"


def test_holm_is_at_least_as_conservative_as_raw() -> None:
    from fragdrp.eval.stats import holm_bonferroni

    raw = np.array([0.001, 0.01, 0.04, 0.5])
    adjusted = holm_bonferroni(raw)
    assert (adjusted >= raw).all()
    assert (adjusted <= 1.0).all()
    assert np.all(np.diff(adjusted[np.argsort(raw)]) >= -1e-12)  # monotone


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_yaml_scientific_notation_is_coerced() -> None:
    """PyYAML reads 1e-5 as a string; the loader must not pass that through."""
    config = load_config(
        Path(__file__).resolve().parent.parent / "configs" / "base.yaml",
        ["train.weight_decay=1e-5", "train.lr=3e-4", "model.dropout=0.25"],
    )
    assert isinstance(config.train.weight_decay, float)
    assert config.train.weight_decay == pytest.approx(1e-5)
    assert config.train.lr == pytest.approx(3e-4)


def test_unknown_config_keys_are_rejected() -> None:
    """A typo must fail loudly rather than being silently ignored."""
    config_root = Path(__file__).resolve().parent.parent / "configs"
    with pytest.raises(KeyError):
        load_config(config_root / "base.yaml", ["model.hiden_dim=64"])


def test_override_parsing() -> None:
    assert parse_override("a.b.c=3") == {"a": {"b": {"c": 3}}}
    assert parse_override("x=true") == {"x": True}
    assert parse_override("x=null") == {"x": None}
    assert parse_override("x=[1, 2]") == {"x": [1, 2]}


def test_every_shipped_config_loads() -> None:
    """Every YAML in configs/ must parse and validate."""
    config_root = Path(__file__).resolve().parent.parent / "configs"
    files = [p for p in config_root.rglob("*.yaml")]
    assert len(files) > 10
    for path in files:
        config = load_config(path)
        assert config.name
        assert config.data.test_mode in ("LPO", "LCO", "LDO", "LTO")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
