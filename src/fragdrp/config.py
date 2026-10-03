"""Typed configuration objects for FragDRP.

Every experiment is fully described by a single :class:`ExperimentConfig`.
Configs live as YAML and compose through a ``defaults`` key, so a run config
stays short:

.. code-block:: yaml

    defaults: [../base.yaml]
    model:
      use_frag_view: false

Nothing magic happens: ``load_config`` reads the YAML, recursively merges the
files listed under ``defaults`` (later entries win, and the current file wins
over all of them), then validates the result into the dataclasses below.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    """Which response data to use and how to split it."""

    dataset_name: str = "GDSC2"
    """Any key of ``drevalpy.datasets.AVAILABLE_DATASETS``: TOYv1, CCLE, GDSC1,
    GDSC2, CTRPv1, CTRPv2, ..."""

    path_data: str = "data"
    """Directory drevalpy downloads and caches the raw dataset into."""

    test_mode: str = "LDO"
    """Split protocol: ``LPO`` (random drug-cell-line pairs), ``LCO`` (unseen
    cell lines), ``LDO`` (unseen drugs).  LDO is the setting in which the drug
    representation actually has to carry information -- see EXPERIMENTS.md."""

    n_cv_splits: int = 5
    validation_ratio: float = 0.1
    """Fraction of the training part held out for early stopping."""

    response_transform: Optional[str] = "standard"
    """``standard`` z-scores the response with training-fold statistics only;
    ``null`` leaves it alone.  Metrics are always reported on the raw scale."""

    cell_line_view: str = "gene_expression"
    """Which omics view from drevalpy feeds the cell-line encoder."""

    gene_subset: Optional[str] = "landmark"
    """``landmark`` restricts expression to the LINCS L1000 landmark genes when
    drevalpy ships that view; ``null`` keeps every gene."""

    max_drugs: Optional[int] = None
    max_cell_lines: Optional[int] = None
    """Debug knobs: subsample so the pipeline runs end to end in a minute."""

    min_pairs_per_drug: int = 20
    """Drugs measured on fewer cell lines than this are dropped; they make LDO
    folds extremely noisy (a held-out drug with 3 measurements has a
    meaningless per-drug correlation)."""

    min_pairs_per_cell_line: int = 5


@dataclass
class FragmenterConfig:
    """How a molecule is cut into fragments."""

    method: str = "brics"
    """``brics``, ``recap``, ``murcko``, ``rotatable_bond``, ``random_cut``,
    ``whole`` (one fragment per molecule) or ``atom`` (one per atom).  The last
    three are controls -- see EXPERIMENTS.md, experiment E3."""

    max_fragments: int = 32
    """Fragments beyond this are merged into their largest neighbour, which
    bounds the size of the attention matrix."""

    min_fragment_atoms: int = 2
    """Fragments smaller than this merge into their largest neighbour, so a
    swarm of single atoms cannot dominate the junction graph."""

    random_cut_seed: int = 0
    """Used only by ``random_cut``: the control that cuts as many bonds as BRICS
    would, but picks which bonds at random."""

    match_reference_count: bool = True
    """``random_cut`` matches the per-molecule fragment count that BRICS
    produces, so the control differs from BRICS only in *where* cuts land."""


@dataclass
class FeaturizerConfig:
    """Fragment-level chemical priors."""

    use_descriptors: bool = True
    """Per-fragment physicochemical descriptors (MW, TPSA, logP, HBD/HBA, ...)."""

    use_functional_groups: bool = True
    """Per-fragment SMARTS indicator vector over a curated pharmacophore list."""

    functional_group_subset: str = "all"
    """Which topological subset of the SMARTS vocabulary to expose:
    ``all``, ``ring`` (groups defined by a cycle), ``star`` (tree-shaped groups
    around a central atom) or ``none``.  Experiment E6 / THEORY.md Prop. 3 tests
    whether the benefit is carried by the ring half.  The vector keeps its full
    width in every setting -- entries are masked to zero, not removed -- so the
    parameter count is identical and cannot explain a difference."""

    use_fragment_vocab: bool = False
    """Reproduces DAF-DRP's learnable fragment lookup table.  Off by default: a
    lookup table cannot represent a fragment absent from the training
    vocabulary, which is exactly the failure mode LDO exposes."""

    vocab_size: int = 798
    """Only used when ``use_fragment_vocab`` is true (DAF-DRP parity setting)."""

    cache_dir: str = "data/featurised"


@dataclass
class ModelConfig:
    """FragDRP architecture.  Ablations are expressed by toggling these flags."""

    name: str = "FragDRP"

    hidden_dim: int = 128
    dropout: float = 0.2

    # --- drug: atom view -------------------------------------------------- #
    use_atom_view: bool = True
    atom_layers: int = 4
    atom_readout: str = "sum"
    """``sum`` | ``mean`` | ``max`` | ``attention``."""

    # --- drug: fragment view ---------------------------------------------- #
    use_frag_view: bool = True
    frag_layers: int = 3
    frag_readout: str = "attention"
    compose_from_atoms: bool = True
    """Initialise each fragment state by attention-pooling the atom embeddings
    inside it.  This is what makes the fragment encoder vocabulary-free: an
    unseen fragment still receives a meaningful vector."""

    # --- cell-line-conditioned fragment attention ------------------------- #
    use_cell_conditioned_attention: bool = True
    """The cell line queries the *set* of fragment tokens.  Attention over a
    single pooled vector softmaxes over one key and is an identity map; this
    version is a real attention and yields per-fragment attributions."""
    attention_heads: int = 4

    # --- cell-line encoder ------------------------------------------------ #
    cell_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])

    # --- interaction + head ----------------------------------------------- #
    interaction: str = "bilinear"
    """``concat`` | ``hadamard`` | ``bilinear``."""
    head_dims: List[int] = field(default_factory=lambda: [256, 128])


@dataclass
class TrainConfig:
    max_epochs: int = 150
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 20
    """Early stopping patience on validation MSE."""
    grad_clip: float = 5.0
    num_workers: int = 0
    """Windows + CUDA: 0 avoids spawn overhead; the collate is cheap anyway."""
    amp: bool = True
    """Mixed precision, which halves activation memory on an 8 GB card."""
    scheduler: str = "cosine"
    """``cosine`` | ``plateau`` | ``none``."""
    warmup_epochs: int = 3
    seed: int = 42
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    """A complete, reproducible run."""

    name: str = "fragdrp_default"
    output_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    fragmenter: FragmenterConfig = field(default_factory=FragmenterConfig)
    featurizer: FeaturizerConfig = field(default_factory=FeaturizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(to_dict(self), sort_keys=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# (de)serialisation helpers
# --------------------------------------------------------------------------- #


def to_dict(obj: Any) -> Any:
    """Recursively turn dataclasses into plain dicts (YAML/JSON friendly)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _default_instance(f: dataclasses.Field) -> Any:
    """The declared default of a field, or None if it has no simple default."""
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    if f.default is not dataclasses.MISSING:
        return f.default
    return None


def _coerce(value: Any, annotation: Any) -> Any:
    """Cast a YAML scalar to the type its field declares.

    This exists because of a PyYAML trap: YAML 1.1 only recognises a float when
    it has a decimal point *and* a signed exponent, so ``1e-5`` loads as the
    **string** ``"1e-5"`` while ``1.0e-5`` loads as a number.  Silently passing
    that string through produces a confusing failure deep inside the optimiser.
    Coercing here turns it into an error at load time, or -- more usefully --
    into the number the author obviously meant.
    """
    text = str(annotation)
    if value is None or isinstance(value, (dict, list)):
        return value
    try:
        if "bool" in text and not isinstance(value, bool):
            if isinstance(value, str):
                return value.strip().lower() in ("true", "yes", "1", "on")
            return bool(value)
        if "float" in text and not isinstance(value, bool):
            return float(value)
        if "int" in text and not isinstance(value, bool):
            return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Cannot read {value!r} as {text}: {error}") from error
    return value


def _from_dict(cls, data: Optional[Dict[str, Any]]):
    """Instantiate ``cls`` from a dict, recursing into nested dataclass fields."""
    if data is None:
        return cls()
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(
            f"Unknown config key(s) {sorted(unknown)} for {cls.__name__}. "
            f"Valid keys: {sorted(known)}"
        )
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        default = _default_instance(known[key])
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _from_dict(type(default), value)
        else:
            kwargs[key] = _coerce(value, known[key].type)
    return cls(**kwargs)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ``override`` into ``base``; nested dicts merge key-wise."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_raw(path: Path | str) -> Dict[str, Any]:
    """Read one YAML file and resolve its ``defaults`` chain into a flat dict."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parents = raw.pop("defaults", []) or []
    merged: Dict[str, Any] = {}
    for parent in parents:
        merged = _deep_merge(merged, load_raw((path.parent / parent).resolve()))
    return _deep_merge(merged, raw)


def parse_override(expr: str) -> Dict[str, Any]:
    """Turn ``model.use_frag_view=false`` into ``{'model': {'use_frag_view': False}}``.

    Values are parsed as YAML, so ``3``, ``1e-4``, ``null``, ``[a, b]`` and
    ``true`` all come through with the right Python type.
    """
    if "=" not in expr:
        raise ValueError(f"Override must look like key.path=value, got {expr!r}")
    dotted, _, raw_value = expr.partition("=")
    value = yaml.safe_load(raw_value)
    node: Dict[str, Any] = {}
    cursor = node
    parts = dotted.strip().split(".")
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return node


def load_config(
    path: Path | str,
    overrides: Optional[List[str] | Dict[str, Any]] = None,
) -> ExperimentConfig:
    """Load a YAML config into a validated :class:`ExperimentConfig`.

    ``overrides`` may be a nested dict, or a list of ``a.b=value`` strings as
    produced by the CLI's ``--set`` flag.  It is merged last.
    """
    raw = load_raw(path)
    if overrides:
        if isinstance(overrides, dict):
            raw = _deep_merge(raw, overrides)
        else:
            for expr in overrides:
                raw = _deep_merge(raw, parse_override(expr))
    return _from_dict(ExperimentConfig, raw)


def config_from_dict(data: Dict[str, Any]) -> ExperimentConfig:
    return _from_dict(ExperimentConfig, data)


def config_hash(cfg: Any) -> str:
    """Short stable hash of a config; used to key on-disk caches."""
    blob = json.dumps(to_dict(cfg), sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:10]
