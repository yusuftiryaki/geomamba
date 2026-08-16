"""Every grid must produce valid, correctly-typed configs before it is queued.

A grid that dies on run 3 of 24 costs a whole GPU session, so this walks all
grid files and builds the real config objects without training anything.
"""
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from geomamba.models.model import GeoMamba, ModelConfig
from geomamba.train.trainer import TrainConfig
from geomamba.utils.config import load_config

GRIDS = sorted(Path("configs/grid").glob("*.yaml"))


def _runs():
    for grid in GRIDS:
        spec = yaml.safe_load(grid.read_text())
        for run in spec["runs"]:
            yield grid.name, run


@pytest.mark.parametrize("grid_name,run", list(_runs()),
                         ids=[f"{g}:{r['name']}" for g, r in _runs()])
def test_grid_run_config_is_valid(grid_name, run):
    overrides = [str(o).strip() for o in run.get("overrides", [])]
    cfg = load_config(run["config"], overrides)

    model_cfg = ModelConfig(n_features=3, **cfg.get("model", {}))
    GeoMamba(model_cfg)  # constructs => shapes and flags are consistent

    train_cfg = TrainConfig(**cfg.get("trainer", {}))
    types = {f.name: f.type for f in fields(TrainConfig)}
    for name in ("lr", "weight_decay", "grad_clip"):
        assert isinstance(getattr(train_cfg, name), float), \
            f"{grid_name}:{run['name']} {name} is {type(getattr(train_cfg, name))}"
    for name in ("steps", "batch_size", "warmup_steps", "eval_every"):
        assert isinstance(getattr(train_cfg, name), int), \
            f"{grid_name}:{run['name']} {name} is {type(getattr(train_cfg, name))}"
    assert types  # dataclass fields resolved


def test_scientific_notation_override_is_numeric():
    """YAML 1.1 reads `1e-4` as a string; a grid using it cost a GPU session."""
    cfg = load_config("configs/exp/rb_mamba.yaml", ["trainer.lr=1e-4"])
    assert cfg["trainer"]["lr"] == pytest.approx(1e-4)
    assert isinstance(TrainConfig(**cfg["trainer"]).lr, float)


def test_string_overrides_stay_strings():
    cfg = load_config("configs/exp/rb_mamba.yaml", ["model.scan_mode=chunked"])
    assert cfg["model"]["scan_mode"] == "chunked"
