"""Train a model from a YAML config.

Usage:
  python scripts/train.py --config configs/exp/smoke.yaml [key.path=value ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import argparse

import torch

from geomamba.data.datamodule import load_splits
from geomamba.models.model import GeoMamba, ModelConfig
from geomamba.train.trainer import TrainConfig, Trainer
from geomamba.utils.config import load_config
from geomamba.utils.runlog import RunDir
from geomamba.utils.seed import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.get("seed", 0))

    splits = load_splits(
        cfg["data"]["path"],
        n_train=cfg["data"].get("n_train"),
        seed=cfg.get("seed", 0),
        max_len=cfg["data"].get("max_len"),
    )
    n_features = splits["train"][0]["input"].shape[-1]

    model_cfg = ModelConfig(n_features=n_features, **cfg.get("model", {}))
    model = GeoMamba(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())

    transferred = 0
    if cfg.get("init_from"):
        # transfer: load every shape-compatible tensor, leave the rest fresh.
        # Embedding/head may legitimately differ (a system's observable can be
        # a different grade), so the shared backbone is what carries over.
        src = Path(cfg["init_from"])
        ckpt_path = src / "best.pt" if (src / "best.pt").exists() else src / "last.pt"
        src_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        own = model.state_dict()
        keep = {k: v for k, v in src_state.items()
                if k in own and own[k].shape == v.shape}
        own.update(keep)
        model.load_state_dict(own)
        transferred = len(keep)
        print(f"transferred {transferred}/{len(own)} tensors from {ckpt_path}")

    run = RunDir(cfg.get("run_root", "experiments/runs"), cfg["name"])
    run.save_config({**cfg, "n_params": n_params, "n_features": n_features,
                     "transferred_tensors": transferred})
    print(f"run dir: {run.path}  params: {n_params:,}")

    trainer = Trainer(TrainConfig(**cfg.get("trainer", {})), run)
    result = trainer.fit(model, splits["train"], splits["val"])
    print(result)


if __name__ == "__main__":
    main()
