"""Evaluate a finished run on the test split: forecast, physics drift, anomaly.

Usage:
  python scripts/evaluate.py --run experiments/runs/<dir> \
      [--horizons 1 10 50 100] [--rollout-context 64] [--anom data/xxx_anom.npz]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from geomamba.data.datamodule import TrajectoryDataset, load_splits
from geomamba.models.model import GeoMamba, ModelConfig
from geomamba.train import anomaly as A
from geomamba.train import metrics as M
from geomamba.train import physics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 50, 100])
    ap.add_argument("--rollout-context", type=int, default=64)
    ap.add_argument("--anom", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((args.run / "config.yaml").read_text())
    model = GeoMamba(ModelConfig(n_features=cfg["n_features"], **cfg.get("model", {})))
    ckpt_path = args.run / "best.pt"
    if not ckpt_path.exists():  # run hit the wall-clock cap before first eval
        ckpt_path = args.run / "last.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    torch.set_num_threads(cfg.get("trainer", {}).get("num_threads", 3))

    data_path = Path(cfg["data"]["path"])
    raw = np.load(data_path)
    mean = raw["train"].mean(axis=(0, 1), keepdims=True)
    std = raw["train"].std(axis=(0, 1), keepdims=True) + 1e-8
    test_ds = TrajectoryDataset(raw["test"], mean, std)
    loader = DataLoader(test_ds, batch_size=16)

    result: dict = {"run": str(args.run), "ckpt_step": int(ckpt.get("step", -1))}
    result["test_mse"] = M.teacher_forced_mse(model, loader)
    result["persistence_mse"] = M.persistence_mse(loader)
    result.update(
        M.rollout_mse(model, loader, args.rollout_context, tuple(args.horizons),
                      max_batches=8)
    )

    # physics drift on a long rollout, if the dataset carries invariant params
    if "test_params" in raw or "lorentz" in data_path.stem:
        n_eval = min(64, len(raw["test"]))
        traj = torch.from_numpy(
            ((raw["test"][:n_eval] - mean) / std).astype(np.float32)
        )
        ctx = args.rollout_context
        horizon = traj.shape[1] - 1 - ctx
        preds = model.rollout(traj[:, :ctx].to(device), horizon).cpu().numpy() * std + mean
        if "rigid_body" in data_path.stem:
            result.update(
                physics.rigid_body_drifts(preds, raw["test_params"][:n_eval])
            )
        elif "particle" in data_path.stem:
            result.update(physics.particle_drifts(preds))

    if args.anom is not None:
        araw = np.load(args.anom)
        x = torch.from_numpy(((araw["x"] - mean) / std).astype(np.float32))
        scores = A.residual_scores(model, x)
        labels = araw["labels"][:, 1:]
        result["anomaly_auroc"] = A.auroc(scores, labels)
        result["detection_delay"] = A.detection_delay(scores, labels)
        ts = A.two_sided_scores(scores)
        result["anomaly_auroc_2s"] = A.auroc(ts, labels)
        result["detection_delay_2s"] = A.detection_delay(ts, labels)

    (args.run / "eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
