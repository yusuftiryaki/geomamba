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
    ap.add_argument("--anomaly-rollout", action="store_true",
                    help="expensive freeze-the-regime rollout scoring")
    ap.add_argument("--rot-test", action="store_true",
                    help="also score a globally rotated test set (equivariance "
                         "probe; only meaningful where the training frame is "
                         "canonical, i.e. rigid_body)")
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
    if "test_params" in raw:
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

    if args.rot_test:
        # Equivariance probe. Euler's equations are exactly SO(3)-covariant: if
        # w(t) solves them for inertia I, then R w(t) solves them for R I R^T.
        # So rotating the test trajectories yields *valid physics* in a frame
        # the model never saw (training inertia is always diagonal and sorted).
        # A fully equivariant pipeline would score identically; the ratio says
        # how much geometry the model actually uses.
        # Caveat: normalisation is per-component and the data is anisotropic,
        # so even a perfect model cannot reach 1.0 -- persistence_rot_gap is
        # that floor, and only differences between arms should be read.
        from scipy.spatial.transform import Rotation

        rot = Rotation.random(len(raw["test"]), random_state=1234).as_matrix()
        x_rot = np.einsum("nij,ntj->nti", rot, raw["test"].astype(np.float64))
        rot_loader = DataLoader(
            TrajectoryDataset(x_rot.astype(np.float32), mean, std), batch_size=16
        )
        result["test_mse_rot"] = M.teacher_forced_mse(model, rot_loader)
        result["persistence_mse_rot"] = M.persistence_mse(rot_loader)
        result.update({
            f"{k}_rot": v
            for k, v in M.rollout_mse(model, rot_loader, args.rollout_context,
                                      tuple(args.horizons), max_batches=8).items()
        })
        result["rot_gap"] = result["test_mse_rot"] / result["test_mse"]
        result["rot_gap_h100"] = (
            result["rollout_mse_h100_rot"] / result["rollout_mse_h100"]
        )
        result["persistence_rot_gap"] = (
            result["persistence_mse_rot"] / result["persistence_mse"]
        )

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

        # v2 scorers
        cs = A.cusum_scores(scores)
        result["anomaly_auroc_cusum"] = A.auroc(cs, labels)
        result["detection_delay_cusum"] = A.detection_delay(cs, labels)
        result.update({f"{k}_cusum": v for k, v in A.event_f1(cs, labels).items()})

        inv = A.invariant_scores(araw["x"])  # model-free level-set monitor
        result["anomaly_auroc_inv"] = A.auroc(inv, labels)
        result.update({f"{k}_inv": v for k, v in A.event_f1(inv, labels).items()})

        if args.anomaly_rollout:
            rs = A.rollout_scores(model, x[:200])
            lab = labels[:200]
            mask = ~np.isnan(rs)
            result["anomaly_auroc_roll"] = A.auroc(rs[mask], lab[mask])
            result.update({f"{k}_roll": v for k, v in A.event_f1(rs, lab).items()})

    (args.run / "eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
