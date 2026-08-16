"""Evaluation metrics: teacher-forced MSE, rollout MSE at horizons, and the
persistence baseline (predict zero delta) as the floor every model must beat."""
from __future__ import annotations

import torch

from geomamba.models.model import GeoMamba


def _model_device(model: GeoMamba) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def teacher_forced_mse(model: GeoMamba, loader, max_batches: int | None = None) -> float:
    device = _model_device(model)
    total, count = 0.0, 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch["input"])["forecast"]
        total += torch.nn.functional.mse_loss(
            pred, batch["target"], reduction="sum"
        ).item()
        count += batch["target"].numel()
    return total / count


@torch.no_grad()
def persistence_mse(loader) -> float:
    """MSE of predicting delta = 0, i.e. x_{t+1} = x_t."""
    total, count = 0.0, 0
    for batch in loader:
        total += (batch["target"] ** 2).sum().item()
        count += batch["target"].numel()
    return total / count


@torch.no_grad()
def rollout_mse(
    model: GeoMamba,
    loader,
    context_len: int,
    horizons: tuple[int, ...] = (1, 10, 100),
    max_batches: int = 4,
) -> dict[str, float]:
    h_max = max(horizons)
    sums = {h: 0.0 for h in horizons}
    counts = {h: 0 for h in horizons}
    device = _model_device(model)
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        # reconstruct absolute trajectory from input (input is x_t sequence)
        traj = batch["input"].to(device)
        if traj.shape[1] < context_len + h_max:
            raise ValueError("trajectory too short for requested rollout")
        context = traj[:, :context_len]
        truth = traj[:, context_len : context_len + h_max]
        preds = model.rollout(context, h_max)
        for h in horizons:
            sums[h] += torch.nn.functional.mse_loss(
                preds[:, :h], truth[:, :h], reduction="sum"
            ).item()
            counts[h] += truth[:, :h].numel()
    return {f"rollout_mse_h{h}": sums[h] / counts[h] for h in horizons}
