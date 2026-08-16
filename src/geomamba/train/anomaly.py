"""Unsupervised anomaly scoring: per-step forecast residual energy.

The model never sees anomalies in training; a physically consistent model
should produce sharply higher residuals after a parameter change.
"""
from __future__ import annotations

import numpy as np
import torch

from geomamba.models.model import GeoMamba


@torch.no_grad()
def residual_scores(model: GeoMamba, x: torch.Tensor, batch: int = 16) -> np.ndarray:
    """x: [N, L, F] normalized trajectories -> scores [N, L-1]."""
    device = next(model.parameters()).device
    scores = []
    for s in range(0, x.shape[0], batch):
        xb = x[s : s + batch].to(device)
        pred = model(xb[:, :-1])["forecast"]
        resid = pred - (xb[:, 1:] - xb[:, :-1])
        scores.append((resid**2).mean(-1).cpu().numpy())
    return np.concatenate(scores)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC over flattened per-step scores/labels."""
    s, y = scores.ravel(), labels.ravel()
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].mean() - (len(pos) + 1) / 2) / len(neg))


def detection_delay(
    scores: np.ndarray, labels: np.ndarray, quantile: float = 0.99
) -> float:
    """Mean steps from anomaly onset to first threshold crossing (threshold =
    per-dataset quantile of scores on fully-normal trajectories)."""
    normal_mask = labels.max(axis=1) < 0.5
    if normal_mask.all() or not normal_mask.any():
        return float("nan")
    thr = np.quantile(scores[normal_mask], quantile)
    delays = []
    for s_i, l_i in zip(scores[~normal_mask], labels[~normal_mask]):
        tau = int(np.argmax(l_i > 0.5))
        # labels are aligned to x[1:]: score index t scores step t+1
        after = s_i[max(tau - 1, 0) :] > thr
        delays.append(int(np.argmax(after)) if after.any() else len(after))
    return float(np.mean(delays))


def two_sided_scores(scores: np.ndarray, calib_frac: float = 0.25) -> np.ndarray:
    """|log s_t - median(log s)| over each trajectory's early (pre-anomaly)
    window. Catches anomalies that make prediction EASIER (lower residual),
    e.g. an inertia increase slowing the dynamics."""
    log_s = np.log(scores + 1e-12)
    n_calib = max(1, int(scores.shape[1] * calib_frac))
    ref = np.median(log_s[:, :n_calib], axis=1, keepdims=True)
    return np.abs(log_s - ref)
