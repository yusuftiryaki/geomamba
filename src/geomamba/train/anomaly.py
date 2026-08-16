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


def cusum_scores(scores: np.ndarray, calib_frac: float = 0.25,
                 drift: float = 0.5) -> np.ndarray:
    """Two-sided CUSUM over robust-z of log-residuals: small persistent shifts
    accumulate instead of vanishing into per-step noise."""
    log_s = np.log(scores + 1e-12)
    n_cal = max(2, int(scores.shape[1] * calib_frac))
    med = np.median(log_s[:, :n_cal], axis=1, keepdims=True)
    mad = np.median(np.abs(log_s[:, :n_cal] - med), axis=1, keepdims=True)
    z = (log_s - med) / (1.4826 * mad + 1e-6)
    out = np.zeros_like(z)
    cp = np.zeros(z.shape[0])
    cn = np.zeros(z.shape[0])
    for t in range(z.shape[1]):
        cp = np.maximum(0.0, cp + z[:, t] - drift)
        cn = np.maximum(0.0, cn - z[:, t] - drift)
        out[:, t] = np.maximum(cp, cn)
    return out


@torch.no_grad()
def rollout_scores(model: GeoMamba, x: torch.Tensor, context: int = 64,
                   k: int = 32, stride: int = 16, batch: int = 64) -> np.ndarray:
    """Freeze-the-regime detector: from each anchor t, roll the model out k
    steps ("what if the old regime continued") and score the growing deviation
    from what actually happened. A one-step residual lets a selective model
    re-adapt every step and hide regime changes; this does not.
    Returns per-step scores [N, L-1] (nan before the first anchor)."""
    N, L, _ = x.shape
    scores = np.full((N, L - 1), np.nan, dtype=np.float64)
    device = next(model.parameters()).device
    for s in range(0, N, batch):
        xb = x[s : s + batch].to(device)
        for t in range(context, L - 1, stride):
            h = min(k, L - t)
            preds = model.rollout(xb[:, :t], h)
            err = ((preds - xb[:, t : t + h]) ** 2).mean(-1)  # [B, h]
            cum = (err.cumsum(-1) / torch.arange(1, h + 1, device=device)).cpu().numpy()
            block = scores[s : s + batch, t - 1 : t - 1 + h]
            np.fmax(block, cum, out=block)
    return scores


def invariant_scores(x_raw: np.ndarray, calib_frac: float = 0.25,
                     n_forms: int = 2) -> np.ndarray:
    """Model-free level-set monitor ("deviation from the normal manifold").

    Healthy trajectories of a conservative system live on level sets of
    conserved quadratic forms (rigid body: energy ellipsoid x momentum sphere
    -> the polhode curve). Estimate the most-conserved quadratic forms from
    each trajectory's calibration window (smallest-variance eigenvectors of
    the squared-observation covariance), then score departure of those forms
    from their calibration values. x_raw: [N, T, F] DENORMALIZED.
    """
    N, T, F = x_raw.shape
    X = x_raw.astype(np.float64) ** 2  # diagonal quadratic monomials
    n_cal = max(F + 1, int(T * calib_frac))
    scores = np.zeros((N, T))
    for i in range(N):
        cal = X[i, :n_cal]
        cov = np.cov(cal.T)
        _, vecs = np.linalg.eigh(cov)  # ascending eigenvalues
        forms = vecs[:, :n_forms]  # most-conserved directions
        vals = X[i] @ forms  # [T, n_forms]
        mu = vals[:n_cal].mean(0)
        sd = vals[:n_cal].std(0) + 1e-9
        scores[i] = (np.abs(vals - mu) / sd).sum(1)
    return scores[:, 1:]  # align to x[1:] like the residual scores


def event_f1(scores: np.ndarray, labels: np.ndarray, tolerance: int = 20,
             quantile: float = 0.99) -> dict[str, float]:
    """Event-level detection: one predicted onset per trajectory (first
    threshold crossing). Suffix per-step AUROC punishes detectors on models
    that re-adapt after the transient; event metrics do not."""
    valid = ~np.isnan(scores)
    normal_mask = labels.max(axis=1) < 0.5
    if normal_mask.all() or not normal_mask.any():
        return {"event_f1": float("nan")}
    thr = np.quantile(scores[normal_mask][valid[normal_mask]], quantile)
    tp = fp = fn = 0
    for s_i, l_i, v_i in zip(scores, labels, valid):
        crossing = np.where((s_i > thr) & v_i)[0]
        onset = int(crossing[0]) if len(crossing) else None
        if l_i.max() > 0.5:
            tau = int(np.argmax(l_i > 0.5))
            if onset is not None and tau - 1 - tolerance <= onset:
                tp += 1
            else:
                fn += 1
        elif onset is not None:
            fp += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"event_f1": f1, "event_precision": prec, "event_recall": rec}
