"""Anomalous trajectory generation: a physical parameter changes at a random
time τ while the state stays continuous. Labels are 1 from τ onward.

Systems: rigid_body (inertia jump), lorentz_particle (field flip/step).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from geomamba.data import lorentz_particle, rigid_body


def rigid_body_anomalous(
    n_traj: int,
    length: int,
    dt: float = 0.05,
    seed: int = 0,
    anomaly_frac: float = 0.2,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = np.empty((n_traj, length, 3), dtype=np.float32)
    labels = np.zeros((n_traj, length), dtype=np.float32)
    for i in range(n_traj):
        inertia = np.sort(rng.uniform(0.5, 3.0, size=3))
        w0 = rng.normal(0, 1, size=3)
        is_anom = rng.random() < anomaly_frac
        tau = rng.integers(length // 4, 3 * length // 4) if is_anom else length
        segs, t_next, w = [], int(tau), w0
        for seg_len, seg_inertia in [
            (t_next, inertia),
            (length - t_next, np.sort(inertia * rng.uniform(1.3, 2.0, size=3))),
        ]:
            if seg_len == 0:
                continue
            t_eval = np.arange(seg_len) * dt
            sol = solve_ivp(
                rigid_body._rhs(seg_inertia),
                (0, t_eval[-1] if seg_len > 1 else dt),
                w,
                t_eval=t_eval,
                rtol=1e-10,
                atol=1e-12,
            )
            segs.append(sol.y.T)
            w = sol.y.T[-1]
        obs[i] = np.concatenate(segs).astype(np.float32)
        if is_anom:
            labels[i, tau:] = 1.0
    return {"x": obs, "labels": labels}


def lorentz_particle_anomalous(
    n_traj: int,
    length: int,
    dt: float = 0.05,
    seed: int = 0,
    anomaly_frac: float = 0.2,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = np.empty((n_traj, length, 3), dtype=np.float32)
    labels = np.zeros((n_traj, length), dtype=np.float32)
    for i in range(n_traj):
        B = rng.normal(0, 1, 3)
        B *= rng.uniform(0.5, 2.0) / np.linalg.norm(B)
        v = rng.normal(0, 1, 3)
        is_anom = rng.random() < anomaly_frac
        tau = rng.integers(length // 4, 3 * length // 4) if is_anom else length
        for t in range(length):
            if t == tau:
                B = -B * rng.uniform(1.2, 1.8)  # flip + strengthen
            obs[i, t] = v
            v = lorentz_particle._boris_step(v, B, dt)
        if is_anom:
            labels[i, tau:] = 1.0
    return {"x": obs, "labels": labels}


ANOMALOUS = {
    "rigid_body": rigid_body_anomalous,
    "lorentz_particle": lorentz_particle_anomalous,
}
