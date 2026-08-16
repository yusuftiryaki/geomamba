"""D2 — charged particle in a static uniform magnetic field (Boris pusher).

v' = v × B (q/m = 1, E = 0), so the speed |v| is exactly conserved and the
motion is gyration about B plus parallel drift — a pure rotor dynamic in
Cl(3,0), where B is naturally a bivector. Observations are v.
params = B vector (per trajectory).
"""
from __future__ import annotations

import numpy as np


def _boris_step(v: np.ndarray, B: np.ndarray, dt: float) -> np.ndarray:
    # standard Boris rotation (E = 0): exactly norm-preserving
    t = 0.5 * dt * B
    s = 2.0 * t / (1.0 + t @ t)
    v_prime = v + np.cross(v, t)
    return v + np.cross(v_prime, s)


def generate(
    n_traj: int,
    length: int,
    dt: float = 0.05,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = np.empty((n_traj, length, 3), dtype=np.float32)
    params = np.empty((n_traj, 3), dtype=np.float32)
    for i in range(n_traj):
        B = rng.normal(0, 1, 3)
        B *= rng.uniform(0.5, 2.0) / np.linalg.norm(B)
        v = rng.normal(0, 1, 3)
        for t in range(length):
            obs[i, t] = v
            v = _boris_step(v, B, dt)
        params[i] = B
    return {"x": obs, "params": params}


def speed(v: np.ndarray) -> np.ndarray:
    """|v| — conserved for E = 0. v [..., T, 3] -> [..., T]"""
    return np.linalg.norm(v, axis=-1)
