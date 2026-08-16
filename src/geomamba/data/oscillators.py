"""D3 — damped coupled oscillators (linear chain, fixed ends).

x'' = -K x - C x'  with tridiagonal stiffness K and diagonal damping C.
Exact discrete-time simulation via the matrix exponential of the state matrix,
so trajectories carry no integrator error. Observations are the positions.
Per-trajectory random stiffness/damping/initial conditions.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def generate(
    n_traj: int,
    length: int,
    n_masses: int = 4,
    dt: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = np.empty((n_traj, length, n_masses), dtype=np.float32)
    for i in range(n_traj):
        k = rng.uniform(0.5, 2.0, size=n_masses + 1)  # springs incl. wall springs
        c = rng.uniform(0.01, 0.1, size=n_masses)
        K = np.zeros((n_masses, n_masses))
        for j in range(n_masses):
            K[j, j] = k[j] + k[j + 1]
            if j > 0:
                K[j, j - 1] = -k[j]
            if j < n_masses - 1:
                K[j, j + 1] = -k[j + 1]
        A = np.block(
            [
                [np.zeros((n_masses, n_masses)), np.eye(n_masses)],
                [-K, -np.diag(c)],
            ]
        )
        step = expm(A * dt)
        state = np.concatenate(
            [rng.uniform(-1, 1, n_masses), rng.uniform(-0.5, 0.5, n_masses)]
        )
        for t in range(length):
            obs[i, t] = state[:n_masses]
            state = step @ state
    return {"x": obs}
