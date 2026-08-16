"""D4 — Lorenz-63. Negative control: chaotic, no rotational conservation
structure, so the rotor bias is expected to help least here."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
TRANSIENT_STEPS = 200


def _rhs(_t, s):
    x, y, z = s
    return [SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z]


def generate(
    n_traj: int,
    length: int,
    dt: float = 0.02,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = np.empty((n_traj, length, 3), dtype=np.float32)
    total = length + TRANSIENT_STEPS
    t_eval = np.arange(total) * dt
    for i in range(n_traj):
        s0 = rng.normal([0, 0, 25], [10, 10, 8])
        sol = solve_ivp(
            _rhs,
            (0, t_eval[-1]),
            s0,
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-9,
        )
        obs[i] = sol.y.T[TRANSIENT_STEPS:].astype(np.float32)
    return {"x": obs}
