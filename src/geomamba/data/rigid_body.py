"""D1 — torque-free rigid body (Euler equations), the natural rotor testbed.

I ω' = (I ω) × ω  in the body frame, diagonal inertia I1 < I2 < I3.
Initial conditions are biased to include intermediate-axis (Dzhanibekov)
tumbling. Observations are ω. Per-trajectory inertia is returned in
'<split>_params' so evaluation can compute the conserved quantities:
kinetic energy E = ω·Iω/2 and body-frame angular momentum magnitude |Iω|.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def _rhs(inertia):
    i1, i2, i3 = inertia

    def f(_t, w):
        return [
            (i2 - i3) / i1 * w[1] * w[2],
            (i3 - i1) / i2 * w[2] * w[0],
            (i1 - i2) / i3 * w[0] * w[1],
        ]

    return f


def generate(
    n_traj: int,
    length: int,
    dt: float = 0.05,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs = np.empty((n_traj, length, 3), dtype=np.float32)
    params = np.empty((n_traj, 3), dtype=np.float32)
    t_eval = np.arange(length) * dt
    for i in range(n_traj):
        inertia = np.sort(rng.uniform(0.5, 3.0, size=3))
        w0 = rng.normal(0, 1, size=3)
        # bias half of trajectories toward intermediate-axis tumbling
        if i % 2 == 0:
            w0 = np.array([0.05, 1.0, 0.05]) * rng.uniform(0.5, 1.5) \
                + rng.normal(0, 0.05, 3)
        sol = solve_ivp(
            _rhs(inertia),
            (0, t_eval[-1] if length > 1 else dt),
            w0,
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-12,
        )
        obs[i] = sol.y.T.astype(np.float32)
        params[i] = inertia
    return {"x": obs, "params": params}


def energy(omega: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    """E = ω·Iω/2. omega [..., T, 3], inertia [..., 3] -> [..., T]"""
    return 0.5 * np.einsum("...ti,...i,...ti->...t", omega, inertia, omega)


def momentum_norm(omega: np.ndarray, inertia: np.ndarray) -> np.ndarray:
    """|L| = |Iω| (conserved in the body frame for torque-free motion)."""
    L = omega * inertia[..., None, :]
    return np.linalg.norm(L, axis=-1)
