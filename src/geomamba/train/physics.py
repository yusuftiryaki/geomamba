"""Physical-consistency metrics: relative drift of conserved quantities along
model rollouts, computed from predicted observables (denormalized).

Headline metrics of the hypothesis: a physically consistent model should show
low drift of invariants (energy, |L|, speed) over long autoregressive rollouts.
"""
from __future__ import annotations

import numpy as np

from geomamba.data import lorentz_particle, rigid_body


def relative_drift(series: np.ndarray) -> np.ndarray:
    """|q_t - q_0| / (|q_0| + eps) per step; series [..., T] -> [..., T]"""
    q0 = series[..., :1]
    return np.abs(series - q0) / (np.abs(q0) + 1e-8)


def rigid_body_drifts(omega: np.ndarray, inertia: np.ndarray) -> dict[str, float]:
    """omega: predicted rollout [N, T, 3] (denormalized), inertia [N, 3].
    Returns mean final-step relative drift of E and |L|."""
    e = rigid_body.energy(omega, inertia)
    l = rigid_body.momentum_norm(omega, inertia)
    return {
        "energy_drift": float(relative_drift(e)[..., -1].mean()),
        "momentum_drift": float(relative_drift(l)[..., -1].mean()),
    }


def particle_drifts(v: np.ndarray) -> dict[str, float]:
    """v: predicted rollout [N, T, 3] (denormalized)."""
    s = lorentz_particle.speed(v)
    return {"speed_drift": float(relative_drift(s)[..., -1].mean())}
