"""Equivariant selective SSM — the (d) rung, built so the constraint survives.

`GeoS6` produces its step size, input gate, grade readout and bivector
generator with plain Linears over *flattened* blade coefficients. Those layers
see blades as free coordinates, so the rotor sandwich inside is wrapped in maps
that do not respect the algebra: the recurrence is geometric, the selectivity
around it is not, and rotating the input does not rotate the output.

Here every input-dependent quantity is either

  * an INVARIANT scalar (step size, dissipation, input gate, grade readout),
    computed from grade norms, which the sandwich leaves unchanged; or
  * an EQUIVARIANT multivector (the bivector generator), produced by a
    CliffordLinear and grade-projected, so it rotates with the input.

Then A_t = decay_t * S(R_t) commutes with a global rotation and the whole
recurrence is equivariant: h(Q x Q~) = Q h(x) Q~.

One deliberate design change over `GeoS6`: the dissipation is invariant-
CONDITIONED (`lambda + f(invariants)`) rather than a fixed per-channel scalar.
A rotor sandwich alone conserves the state norm, which is the wrong prior for a
free rigid body -- there |omega| is not conserved (energy and |I omega| are).
Letting the scale depend on invariants keeps equivariance while dropping the
false conservation law.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomamba.algebra import Algebra, rotor
from geomamba.models import pscan
from geomamba.models.geo_layers import CliffordLinear, grade_invariants
from geomamba.models.geo_ssm import _biv_to_axis


class EquivariantGeoS6(nn.Module):
    """Multivector state, rotor transition, everything else invariant."""

    def __init__(
        self,
        al: Algebra,
        channels: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        lambda_init: float = -3.0,
        scan_mode: str = "chunked",
    ):
        super().__init__()
        self.al = al
        self.c = channels
        self.n_grades = al.n + 1
        self.scan_mode = scan_mode
        n_inv = channels * self.n_grades

        # equivariant: a multivector-valued map whose grade-2 part generates the rotor
        self.gen = CliffordLinear(al, channels, channels)
        nn.init.uniform_(self.gen.weight, -1e-2, 1e-2)  # rotation per step << pi

        # invariant: scalars that may depend on the input without breaking anything
        self.to_dt = nn.Linear(n_inv, channels)
        self.to_gate = nn.Linear(n_inv, channels, bias=False)
        self.to_read = nn.Linear(n_inv, n_inv, bias=False)
        self.to_lambda = nn.Linear(n_inv, channels, bias=False)
        nn.init.zeros_(self.to_lambda.weight)  # start at the plain per-channel decay

        dt = torch.exp(
            torch.rand(channels) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        with torch.no_grad():
            self.to_dt.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        nn.init.zeros_(self.to_dt.weight)

        self.lambda_raw = nn.Parameter(torch.full((channels,), float(lambda_init)))
        # skip connection: one scalar per (channel, grade), never per blade
        self.D = nn.Parameter(torch.ones(channels, self.n_grades))
        self.sp_indices = [0, al.dim - 1]  # scalar + pseudoscalar: sandwich-invariant

    def _gains(self, s: torch.Tensor) -> torch.Tensor:
        masks = self.al.grade_masks.to(dtype=s.dtype, device=s.device)
        return s.reshape(*s.shape[:-1], self.c, self.n_grades) @ masks

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: [B, L, C, dim] -> y: [B, L, C, dim]"""
        b, l, c, d = u.shape
        inv = grade_invariants(self.al, u)

        dt = F.softplus(self.to_dt(inv))  # [B, L, C]
        lam = F.softplus(self.lambda_raw + self.to_lambda(inv))
        decay = torch.exp(-dt * lam)

        g = self.gen(u)[..., self.al.bivector_indices]  # equivariant generator
        Q = rotor.rotation_matrix(g, dt)  # [B, L, C, 3, 3]
        A = decay[..., None, None] * Q

        inputs = (dt * self.to_gate(inv)).unsqueeze(-1) * u

        # grades 1 and 2 rotate by the same 3x3 block; grades 0 and 3 only decay
        u_v = inputs[..., self.al.vector_indices]
        u_b = _biv_to_axis(inputs[..., self.al.bivector_indices])
        h_vb = pscan.scan_block(
            torch.cat([A, A], dim=2), torch.cat([u_v, u_b], dim=2), self.scan_mode
        )
        h_sp = pscan.scan_diag(
            decay.repeat_interleave(2, dim=-1),
            inputs[..., self.sp_indices].reshape(b, l, c * 2),
            self.scan_mode,
        ).reshape(b, l, c, 2)

        h = inputs.new_empty(b, l, c, d)
        h[..., self.al.vector_indices] = h_vb[:, :, :c]
        h[..., self.al.bivector_indices] = _biv_to_axis(h_vb[:, :, c:])
        h[..., self.sp_indices] = h_sp

        y = h * self._gains(self.to_read(inv))
        skip = self.D.to(u.dtype) @ self.al.grade_masks.to(dtype=u.dtype, device=u.device)
        return y + skip * u
