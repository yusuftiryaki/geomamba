"""Geometric selective SSM: multivector hidden state in Cl(3,0).

Per multivector channel c (state h ∈ R^8):

    Δ_t   = softplus(dt_proj(x_t))                    input-dependent step
    decay = exp(-Δ_t · softplus(λ_c))                 explicit dissipation
    variant "rotor" (c):  A_t = decay · S(R_t),  R_t = exp(-Δ_t b_t / 2)
        with input-dependent bivector generator b_t and sandwich matrix S —
        an orthogonal (norm-preserving) transition times a scalar contraction.
    variant "mv" (b):     A_t = decay · (I + Δ_t M_t) on the same grade-split
        state, with M_t an input-dependent UNCONSTRAINED 3x3 generator — the
        exact contrast to "rotor" is gl(3) vs so(3), same machinery otherwise.
    variant "mv8":        legacy full 8x8 constant generator (slow; optional).
    h_t = A_t h_{t-1} + Δ_t B_t ⊙ u_t                 gated input injection
    y_t = (grade-wise readout C_t) ⊙ h_t + D ⊙ x_t

The selectivity of Mamba (input-dependent Δ) becomes the step size of a
variable-step Lie-group integrator on Spin(3); the conservative and
dissipative parts of the transition are architecturally separated.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomamba.algebra import Algebra, rotor
from geomamba.models import pscan


def _biv_to_axis(c: torch.Tensor) -> torch.Tensor:
    """Bivector blade coeffs [e12, e13, e23] <-> dual axis vector (involution)."""
    return torch.stack([c[..., 2], -c[..., 1], c[..., 0]], dim=-1)


class GeoS6(nn.Module):
    def __init__(
        self,
        d_inner: int,
        transition: str = "rotor",  # rotor | mv
        d_state: int = 8,  # multivector dim; 8 = Cl(3,0)
        dt_rank: int | None = None,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        lambda_init: float = -3.0,
        scan_mode: str = "chunked",
    ):
        super().__init__()
        if d_state != 8:
            raise NotImplementedError("GeoS6 currently assumes Cl(3,0), d_state=8")
        if d_inner % d_state != 0:
            raise ValueError("d_inner must be divisible by the multivector dim")
        if transition not in ("rotor", "mv", "mv8"):
            raise ValueError(f"unknown transition: {transition}")
        self.al = Algebra.get(3, 0)
        self.d_inner = d_inner
        self.mv_dim = d_state
        self.n_ch = d_inner // d_state
        self.n_grades = self.al.n + 1
        self.transition = transition
        self.scan_mode = scan_mode
        self.dt_rank = dt_rank or math.ceil(d_inner / 16)

        C = self.n_ch
        # fused projection: [dt_low | input gate B (C) | grade readout (C*4)]
        self.x_proj = nn.Linear(d_inner, self.dt_rank + C + C * self.n_grades, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, C, bias=True)
        dt = torch.exp(
            torch.rand(C) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        if transition == "rotor":
            n_biv = len(self.al.bivector_indices)
            self.gen_proj = nn.Linear(d_inner, C * n_biv, bias=False)
            # small init: rotation per step << pi at start
            nn.init.normal_(self.gen_proj.weight, std=1e-2)
        elif transition == "mv":
            self.gen_proj = nn.Linear(d_inner, C * 9, bias=False)
            nn.init.normal_(self.gen_proj.weight, std=1e-2)
        else:
            self.G = nn.Parameter(torch.randn(C, d_state, d_state) * 1e-2)

        # scalar/pseudoscalar blade positions (sandwich-invariant grades)
        self.sp_indices = [0, self.al.dim - 1]
        # dissipation init: softplus(lambda_init); -3 ~ 0.05, -12 ~ pure-conservative
        self.lambda_raw = nn.Parameter(torch.full((C,), float(lambda_init)))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_inner] -> y: [B, L, d_inner]"""
        b, l, _ = x.shape
        C, d = self.n_ch, self.mv_dim
        u = x.reshape(b, l, C, d)

        proj = self.x_proj(x)
        dt_low, gate, grades = torch.split(
            proj, [self.dt_rank, C, C * self.n_grades], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_low))  # [B, L, C]
        decay = torch.exp(-dt * F.softplus(self.lambda_raw))  # [B, L, C]

        inputs = (dt * gate).unsqueeze(-1) * u
        if self.transition in ("rotor", "mv"):
            # Fast path exploiting the grade structure of the Cl(3,0) sandwich:
            # grades 0/3 are invariant (transition = pure decay), grades 1/2
            # both transform by the same 3x3 matrix -> one 3x3 block scan on
            # stacked vector+bivector channels instead of an 8x8 scan.
            # rotor: orthogonal Q = exp of so(3); mv: unconstrained Euler step
            # of gl(3) — the pair isolates exactly the rotor constraint.
            if self.transition == "rotor":
                g = self.gen_proj(x).reshape(b, l, C, -1)
                Q = rotor.rotation_matrix(g, dt)  # [B, L, C, 3, 3]
            else:
                M = self.gen_proj(x).reshape(b, l, C, 3, 3)
                eye3 = torch.eye(3, dtype=x.dtype, device=x.device)
                Q = eye3 + dt[..., None, None] * M
            A_rot = decay[..., None, None] * Q
            u_v = inputs[..., self.al.vector_indices]
            u_b = _biv_to_axis(inputs[..., self.al.bivector_indices])
            h_vb = pscan.scan_block(
                torch.cat([A_rot, A_rot], dim=2),
                torch.cat([u_v, u_b], dim=2),
                self.scan_mode,
            )
            u_sp = inputs[..., self.sp_indices]  # scalar + pseudoscalar
            h_sp = pscan.scan_diag(
                decay.repeat_interleave(2, dim=-1),
                u_sp.reshape(b, l, C * 2),
                self.scan_mode,
            ).reshape(b, l, C, 2)
            h = inputs.new_empty(b, l, C, d)
            h[..., self.al.vector_indices] = h_vb[:, :, :C]
            h[..., self.al.bivector_indices] = _biv_to_axis(h_vb[:, :, C:])
            h[..., self.sp_indices] = h_sp
        else:
            eye = torch.eye(d, dtype=x.dtype, device=x.device)
            A = decay[..., None, None] * (eye + dt[..., None, None] * self.G)
            h = pscan.scan_block(A, inputs, self.scan_mode)  # [B, L, C, d]

        grade_w = grades.reshape(b, l, C, self.n_grades)
        readout = grade_w @ self.al.grade_masks.to(dtype=x.dtype, device=x.device)
        y = (h * readout).reshape(b, l, self.d_inner)
        return y + self.D * x
