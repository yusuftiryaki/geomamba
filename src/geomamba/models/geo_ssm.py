"""Geometric selective SSM: multivector hidden state in Cl(3,0).

Per multivector channel c (state h ∈ R^8):

    Δ_t   = softplus(dt_proj(x_t))                    input-dependent step
    decay = exp(-Δ_t · softplus(λ_c))                 explicit dissipation
    variant "rotor" (c):  A_t = decay · S(R_t),  R_t = exp(-Δ_t b_t / 2)
        with input-dependent bivector generator b_t and sandwich matrix S —
        an orthogonal (norm-preserving) transition times a scalar contraction.
    variant "mv" (b):     A_t = decay · (I + Δ_t G_c)
        with an unconstrained learned per-channel generator G_c (Euler step).
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


class GeoS6(nn.Module):
    def __init__(
        self,
        d_inner: int,
        transition: str = "rotor",  # rotor | mv
        d_state: int = 8,  # multivector dim; 8 = Cl(3,0)
        dt_rank: int | None = None,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        scan_mode: str = "chunked",
    ):
        super().__init__()
        if d_state != 8:
            raise NotImplementedError("GeoS6 currently assumes Cl(3,0), d_state=8")
        if d_inner % d_state != 0:
            raise ValueError("d_inner must be divisible by the multivector dim")
        if transition not in ("rotor", "mv"):
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
        else:
            self.G = nn.Parameter(torch.randn(C, d_state, d_state) * 1e-2)

        # near-conservative init: softplus(lambda_raw) ~ 0.05
        self.lambda_raw = nn.Parameter(torch.full((C,), -3.0))
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

        if self.transition == "rotor":
            biv = self.gen_proj(x).reshape(b, l, C, -1)
            R = rotor.bivector_exp(self.al, -0.5 * dt.unsqueeze(-1) * biv)
            A = decay[..., None, None] * rotor.sandwich_matrix(self.al, R)
        else:
            eye = torch.eye(d, dtype=x.dtype, device=x.device)
            A = decay[..., None, None] * (eye + dt[..., None, None] * self.G)

        inputs = (dt * gate).unsqueeze(-1) * u
        h = pscan.scan_block(A, inputs, self.scan_mode)  # [B, L, C, d]

        grade_w = grades.reshape(b, l, C, self.n_grades)
        readout = grade_w @ self.al.grade_masks.to(x.dtype)  # [B, L, C, d]
        y = (h * readout).reshape(b, l, self.d_inner)
        return y + self.D * x
