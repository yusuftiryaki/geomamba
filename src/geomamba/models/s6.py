"""Vanilla selective SSM (S6) — ablation variant (a), pure PyTorch.

Follows Mamba (Gu & Dao 2023): per-channel diagonal real A, input-dependent
Δ/B/C, ZOH discretization, computed with an associative scan.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomamba.models import pscan


class S6(nn.Module):
    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        dt_rank: int | None = None,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        scan_mode: str = "chunked",
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_inner / 16)
        self.scan_mode = scan_mode

        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # dt bias init so softplus(bias) ~ U(dt_min, dt_max) (Mamba init)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)

        # A = -[1..N] per channel (S4D-real init)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_inner] -> y: [B, L, d_inner]"""
        dbc = self.x_proj(x)
        dt, B, C = torch.split(
            dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # [B, L, D]

        A = -torch.exp(self.A_log)  # [D, N]
        A_bar = torch.exp(dt.unsqueeze(-1) * A)  # [B, L, D, N]
        Bx = (dt * x).unsqueeze(-1) * B.unsqueeze(2)  # [B, L, D, N]

        b, l, d, n = A_bar.shape
        h = pscan.scan_diag(
            A_bar.reshape(b, l, d * n), Bx.reshape(b, l, d * n), self.scan_mode
        ).reshape(b, l, d, n)

        y = (h * C.unsqueeze(2)).sum(-1) + self.D * x
        return y
