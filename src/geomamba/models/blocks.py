"""Mamba blocks: shared conv+gate skeleton, SSM core varies by variant."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomamba.algebra import Algebra
from geomamba.models.geo_layers import CliffordLinear
from geomamba.models.geo_ssm import GeoS6
from geomamba.models.s6 import S6


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        expand: int = 2,
        d_conv: int = 4,
        d_state: int = 16,
        scan_mode: str = "chunked",
        transition: str | None = None,  # None = vanilla S6; "mv"/"rotor" = GeoS6
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        lambda_init: float = -3.0,
        geo_proj: bool = False,
    ):
        super().__init__()
        d_inner = expand * d_model
        self.geo_proj = geo_proj
        if geo_proj:
            al = Algebra.get(3, 0)
            if d_model % al.dim or d_inner % al.dim:
                raise ValueError("geo_proj needs d_model/d_inner divisible by 8")
            self.mv_dim = al.dim
            self.in_proj = CliffordLinear(al, d_model // al.dim, 2 * d_inner // al.dim)
            self.out_proj = CliffordLinear(al, d_inner // al.dim, d_model // al.dim)
        else:
            self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1
        )
        if transition is None:
            self.ssm = S6(d_inner, d_state=d_state, scan_mode=scan_mode,
                          dt_min=dt_min, dt_max=dt_max)
        else:
            self.ssm = GeoS6(d_inner, transition=transition, scan_mode=scan_mode,
                             dt_min=dt_min, dt_max=dt_max, lambda_init=lambda_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model]"""
        L = x.shape[1]
        if self.geo_proj:
            b, l, d = x.shape
            proj = self.in_proj(x.reshape(b, l, d // self.mv_dim, self.mv_dim))
            u, z = proj.reshape(b, l, -1).chunk(2, dim=-1)
        else:
            u, z = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2)  # causal
        u = F.silu(u)
        u = self.ssm(u)
        u = u * F.silu(z)
        if self.geo_proj:
            b, l, d = u.shape
            out = self.out_proj(u.reshape(b, l, d // self.mv_dim, self.mv_dim))
            return out.reshape(b, l, -1)
        return self.out_proj(u)
