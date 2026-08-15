"""Mamba blocks: shared conv+gate skeleton, SSM core varies by variant."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    ):
        super().__init__()
        d_inner = expand * d_model
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1
        )
        if transition is None:
            self.ssm = S6(d_inner, d_state=d_state, scan_mode=scan_mode)
        else:
            self.ssm = GeoS6(d_inner, transition=transition, scan_mode=scan_mode)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model]"""
        L = x.shape[1]
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2)  # causal
        u = F.silu(u)
        u = self.ssm(u)
        u = u * F.silu(z)
        return self.out_proj(u)
