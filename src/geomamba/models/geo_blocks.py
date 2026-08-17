"""The (d) rung: a Mamba block whose every component respects the algebra.

Baseline block            ->  equivariant counterpart
  Linear in/out proj      ->  CliffordLinear (per-grade channel mixing)
  depthwise conv1d        ->  BladeSharedConv1d (one kernel for all blades)
  u * silu(z)             ->  GeoGate (scalar per grade, from invariants)
  (nothing)               ->  GeometricBilinear -- the geometric product itself,
                              the only place grades interact
  S6 / GeoS6              ->  EquivariantGeoS6

Nothing here sees a blade coefficient as a free coordinate, so a rotation of
the input produces exactly the same rotation of the output. That property is
asserted numerically in tests/test_equivariance.py -- it is the claim the
earlier `geo_proj` "full geometric" arm silently failed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from geomamba.algebra import Algebra
from geomamba.models.geo_layers import (
    BladeSharedConv1d,
    CliffordLinear,
    GeoGate,
    GeometricBilinear,
)
from geomamba.models.geo_ssm_eq import EquivariantGeoS6


class GeoBlock(nn.Module):
    def __init__(
        self,
        al: Algebra,
        c_model: int,
        expand: int = 2,
        d_conv: int = 4,
        scan_mode: str = "chunked",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        lambda_init: float = -3.0,
        bilinear: bool = True,
    ):
        super().__init__()
        c_inner = expand * c_model
        self.in_proj = CliffordLinear(al, c_model, 2 * c_inner)
        self.conv = BladeSharedConv1d(c_inner, d_conv)
        self.ssm = EquivariantGeoS6(
            al, c_inner, dt_min=dt_min, dt_max=dt_max,
            lambda_init=lambda_init, scan_mode=scan_mode,
        )
        self.gate = GeoGate(al, c_inner)
        self.bilinear = GeometricBilinear(al, c_inner, c_inner) if bilinear else None
        self.out_proj = CliffordLinear(al, c_inner, c_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C, dim]"""
        u, z = self.in_proj(x).chunk(2, dim=-2)
        u = self.conv(u)
        u = self.ssm(u)
        u = self.gate(u, z)
        if self.bilinear is not None:
            u = u + self.bilinear(u, z)
        return self.out_proj(u)
