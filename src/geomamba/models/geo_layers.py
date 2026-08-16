"""Grade-respecting layers: physics-informed tokenization and channel mixing.

The default pipeline embeds inputs with a plain Linear and mixes channels with
plain Linears, so the multivector layout is imposed by reshape alone — the
model must discover for itself which blade slot to put a quantity in, and any
rotor constraint downstream then acts on an arbitrary scramble.

These layers close that gap:

- `CliffordLinear` mixes channels with a separate weight matrix per grade, so
  a rotation applied to every channel commutes with the layer (equivariance):
    L(R x R~) == R L(x) R~
- `GeoEmbed` places known geometric quantities directly in their blade slots
  (rigid-body omega and magnetic fields are bivectors, velocities are vectors),
  keeping the direction intact and learning only per-channel scales.
- `GeoHead` reads the prediction back out of the same grade.

Together they make the whole path grade-typed; equivariance still breaks at
the SiLU gate and the depthwise conv, which stay for parity with the baseline.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from geomamba.algebra import Algebra


class CliffordLinear(nn.Module):
    """[.., C_in, dim] -> [.., C_out, dim], one weight matrix per grade."""

    def __init__(self, al: Algebra, c_in: int, c_out: int):
        super().__init__()
        self.al = al
        self.c_in, self.c_out = c_in, c_out
        self.weight = nn.Parameter(torch.empty(al.n + 1, c_out, c_in))
        nn.init.uniform_(self.weight, -1 / math.sqrt(c_in), 1 / math.sqrt(c_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # per-blade weights: pick each blade's grade row -> [dim, c_out, c_in]
        grades = self.al.grades.to(x.device)
        w = self.weight.to(x.dtype)[grades]  # [dim, c_out, c_in]
        return torch.einsum("...ib,boi->...ob", x, w)


class GeoEmbed(nn.Module):
    """Slot input features into blade positions of `grade`, plus a scalar slot.

    features [.., F] -> multivector channels [.., c_out, dim].
    The first `len(grade blades)` features are treated as one geometric object
    of that grade (direction preserved); any remaining features go to grade 0.
    """

    def __init__(self, al: Algebra, n_features: int, c_out: int, grade: int = 2):
        super().__init__()
        if len(al.blade_indices_of_grade[grade]) > n_features:
            raise ValueError(
                f"need >= {len(al.blade_indices_of_grade[grade])} features "
                f"to slot a grade-{grade} object"
            )
        self.al = al
        self.grade = grade
        self.n_geo = len(al.blade_indices_of_grade[grade])
        self.n_scalar = n_features - self.n_geo
        self.mix = CliffordLinear(al, 1, c_out)
        self.scalar_proj = (
            nn.Linear(self.n_scalar, c_out, bias=False) if self.n_scalar else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mv = x.new_zeros(*x.shape[:-1], 1, self.al.dim)
        mv[..., 0, self.al.blade_indices_of_grade[self.grade]] = x[..., : self.n_geo]
        out = self.mix(mv)
        if self.scalar_proj is not None:
            out = out.clone()
            out[..., 0] = out[..., 0] + self.scalar_proj(x[..., self.n_geo :])
        return out


class GeoHead(nn.Module):
    """Read a prediction of the same geometric type back out: [.., C, dim] ->
    [.., F]. Grade coefficients come from a Clifford-linear pooling to one
    channel; extra scalar features (if any) come from the scalar slot."""

    def __init__(self, al: Algebra, c_in: int, n_features: int, grade: int = 2):
        super().__init__()
        self.al = al
        self.grade = grade
        self.n_geo = len(al.blade_indices_of_grade[grade])
        self.n_scalar = n_features - self.n_geo
        self.pool = CliffordLinear(al, c_in, 1)
        self.scalar_out = (
            nn.Linear(1, self.n_scalar, bias=False) if self.n_scalar else None
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(h)[..., 0, :]  # [.., dim]
        out = pooled[..., self.al.blade_indices_of_grade[self.grade]]
        if self.scalar_out is not None:
            out = torch.cat([out, self.scalar_out(pooled[..., :1])], dim=-1)
        return out


class GradeRMSNorm(nn.Module):
    """RMSNorm whose learnable gain is shared within each grade, so it rescales
    geometric objects without mixing or distorting their components."""

    def __init__(self, al: Algebra, channels: int, eps: float = 1e-6):
        super().__init__()
        self.al = al
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels, al.n + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [.., C, dim]"""
        rms = torch.sqrt(x.pow(2).mean(dim=(-2, -1), keepdim=True) + self.eps)
        grades = self.al.grades.to(x.device)
        gain = self.weight.to(x.dtype)[:, grades]  # [C, dim]
        return x / rms * gain
