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
import torch.nn.functional as F

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


def resolve_blades(al: Algebra, grade: int, blades: list[int] | None) -> list[int]:
    """Which blade slots an input object occupies.

    `blades` overrides `grade` and may name slots of mixed grade — that is the
    `scatter` control: same sparsity and parameter count as grade slotting, but
    the chosen slots do not rotate together, so the triple is no longer a
    geometric object. If the two agree, geometry is not what slotting buys.
    """
    if blades is None:
        return list(al.blade_indices_of_grade[grade])
    if not all(0 <= b < al.dim for b in blades):
        raise ValueError(f"blade indices out of range for dim {al.dim}: {blades}")
    if len(set(blades)) != len(blades):
        raise ValueError(f"duplicate blade indices: {blades}")
    return list(blades)


class GeoEmbed(nn.Module):
    """Slot input features into blade positions of `grade`, plus a scalar slot.

    features [.., F] -> multivector channels [.., c_out, dim].
    The first `len(blades)` features are treated as one geometric object of
    that grade (direction preserved); any remaining features go to grade 0.
    Pass `blades` to place them somewhere other than a whole grade.
    """

    def __init__(
        self,
        al: Algebra,
        n_features: int,
        c_out: int,
        grade: int = 2,
        blades: list[int] | None = None,
    ):
        super().__init__()
        self.blades = resolve_blades(al, grade, blades)
        if len(self.blades) > n_features:
            raise ValueError(
                f"need >= {len(self.blades)} features to slot into {self.blades}"
            )
        self.al = al
        self.grade = grade
        self.n_geo = len(self.blades)
        self.n_scalar = n_features - self.n_geo
        self.mix = CliffordLinear(al, 1, c_out)
        self.scalar_proj = (
            nn.Linear(self.n_scalar, c_out, bias=False) if self.n_scalar else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mv = x.new_zeros(*x.shape[:-1], 1, self.al.dim)
        mv[..., 0, self.blades] = x[..., : self.n_geo]
        out = self.mix(mv)
        if self.scalar_proj is not None:
            out = out.clone()
            out[..., 0] = out[..., 0] + self.scalar_proj(x[..., self.n_geo :])
        return out


class GeoHead(nn.Module):
    """Read a prediction of the same geometric type back out: [.., C, dim] ->
    [.., F]. Grade coefficients come from a Clifford-linear pooling to one
    channel; extra scalar features (if any) come from the scalar slot."""

    def __init__(
        self,
        al: Algebra,
        c_in: int,
        n_features: int,
        grade: int = 2,
        blades: list[int] | None = None,
    ):
        super().__init__()
        self.al = al
        self.grade = grade
        self.blades = resolve_blades(al, grade, blades)
        self.n_geo = len(self.blades)
        self.n_scalar = n_features - self.n_geo
        self.pool = CliffordLinear(al, c_in, 1)
        self.scalar_out = (
            nn.Linear(1, self.n_scalar, bias=False) if self.n_scalar else None
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(h)[..., 0, :]  # [.., dim]
        out = pooled[..., self.blades]
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


def grade_invariants(al: Algebra, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spin-invariant summary of multivector channels: [.., C, dim] -> [.., C*(n+1)].

    Grade norms are unchanged by the sandwich (each grade subspace is an
    orthogonal representation), so anything computed from these is invariant --
    which is what lets a gate or a step size be input-dependent AND equivariant.
    """
    masks = al.grade_masks.to(dtype=x.dtype, device=x.device)
    sq = torch.einsum("...i,gi->...g", x**2, masks)
    return torch.sqrt(sq + eps).flatten(-2)


def _grade_gains(al: Algebra, s: torch.Tensor, c: int) -> torch.Tensor:
    """Per-(channel, grade) scalars -> a per-blade multiplier [.., C, dim]."""
    masks = al.grade_masks.to(dtype=s.dtype, device=s.device)
    return s.reshape(*s.shape[:-1], c, al.n + 1) @ masks


class GeoGate(nn.Module):
    """Equivariant replacement for `u * silu(z)`.

    The baseline gate multiplies blade coefficients elementwise, which mixes a
    multivector with an arbitrary one and destroys equivariance. Here the gate
    is a scalar per (channel, grade), computed from invariants of z -- scaling a
    whole grade commutes with the sandwich, so the block stays equivariant while
    keeping the input-dependent gating that makes Mamba work.
    """

    def __init__(self, al: Algebra, channels: int):
        super().__init__()
        self.al, self.c = al, channels
        n_inv = channels * (al.n + 1)
        self.to_gain = nn.Linear(n_inv, n_inv)

    def forward(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        s = F.silu(self.to_gain(grade_invariants(self.al, z)))
        return u * _grade_gains(self.al, s, self.c)


class GeometricBilinear(nn.Module):
    """The geometric product itself: y = W3( (W1 u) * (W2 z) ).

    This is the only place where grades interact -- a vector times a vector
    produces scalar and bivector parts. Equivariant because
    R(xy)R~ = (RxR~)(RyR~), and it is what the linear+gate path structurally
    cannot express. Kept outside the scan so the recurrence stays linear.
    """

    def __init__(self, al: Algebra, c_in: int, c_out: int):
        super().__init__()
        self.al = al
        self.left = CliffordLinear(al, c_in, c_out)
        self.right = CliffordLinear(al, c_in, c_out)
        self.mix = CliffordLinear(al, c_out, c_out)

    def forward(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        cayley = self.al.cayley.to(dtype=u.dtype, device=u.device)
        prod = torch.einsum("...i,...j,ijk->...k", self.left(u), self.right(z), cayley)
        return self.mix(prod)


class BladeSharedConv1d(nn.Module):
    """Causal depthwise conv over time, one kernel per channel SHARED by every
    blade. A per-blade kernel (the baseline) filters blade coefficients
    independently, which is not equivariant; a shared kernel is a scalar
    combination of past multivectors, which is. No bias: adding a fixed
    non-scalar multivector would break equivariance again.
    """

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.k = kernel
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel) / math.sqrt(kernel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C, dim]"""
        b, l, c, d = x.shape
        t = x.permute(0, 3, 2, 1).reshape(b * d, c, l)
        t = F.conv1d(F.pad(t, (self.k - 1, 0)), self.weight.to(x.dtype), groups=c)
        return t.reshape(b, d, c, l).permute(0, 3, 2, 1)
