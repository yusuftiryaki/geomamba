"""Batched multivector operations. Tensors have blade coefficients on the
trailing dim: x.shape == [..., algebra.dim]."""
from __future__ import annotations

import torch

from geomamba.algebra.cayley import Algebra


def gp(al: Algebra, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Geometric product x * y."""
    return torch.einsum("...i,...j,ijk->...k", x, y, al.cayley.to(dtype=x.dtype, device=x.device))


def reverse(al: Algebra, x: torch.Tensor) -> torch.Tensor:
    return x * al.reversion_signs.to(dtype=x.dtype, device=x.device)


def grade_project(al: Algebra, x: torch.Tensor, k: int) -> torch.Tensor:
    return x * al.grade_masks[k].to(dtype=x.dtype, device=x.device)


def grade_norms(al: Algebra, x: torch.Tensor) -> torch.Tensor:
    """Euclidean norm of each grade's coefficients: [..., n+1]."""
    masks = al.grade_masks.to(dtype=x.dtype, device=x.device)  # [n+1, dim]
    sq = torch.einsum("...i,gi->...g", x**2, masks)
    return torch.sqrt(sq.clamp_min(0))


def left_matrix(al: Algebra, x: torch.Tensor) -> torch.Tensor:
    """L(x) with (x*y) == L(x) @ y. Shape [..., dim, dim]."""
    return torch.einsum("...i,ijk->...kj", x, al.cayley.to(dtype=x.dtype, device=x.device))


def right_matrix(al: Algebra, y: torch.Tensor) -> torch.Tensor:
    """R(y) with (x*y) == R(y) @ x. Shape [..., dim, dim]."""
    return torch.einsum("...j,ijk->...ki", y, al.cayley.to(dtype=y.dtype, device=y.device))


def embed_grade(al: Algebra, coeffs: torch.Tensor, k: int) -> torch.Tensor:
    """Place [..., len(grade-k blades)] coefficients into full multivectors."""
    idx = al.blade_indices_of_grade[k]
    out = coeffs.new_zeros(*coeffs.shape[:-1], al.dim)
    out[..., idx] = coeffs
    return out


def extract_grade(al: Algebra, x: torch.Tensor, k: int) -> torch.Tensor:
    """Inverse of embed_grade: pull out grade-k coefficients."""
    return x[..., al.blade_indices_of_grade[k]]
