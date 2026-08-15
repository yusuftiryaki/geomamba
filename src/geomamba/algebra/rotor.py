"""Rotors: closed-form bivector exponential and sandwich products.

Restricted to Euclidean signatures Cl(p, 0) with p <= 3, where every bivector
B is simple and satisfies B^2 = -|B|^2, so

    exp(B) = cos(|B|) + sin(|B|)/|B| * B

holds exactly. The sin(t)/t factor is Taylor-guarded near t = 0 so gradients
stay finite at zero rotation.
"""
from __future__ import annotations

import torch

from geomamba.algebra.cayley import Algebra
from geomamba.algebra import multivector as mv

_SINC_EPS = 1e-4


def _sinc(theta: torch.Tensor) -> torch.Tensor:
    """sin(theta)/theta with a Taylor branch near 0 (theta >= 0)."""
    small = theta < _SINC_EPS
    # Where small, avoid 0/0 by substituting 1 before dividing.
    safe = torch.where(small, torch.ones_like(theta), theta)
    return torch.where(small, 1 - theta**2 / 6, torch.sin(safe) / safe)


def bivector_exp(al: Algebra, b: torch.Tensor) -> torch.Tensor:
    """exp(B) for bivector coefficients b: [..., n_biv] -> rotor [..., dim].

    b holds coefficients over al.bivector_indices (canonical blade order).
    """
    if al.q != 0 or al.p > 3:
        raise NotImplementedError("closed-form bivector exp requires Cl(p<=3, 0)")
    theta = torch.linalg.vector_norm(b, dim=-1, keepdim=True)
    scalar = torch.cos(theta)
    biv = _sinc(theta) * b
    rotor = b.new_zeros(*b.shape[:-1], al.dim)
    rotor[..., 0] = scalar.squeeze(-1)
    rotor[..., al.bivector_indices] = biv
    return rotor


def sandwich(al: Algebra, r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """R x R~ computed directly via two geometric products."""
    return mv.gp(al, mv.gp(al, r, x), mv.reverse(al, r))


def sandwich_matrix(al: Algebra, r: torch.Tensor) -> torch.Tensor:
    """S(R) with (R x R~) == S(R) @ x. Shape [..., dim, dim].

    S = R_mat(R~) @ L(R): first left-multiply by R, then right-multiply by R~.
    For unit rotors S is orthogonal and block-diagonal over grades.
    """
    return mv.right_matrix(al, mv.reverse(al, r)) @ mv.left_matrix(al, r)
