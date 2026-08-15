"""Clifford algebra Cl(p, q) structure constants.

Basis blades are indexed by bitmask: bit i set means generator e_{i+1} is in
the blade, so index 0 is the scalar and index 2**n - 1 the pseudoscalar.
Generators e_1..e_p square to +1, e_{p+1}..e_{p+q} to -1. Within a blade,
generators are in canonical ascending order.

Everything downstream (geometric product, reversion, grade projection,
matricized left/right multiplication) is driven by tensors precomputed here
once per signature, so the hot path is pure einsum.
"""
from __future__ import annotations

from functools import lru_cache

import torch


def _reordering_sign(a: int, b: int) -> int:
    """Sign of sorting the concatenation of blades a, b into canonical order."""
    a >>= 1
    swaps = 0
    while a:
        swaps += (a & b).bit_count()
        a >>= 1
    return -1 if swaps & 1 else 1


def _blade_product(a: int, b: int, metric: tuple[int, ...]) -> tuple[int, int]:
    """Product of basis blades a*b -> (sign, blade)."""
    sign = _reordering_sign(a, b)
    common = a & b
    i = 0
    while common:
        if common & 1:
            sign *= metric[i]
        common >>= 1
        i += 1
    return sign, a ^ b


class Algebra:
    """Precomputed structure of Cl(p, q). Build via Algebra.get(p, q)."""

    def __init__(self, p: int, q: int):
        self.p, self.q = p, q
        self.n = p + q
        self.dim = 1 << self.n
        self.metric = tuple([1] * p + [-1] * q)

        cayley = torch.zeros(self.dim, self.dim, self.dim)
        for a in range(self.dim):
            for b in range(self.dim):
                sign, blade = _blade_product(a, b, self.metric)
                cayley[a, b, blade] = float(sign)
        self.cayley = cayley

        self.grades = torch.tensor([i.bit_count() for i in range(self.dim)])
        # reversion: (-1)^{g(g-1)/2}
        g = self.grades
        self.reversion_signs = torch.where((g * (g - 1) // 2) % 2 == 0, 1.0, -1.0)

        self.blade_indices_of_grade = {
            k: [i for i in range(self.dim) if i.bit_count() == k]
            for k in range(self.n + 1)
        }
        self.grade_masks = torch.stack(
            [(self.grades == k).float() for k in range(self.n + 1)]
        )  # [n+1, dim]

    @staticmethod
    @lru_cache(maxsize=None)
    def get(p: int, q: int = 0) -> "Algebra":
        return Algebra(p, q)

    @property
    def vector_indices(self) -> list[int]:
        return self.blade_indices_of_grade[1]

    @property
    def bivector_indices(self) -> list[int]:
        return self.blade_indices_of_grade[2]
