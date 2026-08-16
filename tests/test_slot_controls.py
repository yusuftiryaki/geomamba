"""The slot-control grid is only meaningful if its arms differ the way the
pre-registration claims (report/preregistration_slot_controls.md). These tests
check that on the bench, before a GPU session is spent on it.

Claims under test:
  1. grade-1 and grade-2 blade sets are CLOSED under the rotor sandwich, so
     each carries a genuine SO(3) action -> `slot1`/`slot2` are geometric.
  2. the two differ only by a fixed relabelling P (M2 = P M1 P^T, same P for
     every rotor), which downstream free Linear layers can absorb -> the
     pre-registered prediction "slot1 must not matter" is well founded.
  3. the `scatter` blade set is NOT closed: a rotation moves content out of the
     slotted positions, so the triple is not a geometric object -> the control
     really does remove geometry while keeping sparsity and parameter count.
  4. `freeze_interface` actually stops the interface from training.
"""
from __future__ import annotations

import itertools

import torch

from geomamba.algebra import Algebra
from geomamba.algebra.rotor import bivector_exp, sandwich
from geomamba.models.model import GeoMamba, ModelConfig

GRADE1 = [1, 2, 4]
GRADE2 = [3, 5, 6]
SCATTER = [0, 1, 7]  # scalar + one vector blade + pseudoscalar


def _induced(al: Algebra, rotor: torch.Tensor, blades: list[int]) -> torch.Tensor:
    """Matrix of the sandwich restricted to `blades`, plus what leaks out."""
    cols = []
    for b in blades:
        e = torch.zeros(1, al.dim)
        e[0, b] = 1.0
        cols.append(sandwich(al, rotor, e)[0])
    full = torch.stack(cols, dim=1)  # [dim, len(blades)]
    leak = full.clone()
    leak[blades, :] = 0
    return full[blades, :], leak


def _rotors(al: Algebra, n: int = 4) -> list[torch.Tensor]:
    torch.manual_seed(1)
    return [bivector_exp(al, torch.randn(1, 3) * s) for s in (0.3, 0.9, 1.7, 2.5)[:n]]


def test_grade_blade_sets_are_closed_and_are_rotations():
    al = Algebra.get(3, 0)
    for rotor in _rotors(al):
        for blades in (GRADE1, GRADE2):
            inside, leak = _induced(al, rotor, blades)
            assert leak.abs().max() < 1e-6, "grade slots must be closed"
            assert torch.linalg.det(inside) > 0.99
            assert torch.allclose(inside @ inside.T, torch.eye(3), atol=1e-5)


def test_scatter_blade_set_is_not_closed():
    """Without this the `scatter` arm would be a geometric slotting too, and
    the whole control would be void."""
    al = Algebra.get(3, 0)
    leaks = [_induced(al, r, SCATTER)[1].abs().max().item() for r in _rotors(al)]
    assert max(leaks) > 0.1, f"scatter must leak out of its slots, got {leaks}"


def test_grade1_and_grade2_differ_only_by_a_fixed_relabelling():
    al = Algebra.get(3, 0)
    rotors = _rotors(al)
    m1 = [_induced(al, r, GRADE1)[0] for r in rotors]
    m2 = [_induced(al, r, GRADE2)[0] for r in rotors]
    best = min(
        max((P @ a @ P.T - b).abs().max().item() for a, b in zip(m1, m2))
        for P in _signed_permutations()
    )
    assert best < 1e-5, f"no fixed relabelling links the two grades (err {best})"


def _signed_permutations():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            P = torch.zeros(3, 3)
            for i, (p, s) in enumerate(zip(perm, signs)):
                P[i, p] = float(s)
            yield P


def test_scatter_and_slot_have_the_same_parameter_count():
    """The control must not win or lose on capacity."""
    base = dict(variant="mamba", n_features=3, d_model=72, d_state=16, geo_embed=True)
    slot = GeoMamba(ModelConfig(slot_grade=2, **base))
    scatter = GeoMamba(ModelConfig(slot_blades=SCATTER, **base))
    assert sum(p.numel() for p in slot.parameters()) == sum(
        p.numel() for p in scatter.parameters()
    )


def test_freeze_interface_freezes_embed_and_head():
    cfg = dict(variant="mamba", n_features=3, d_model=76)
    free = GeoMamba(ModelConfig(**cfg))
    frozen = GeoMamba(ModelConfig(freeze_interface=True, **cfg))
    assert sum(p.numel() for p in free.parameters()) == sum(
        p.numel() for p in frozen.parameters()
    )
    trainable = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)  # noqa: E731
    assert trainable(frozen) < trainable(free)
    for module in (frozen.backbone.embed, frozen.heads["forecast"]):
        assert all(not p.requires_grad for p in module.parameters())


def test_slot_blades_round_trip_through_the_model():
    """A scattered model still runs end to end and produces the right shape."""
    model = GeoMamba(
        ModelConfig(variant="rotor_mamba", n_features=3, d_model=64,
                    geo_embed=True, slot_blades=SCATTER)
    )
    out = model(torch.randn(2, 16, 3))["forecast"]
    assert out.shape == (2, 16, 3)
