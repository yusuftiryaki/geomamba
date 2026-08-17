"""Equivariance's other half: what must NOT change under a rotation.

`test_equivariance.py` checks that the output rotates with the input. That is
necessary but it hides *why*: the architecture only works because every
input-dependent SCALAR it computes (step size, gate, dissipation, readout
gains) is a function of rotation-invariant quantities. If one of those silently
started depending on blade coefficients directly, end-to-end equivariance would
break and the test would say "not equivariant" without saying which piece.
These tests pin each invariant separately.

A separate warning is documented at the bottom: group invariance is NOT
conservation. An exactly equivariant model can still let energy drift, because
nothing here derives a conserved quantity from the symmetry -- that would need
variational or symplectic structure, which this architecture does not have.
Confusing the two is what made the rotor arm look self-contradictory for weeks:
it conserved the state norm |h| while the observable energy drifted, because
|h| was never the invariant of the physics.
"""
from __future__ import annotations

import pytest
import torch

from geomamba.algebra import Algebra
from geomamba.algebra.rotor import bivector_exp, sandwich
from geomamba.models.geo_layers import GeoGate, grade_invariants
from geomamba.models.geo_ssm_eq import EquivariantGeoS6

AL = Algebra.get(3, 0)


def _rotate(x: torch.Tensor) -> torch.Tensor:
    """Apply one random rotor to every multivector channel: x -> R x R~."""
    rotor = bivector_exp(AL, torch.randn(1, 3, dtype=torch.float64) * 0.9)
    flat = x.reshape(-1, AL.dim)
    return sandwich(AL, rotor.expand(flat.shape[0], -1), flat).reshape(x.shape)


def _state(seed: int = 0):
    torch.manual_seed(seed)
    return torch.randn(2, 8, 3, AL.dim, dtype=torch.float64)


def test_grade_invariants_are_invariant():
    x = _state()
    assert torch.allclose(grade_invariants(AL, _rotate(x)), grade_invariants(AL, x),
                          atol=1e-12)


@pytest.mark.parametrize("attr", ["to_dt", "to_gate", "to_lambda", "to_read"])
def test_every_input_dependent_scalar_of_the_ssm_is_invariant(attr):
    """dt, input gate, dissipation and readout gains must not see the frame."""
    torch.manual_seed(0)
    ssm = EquivariantGeoS6(AL, channels=3, scan_mode="sequential").double()
    x = _state()
    layer = getattr(ssm, attr)
    a = layer(grade_invariants(AL, x))
    b = layer(grade_invariants(AL, _rotate(x)))
    assert torch.allclose(a, b, atol=1e-12), f"{attr} depends on the frame"


def test_the_rotor_generator_is_equivariant_not_invariant():
    """The generator is the one input-dependent quantity that MUST rotate --
    a rotation axis that ignored the frame would be a bug, not a feature."""
    torch.manual_seed(0)
    ssm = EquivariantGeoS6(AL, channels=3, scan_mode="sequential").double()
    x = _state()
    g, g_rot = ssm.gen(x), ssm.gen(_rotate(x))
    assert not torch.allclose(g, g_rot, atol=1e-6), "generator must not be invariant"
    # ... but its magnitude per grade is invariant, which is what makes dt safe
    assert torch.allclose(grade_invariants(AL, g), grade_invariants(AL, g_rot),
                          atol=1e-10)


def test_gate_gains_are_invariant_even_though_the_gated_output_is_not():
    torch.manual_seed(0)
    gate = GeoGate(AL, channels=3).double()
    u, z = _state(0), _state(1)
    inv_z, inv_z_rot = grade_invariants(AL, z), grade_invariants(AL, _rotate(z))
    assert torch.allclose(gate.to_gain(inv_z), gate.to_gain(inv_z_rot), atol=1e-12)


def test_invariance_does_not_imply_conservation():
    """Documents the distinction, with a concrete demonstration: the equivariant
    SSM is free to change the norm of the state. That freedom is deliberate --
    a free rigid body does NOT conserve |omega| (it conserves energy and |I
    omega|), so a norm-preserving transition would impose a false physical law.
    Conservation, if we ever want it, needs its own structure; symmetry alone
    does not deliver it.
    """
    torch.manual_seed(0)
    ssm = EquivariantGeoS6(AL, channels=3, scan_mode="sequential").double()
    with torch.no_grad():
        ssm.to_lambda.weight.normal_(0, 0.5)  # let dissipation follow the input
    x = _state()
    norms = ssm(x).reshape(2, 8, -1).norm(dim=-1)
    assert (norms.max(dim=1).values / norms.min(dim=1).values > 1.01).all(), (
        "the state norm should be free to vary; if it cannot, the architecture "
        "is imposing a conservation law the physics may not have"
    )
