"""Does the network actually respect the geometry? Asserted, not claimed.

The `geo_proj` arm was described as "full geometric" but was never tested for
equivariance, and it does not have it: its gate multiplies blade coefficients
elementwise and its conv filters each blade separately, so rotating the input
does NOT rotate the output. That silent failure is what these tests prevent.

The statement under test, end to end: for a rotation acting on the slotted
grade by the 3x3 matrix M,

    model(x @ M.T) == model(x) @ M.T

which is exactly the property that makes a rotated test set (`--rot-test`)
score identically -- i.e. what "the model learned the geometry" means.
"""
from __future__ import annotations

import pytest
import torch

from geomamba.algebra import Algebra
from geomamba.algebra.rotor import bivector_exp, sandwich
from geomamba.models.model import GeoMamba, ModelConfig

AL = Algebra.get(3, 0)


def _induced(angle: float, blades: list[int]) -> torch.Tensor:
    """3x3 matrix the sandwich induces on the coefficients of `blades`.

    Built in float64 throughout: a float32 rotor is only orthogonal to ~1e-7,
    which would swamp the equivariance error we are trying to measure.
    """
    rotor = bivector_exp(AL, torch.randn(1, 3, dtype=torch.float64) * angle)
    cols = []
    for b in blades:
        e = torch.zeros(1, AL.dim, dtype=torch.float64)
        e[0, b] = 1.0
        cols.append(sandwich(AL, rotor, e)[0][blades])
    return torch.stack(cols, dim=1)


def _model(**over):
    cfg = dict(variant="full_geo", n_features=3, d_model=32, n_layers=2,
               expand=2, geo_embed=True, slot_grade=2, scan_mode="sequential")
    cfg.update(over)
    return GeoMamba(ModelConfig(**cfg)).double().eval()


def _rot_error(model, grade: int, angle: float) -> float:
    blades = AL.blade_indices_of_grade[grade]
    M = _induced(angle, blades)
    x = torch.randn(2, 12, 3, dtype=torch.float64)
    y = model(x)["forecast"]
    return float((model(x @ M.T)["forecast"] - y @ M.T).abs().max() / y.abs().max())


@pytest.mark.parametrize("grade", [1, 2])
@pytest.mark.parametrize("angle", [0.2, 1.1, 2.4])
def test_full_geo_is_equivariant(grade, angle):
    torch.manual_seed(0)
    err = _rot_error(_model(slot_grade=grade), grade, angle)
    assert err < 1e-8, f"not equivariant: relative error {err:.2e}"


def test_still_equivariant_without_the_bilinear_layer():
    torch.manual_seed(0)
    assert _rot_error(_model(geo_bilinear=False), 2, 0.9) < 1e-8


def test_bilinear_layer_actually_fires():
    """The geometric product must change the function, else the (d) rung is
    just (c) with extra parameters."""
    torch.manual_seed(0)
    x = torch.randn(2, 12, 3, dtype=torch.float64)
    assert not torch.allclose(
        _model(geo_bilinear=True)(x)["forecast"],
        _model(geo_bilinear=False)(x)["forecast"],
    )


def test_old_geo_proj_arm_is_not_equivariant():
    """Locks in the correction: the arm previously reported as `fullgeo` fails
    this test, so its result must not be cited as evidence about geometric
    architectures."""
    torch.manual_seed(0)
    model = GeoMamba(ModelConfig(
        variant="rotor_mamba", n_features=3, d_model=32, n_layers=2,
        geo_embed=True, geo_proj=True, slot_grade=2, scan_mode="sequential",
    )).double().eval()
    assert _rot_error(model, 2, 0.9) > 1e-3
