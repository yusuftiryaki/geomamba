import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from geomamba.algebra import Algebra, multivector as mv, rotor


def make_rotor(al, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """R = exp(-angle/2 * (I n)) for unit axis n, over bivector basis [e12, e13, e23].

    In Cl(3,0): I*n = n1 e23 - n2 e13 + n3 e12.
    """
    n1, n2, n3 = axis.unbind(-1)
    b = torch.stack([n3, -n2, n1], dim=-1)
    return rotor.bivector_exp(al, -0.5 * angle.unsqueeze(-1) * b)


def test_rotor_matches_scipy_rotation():
    al = Algebra.get(3, 0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-2 * np.pi, 2 * np.pi)
        v = rng.normal(size=3)

        r = make_rotor(
            al,
            torch.tensor(axis, dtype=torch.float64),
            torch.tensor(angle, dtype=torch.float64),
        )
        v_mv = mv.embed_grade(al, torch.tensor(v, dtype=torch.float64), 1)
        out = rotor.sandwich(al, r, v_mv)

        expected = Rotation.from_rotvec(angle * axis).apply(v)
        torch.testing.assert_close(
            mv.extract_grade(al, out, 1),
            torch.tensor(expected),
            atol=1e-9,
            rtol=1e-9,
        )
        # sandwich of a vector by a rotor stays grade 1
        non_vector = out - mv.grade_project(al, out, 1)
        assert non_vector.abs().max() < 1e-9


def test_sandwich_preserves_norm_and_grades():
    al = Algebra.get(3, 0)
    torch.manual_seed(0)
    b = torch.randn(16, 3, dtype=torch.float64)
    r = rotor.bivector_exp(al, b)
    x = torch.randn(16, al.dim, dtype=torch.float64)
    out = rotor.sandwich(al, r, x)
    torch.testing.assert_close(mv.grade_norms(al, out), mv.grade_norms(al, x))


def test_sandwich_matrix_equals_direct():
    al = Algebra.get(3, 0)
    torch.manual_seed(1)
    b = torch.randn(16, 3, dtype=torch.float64)
    r = rotor.bivector_exp(al, b)
    x = torch.randn(16, al.dim, dtype=torch.float64)
    direct = rotor.sandwich(al, r, x)
    via_matrix = (rotor.sandwich_matrix(al, r) @ x.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(via_matrix, direct)
    # unit rotor => orthogonal sandwich matrix
    s = rotor.sandwich_matrix(al, r)
    eye = torch.eye(al.dim, dtype=torch.float64).expand_as(s)
    torch.testing.assert_close(s @ s.transpose(-1, -2), eye)


def test_rotor_composition_same_axis():
    al = Algebra.get(3, 0)
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    t1, t2 = torch.tensor(0.7, dtype=torch.float64), torch.tensor(1.1, dtype=torch.float64)
    r12 = mv.gp(al, make_rotor(al, axis, t1), make_rotor(al, axis, t2))
    torch.testing.assert_close(r12, make_rotor(al, axis, t1 + t2))


def test_bivector_exp_gradients_at_zero():
    al = Algebra.get(3, 0)
    b = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
    r = rotor.bivector_exp(al, b)
    x = torch.randn(4, al.dim, dtype=torch.float64)
    rotor.sandwich(al, r, x).sum().backward()
    assert torch.isfinite(b.grad).all()

    # identity rotor at b = 0
    with torch.no_grad():
        expected = torch.zeros(4, al.dim, dtype=torch.float64)
        expected[:, 0] = 1.0
        torch.testing.assert_close(rotor.bivector_exp(al, b.detach()), expected)


def test_bivector_exp_rejects_unsupported_signature():
    al = Algebra.get(1, 3)
    with pytest.raises(NotImplementedError):
        rotor.bivector_exp(al, torch.randn(2, len(al.bivector_indices)))


def test_rotation_matrix_matches_sandwich():
    al = Algebra.get(3, 0)
    torch.manual_seed(3)
    g = torch.randn(32, 3, dtype=torch.float64)
    dt = torch.rand(32, dtype=torch.float64) * 2

    def biv_to_axis(c):
        return torch.stack([c[..., 2], -c[..., 1], c[..., 0]], dim=-1)

    Q = rotor.rotation_matrix(g, dt)
    R = rotor.bivector_exp(al, -0.5 * dt.unsqueeze(-1) * g)

    v = torch.randn(32, 3, dtype=torch.float64)
    out_v = rotor.sandwich(al, R, mv.embed_grade(al, v, 1))
    torch.testing.assert_close(
        mv.extract_grade(al, out_v, 1), (Q @ v.unsqueeze(-1)).squeeze(-1)
    )

    c = torch.randn(32, 3, dtype=torch.float64)
    out_b = rotor.sandwich(al, R, mv.embed_grade(al, c, 2))
    expected = biv_to_axis((Q @ biv_to_axis(c).unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(mv.extract_grade(al, out_b, 2), expected)

    # grades 0 and 3 are invariant under the sandwich
    sp = torch.randn(32, 2, dtype=torch.float64)
    x = mv.embed_grade(al, sp[:, :1], 0) + mv.embed_grade(al, sp[:, 1:], 3)
    torch.testing.assert_close(rotor.sandwich(al, R, x), x)

    # orthogonality and differentiability at zero rotation
    torch.testing.assert_close(
        Q @ Q.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand_as(Q)
    )
    g0 = torch.zeros(4, 3, dtype=torch.float64, requires_grad=True)
    rotor.rotation_matrix(g0, torch.ones(4, dtype=torch.float64)).sum().backward()
    assert torch.isfinite(g0.grad).all()
