import pytest
import torch

from geomamba.algebra import Algebra, multivector as mv

# Hand-written oracle for Cl(2,0), basis order [1, e1, e2, e12] = indices [0,1,2,3].
# CL20_TABLE[a][b] = (sign, blade_index)
CL20_TABLE = [
    [(1, 0), (1, 1), (1, 2), (1, 3)],
    [(1, 1), (1, 0), (1, 3), (1, 2)],
    [(1, 2), (-1, 3), (1, 0), (-1, 1)],
    [(1, 3), (-1, 2), (1, 1), (-1, 0)],
]


def test_cl20_full_table():
    al = Algebra.get(2, 0)
    for a in range(4):
        for b in range(4):
            sign, blade = CL20_TABLE[a][b]
            expected = torch.zeros(4)
            expected[blade] = sign
            torch.testing.assert_close(al.cayley[a, b], expected)


def test_cl30_spot_checks():
    al = Algebra.get(3, 0)
    # index order: 0:1, 1:e1, 2:e2, 3:e12, 4:e3, 5:e13, 6:e23, 7:e123
    e1, e2, e12, e3, e13, e23, I = 1, 2, 3, 4, 5, 6, 7

    def prod(a, b):
        row = al.cayley[a, b]
        k = int(row.abs().argmax())
        return int(row[k].item()), k

    assert prod(e1, e2) == (1, e12)
    assert prod(e2, e1) == (-1, e12)
    assert prod(e12, e12) == (-1, 0)
    assert prod(e12, e23) == (1, e13)
    assert prod(I, I) == (-1, 0)
    assert prod(I, e1) == (1, e23)
    assert prod(I, e2) == (-1, e13)
    assert prod(I, e3) == (1, e12)


@pytest.mark.parametrize("p,q", [(2, 0), (3, 0), (1, 3)])
def test_associativity(p, q):
    al = Algebra.get(p, q)
    torch.manual_seed(0)
    x, y, z = torch.randn(3, 8, al.dim, dtype=torch.float64)
    left = mv.gp(al, mv.gp(al, x, y), z)
    right = mv.gp(al, x, mv.gp(al, y, z))
    torch.testing.assert_close(left, right)


@pytest.mark.parametrize("p,q", [(3, 0), (1, 3)])
def test_reversion_antiautomorphism(p, q):
    al = Algebra.get(p, q)
    torch.manual_seed(1)
    x, y = torch.randn(2, 8, al.dim, dtype=torch.float64)
    lhs = mv.reverse(al, mv.gp(al, x, y))
    rhs = mv.gp(al, mv.reverse(al, y), mv.reverse(al, x))
    torch.testing.assert_close(lhs, rhs)


def test_left_right_matrix_consistency():
    al = Algebra.get(3, 0)
    torch.manual_seed(2)
    x, y = torch.randn(2, 8, al.dim, dtype=torch.float64)
    prod = mv.gp(al, x, y)
    torch.testing.assert_close((mv.left_matrix(al, x) @ y.unsqueeze(-1)).squeeze(-1), prod)
    torch.testing.assert_close((mv.right_matrix(al, y) @ x.unsqueeze(-1)).squeeze(-1), prod)


def test_embed_extract_roundtrip():
    al = Algebra.get(3, 0)
    v = torch.randn(5, 3)
    assert torch.equal(mv.extract_grade(al, mv.embed_grade(al, v, 1), 1), v)
    b = torch.randn(5, 3)
    assert torch.equal(mv.extract_grade(al, mv.embed_grade(al, b, 2), 2), b)


def test_grade_norms():
    al = Algebra.get(3, 0)
    x = mv.embed_grade(al, torch.tensor([[3.0, 4.0, 0.0]]), 1)
    norms = mv.grade_norms(al, x)
    torch.testing.assert_close(norms[0], torch.tensor([0.0, 5.0, 0.0, 0.0]))


def test_against_clifford_library_oracle():
    clifford = pytest.importorskip("clifford")
    layout, blades = clifford.Cl(3)
    al = Algebra.get(3, 0)
    torch.manual_seed(3)
    x, y = torch.randn(2, al.dim, dtype=torch.float64)
    ours = mv.gp(al, x, y)
    # clifford uses the same bitmask blade ordering for Cl(3)
    theirs = torch.tensor(
        (layout.MultiVector(x.numpy()) * layout.MultiVector(y.numpy())).value
    )
    torch.testing.assert_close(ours, theirs)
