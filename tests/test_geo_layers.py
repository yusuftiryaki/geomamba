import torch

from geomamba.algebra import Algebra, multivector as mv, rotor
from geomamba.models.geo_layers import CliffordLinear, GeoEmbed, GeoHead, GradeRMSNorm
from geomamba.models.model import GeoMamba, ModelConfig


def random_rotor(al, n=1, seed=0):
    torch.manual_seed(seed)
    return rotor.bivector_exp(al, torch.randn(n, 3, dtype=torch.float64))


def test_clifford_linear_is_rotation_equivariant():
    al = Algebra.get(3, 0)
    layer = CliffordLinear(al, 4, 6).double()
    x = torch.randn(5, 4, al.dim, dtype=torch.float64)
    R = random_rotor(al, 1)

    rot_then_mix = layer(rotor.sandwich(al, R, x))
    mix_then_rot = rotor.sandwich(al, R, layer(x))
    torch.testing.assert_close(rot_then_mix, mix_then_rot)


def test_geo_embed_slots_features_into_grade():
    al = Algebra.get(3, 0)
    emb = GeoEmbed(al, n_features=3, c_out=4, grade=2).double()
    x = torch.randn(7, 3, dtype=torch.float64)
    out = emb(x)
    assert out.shape == (7, 4, al.dim)
    # nothing outside grade 2 is populated (no extra scalar features here)
    non_biv = out - mv.grade_project(al, out, 2)
    assert non_biv.abs().max() == 0
    # direction preserved per channel: each channel is a scalar multiple of x
    coeffs = mv.extract_grade(al, out, 2)
    ratios = coeffs / x.unsqueeze(-2)
    torch.testing.assert_close(ratios, ratios[..., :1].expand_as(ratios))


def test_embed_head_roundtrip_is_equivariant():
    al = Algebra.get(3, 0)
    emb = GeoEmbed(al, 3, 4, grade=2).double()
    head = GeoHead(al, 4, 3, grade=2).double()
    torch.manual_seed(1)
    x = torch.randn(6, 3, dtype=torch.float64)
    R = random_rotor(al, 1, seed=2)

    def rotate_vec(v):
        emb_v = mv.embed_grade(al, v, 2)
        return mv.extract_grade(al, rotor.sandwich(al, R, emb_v), 2)

    out_then_rot = rotate_vec(head(emb(x)))
    rot_then_out = head(emb(rotate_vec(x)))
    torch.testing.assert_close(out_then_rot, rot_then_out)


def test_grade_rmsnorm_preserves_direction():
    al = Algebra.get(3, 0)
    norm = GradeRMSNorm(al, 3).double()
    x = torch.randn(4, 3, al.dim, dtype=torch.float64)
    out = norm(x)
    # with unit gains it is a pure rescale, so grade directions are unchanged
    ratios = out / x
    for k in range(al.n + 1):
        idx = al.blade_indices_of_grade[k]
        r = ratios[..., idx]
        torch.testing.assert_close(r, r[..., :1].expand_as(r))


def test_model_variants_with_geo_embed_train_step():
    for kw in [{"geo_embed": True}, {"geo_embed": True, "geo_proj": True}]:
        model = GeoMamba(ModelConfig(n_features=3, d_model=32, n_layers=2,
                                     variant="rotor_mamba", **kw))
        out = model(torch.randn(2, 24, 3))["forecast"]
        assert out.shape == (2, 24, 3)
        out.sum().backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
