import torch

from geomamba.algebra import Algebra, rotor
from geomamba.models import pscan
from geomamba.models.geo_ssm import GeoS6
from geomamba.models.model import GeoMamba, ModelConfig


def make_model(variant: str) -> GeoMamba:
    cfg = ModelConfig(n_features=3, d_model=16, n_layers=2, variant=variant)
    return GeoMamba(cfg)


def test_variants_forward_backward():
    x = torch.randn(2, 32, 3)
    for variant in ["mv_mamba", "rotor_mamba"]:
        model = make_model(variant)
        out = model(x)["forecast"]
        assert out.shape == (2, 32, 3)
        out.sum().backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name


def test_rotor_variant_causality():
    model = make_model("rotor_mamba")
    model.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 32, 3)
    x2 = x.clone()
    x2[:, -1] += 10.0
    with torch.no_grad():
        y1, y2 = model(x)["forecast"], model(x2)["forecast"]
    torch.testing.assert_close(y1[:, :-1], y2[:, :-1])


def test_zero_generator_reduces_to_pure_decay():
    """rotor with b=0 and mv with G=0 both give A = decay*I, so with shared
    projections they must agree exactly (M3 DoD reduction check)."""
    torch.manual_seed(0)
    ssm_rotor = GeoS6(d_inner=32, transition="rotor")
    ssm_mv = GeoS6(d_inner=32, transition="mv")
    with torch.no_grad():
        ssm_rotor.gen_proj.weight.zero_()
        ssm_mv.G.zero_()
        for dst, src in [
            (ssm_mv.x_proj, ssm_rotor.x_proj),
            (ssm_mv.dt_proj, ssm_rotor.dt_proj),
        ]:
            dst.weight.copy_(src.weight)
            if src.bias is not None:
                dst.bias.copy_(src.bias)
        ssm_mv.lambda_raw.copy_(ssm_rotor.lambda_raw)
        ssm_mv.D.copy_(ssm_rotor.D)
    x = torch.randn(2, 24, 32)
    with torch.no_grad():
        torch.testing.assert_close(ssm_rotor(x), ssm_mv(x), atol=1e-6, rtol=1e-6)


def test_rotor_scan_preserves_norm_without_decay():
    """Pure rotor transitions (no decay): state injected at t=0 keeps its norm."""
    al = Algebra.get(3, 0)
    torch.manual_seed(1)
    b = torch.randn(2, 64, 4, 3, dtype=torch.float64) * 0.5
    A = rotor.sandwich_matrix(al, rotor.bivector_exp(al, b))
    u = torch.zeros(2, 64, 4, 8, dtype=torch.float64)
    u[:, 0] = torch.randn(2, 4, 8, dtype=torch.float64)
    h = pscan.scan_block(A, u, "sequential")
    norms = torch.linalg.vector_norm(h, dim=-1)  # [2, 64, 4]
    torch.testing.assert_close(
        norms, norms[:, :1].expand_as(norms), atol=1e-10, rtol=1e-10
    )


def test_param_counts_comparable():
    counts = {
        v: sum(p.numel() for p in make_model(v).parameters())
        for v in ["mamba", "mv_mamba", "rotor_mamba"]
    }
    base = counts["mamba"]
    for v, c in counts.items():
        assert abs(c - base) / base < 0.35, counts
