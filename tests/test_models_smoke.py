import torch

from geomamba.models.model import GeoMamba, ModelConfig


def tiny_model(**kw) -> GeoMamba:
    cfg = ModelConfig(
        n_features=4, d_model=16, n_layers=2, d_state=4, heads=["forecast"], **kw
    )
    return GeoMamba(cfg)


def test_forward_backward_shapes():
    model = tiny_model()
    x = torch.randn(3, 32, 4)
    out = model(x)["forecast"]
    assert out.shape == (3, 32, 4)
    out.sum().backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_causality():
    model = tiny_model()
    model.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 32, 4)
    x_perturbed = x.clone()
    x_perturbed[:, -1] += 10.0
    with torch.no_grad():
        y1 = model(x)["forecast"]
        y2 = model(x_perturbed)["forecast"]
    torch.testing.assert_close(y1[:, :-1], y2[:, :-1])
    assert not torch.allclose(y1[:, -1], y2[:, -1])


def test_parallel_and_sequential_scan_agree_in_model():
    torch.manual_seed(0)
    m_par = tiny_model(scan_mode="chunked")
    m_seq = tiny_model(scan_mode="sequential")
    m_seq.load_state_dict(m_par.state_dict())
    m_par.eval(), m_seq.eval()
    x = torch.randn(2, 48, 4)
    with torch.no_grad():
        torch.testing.assert_close(
            m_par(x)["forecast"], m_seq(x)["forecast"], atol=1e-5, rtol=1e-5
        )


def test_rollout_shape():
    model = tiny_model()
    model.eval()
    preds = model.rollout(torch.randn(2, 16, 4), horizon=5)
    assert preds.shape == (2, 5, 4)
