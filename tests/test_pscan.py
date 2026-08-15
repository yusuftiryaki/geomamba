import pytest
import torch

from geomamba.models import pscan


@pytest.mark.parametrize("L", [1, 2, 37, 64])
def test_diag_parallel_equals_sequential(L):
    torch.manual_seed(0)
    A = torch.rand(3, L, 5, dtype=torch.float64) * 0.9 + 0.05
    u = torch.randn(3, L, 5, dtype=torch.float64)
    torch.testing.assert_close(
        pscan.scan_diag_parallel(A, u), pscan.scan_diag_sequential(A, u)
    )


@pytest.mark.parametrize("L", [1, 2, 37, 64])
def test_block_parallel_equals_sequential(L):
    torch.manual_seed(1)
    # near-orthogonal-scale blocks to keep products bounded
    A = torch.randn(2, L, 3, 4, 4, dtype=torch.float64) * 0.4
    u = torch.randn(2, L, 3, 4, dtype=torch.float64)
    torch.testing.assert_close(
        pscan.scan_block_parallel(A, u), pscan.scan_block_sequential(A, u)
    )


@pytest.mark.parametrize("L,chunk", [(1, 16), (37, 16), (64, 16), (100, 32)])
def test_chunked_equals_sequential(L, chunk):
    torch.manual_seed(2)
    A = torch.rand(2, L, 5, dtype=torch.float64) * 0.9 + 0.05
    u = torch.randn(2, L, 5, dtype=torch.float64)
    torch.testing.assert_close(
        pscan.scan_diag_chunked(A, u, chunk), pscan.scan_diag_sequential(A, u)
    )
    Ab = torch.randn(2, L, 2, 3, 3, dtype=torch.float64) * 0.4
    ub = torch.randn(2, L, 2, 3, dtype=torch.float64)
    torch.testing.assert_close(
        pscan.scan_block_chunked(Ab, ub, chunk), pscan.scan_block_sequential(Ab, ub)
    )


def test_diag_gradcheck():
    A = (torch.rand(1, 5, 2, dtype=torch.float64) * 0.8 + 0.1).requires_grad_()
    u = torch.randn(1, 5, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(pscan.scan_diag_parallel, (A, u))


def test_block_gradcheck():
    A = (torch.randn(1, 4, 1, 3, 3, dtype=torch.float64) * 0.3).requires_grad_()
    u = torch.randn(1, 4, 1, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(pscan.scan_block_parallel, (A, u))
