"""Associative scans for the affine recurrence h_t = A_t h_{t-1} + u_t (h_0 = u_1 ... i.e. h_{-1} = 0).

Two state-transition representations share one interface:
- diagonal: A, u of shape [B, L, D]           (vanilla S6, elementwise)
- block:    A [B, L, C, d, d], u [B, L, C, d] (geometric variants, small
             per-channel matrices, e.g. d = 8 for Cl(3,0))

The parallel versions use Hillis–Steele doubling over the composition
(A2, u2) ∘ (A1, u1) = (A2·A1, A2·u1 + u2): after ceil(log2 L) rounds the u
slot holds the scan output. Implemented functionally (torch.cat, no in-place)
so autograd works without a custom backward; costs O(log L) extra memory.
Sequential fallbacks trade speed for memory. Correctness is pinned by tests;
speed choice on ARM is made in the M1 benchmark.
"""
from __future__ import annotations

import torch


def scan_diag_sequential(A: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    h = torch.zeros_like(u[:, 0])
    out = []
    for t in range(u.shape[1]):
        h = A[:, t] * h + u[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def scan_diag_parallel(A: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    L = u.shape[1]
    offset = 1
    while offset < L:
        u = torch.cat(
            [u[:, :offset], u[:, offset:] + A[:, offset:] * u[:, :-offset]], dim=1
        )
        A = torch.cat([A[:, :offset], A[:, offset:] * A[:, :-offset]], dim=1)
        offset *= 2
    return u


def scan_block_sequential(A: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    h = torch.zeros_like(u[:, 0])
    out = []
    for t in range(u.shape[1]):
        h = (A[:, t] @ h.unsqueeze(-1)).squeeze(-1) + u[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def scan_block_parallel(A: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    L = u.shape[1]
    offset = 1
    while offset < L:
        carried = (A[:, offset:] @ u[:, :-offset].unsqueeze(-1)).squeeze(-1)
        u = torch.cat([u[:, :offset], u[:, offset:] + carried], dim=1)
        A = torch.cat([A[:, :offset], A[:, offset:] @ A[:, :-offset]], dim=1)
        offset *= 2
    return u


def scan_diag_chunked(A: torch.Tensor, u: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Parallel scan within chunks, sequential carry across chunks.

    Same result as the full scans but with an O(chunk) working set — on
    memory-bandwidth-bound CPUs this beats the full Hillis–Steele sweep.
    """
    L = u.shape[1]
    carry = torch.zeros_like(u[:, 0])
    outs = []
    for s in range(0, L, chunk):
        Ac, uc = A[:, s : s + chunk], u[:, s : s + chunk]
        # fold the carry into the chunk's first input
        uc = torch.cat([(uc[:, :1] + Ac[:, :1] * carry.unsqueeze(1)), uc[:, 1:]], dim=1)
        h = scan_diag_parallel(Ac, uc)
        carry = h[:, -1]
        outs.append(h)
    return torch.cat(outs, dim=1)


def scan_block_chunked(A: torch.Tensor, u: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    L = u.shape[1]
    carry = torch.zeros_like(u[:, 0])
    outs = []
    for s in range(0, L, chunk):
        Ac, uc = A[:, s : s + chunk], u[:, s : s + chunk]
        first = uc[:, :1] + (Ac[:, :1] @ carry.unsqueeze(1).unsqueeze(-1)).squeeze(-1)
        uc = torch.cat([first, uc[:, 1:]], dim=1)
        h = scan_block_parallel(Ac, uc)
        carry = h[:, -1]
        outs.append(h)
    return torch.cat(outs, dim=1)


def scan_diag(A: torch.Tensor, u: torch.Tensor, mode: str = "chunked") -> torch.Tensor:
    if mode == "parallel":
        return scan_diag_parallel(A, u)
    if mode == "sequential":
        return scan_diag_sequential(A, u)
    if mode == "chunked":
        return scan_diag_chunked(A, u)
    raise ValueError(f"unknown scan mode: {mode}")


def scan_block(A: torch.Tensor, u: torch.Tensor, mode: str = "chunked") -> torch.Tensor:
    if mode == "parallel":
        return scan_block_parallel(A, u)
    if mode == "sequential":
        return scan_block_sequential(A, u)
    if mode == "chunked":
        return scan_block_chunked(A, u)
    raise ValueError(f"unknown scan mode: {mode}")
