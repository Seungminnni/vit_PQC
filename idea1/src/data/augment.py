from __future__ import annotations

import torch


def row_permutation(A: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, m, _ = A.shape
    perm = torch.randperm(m, device=A.device)
    return A[:, perm, :], b[:, perm]


def column_permutation(A: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, n = A.shape
    perm = torch.randperm(n, device=A.device)
    return A[:, :, perm], s[:, perm], perm


def row_subsample(A: torch.Tensor, b: torch.Tensor, m_sub: int) -> tuple[torch.Tensor, torch.Tensor]:
    _, m, _ = A.shape
    if not 1 <= m_sub <= m:
        raise ValueError(f"m_sub must be in [1,{m}], got {m_sub}")
    idx = torch.randperm(m, device=A.device)[:m_sub]
    return A[:, idx, :], b[:, idx]

