from __future__ import annotations

from collections.abc import Sequence

import torch

from src.lwe.modular import center_mod


def residual_score_binary(A: torch.Tensor, b: torch.Tensor, support: Sequence[int], q: int) -> float:
    n = A.shape[1]
    s_hat = torch.zeros(n, dtype=torch.long, device=A.device)
    s_hat[list(support)] = 1
    residual = torch.remainder(b - A.matmul(s_hat), q)
    residual_center = center_mod(residual, q).float()
    return float(torch.mean(torch.abs(residual_center)).item())


def residual_score_for_secret(A: torch.Tensor, b: torch.Tensor, s_hat: torch.Tensor, q: int) -> float:
    residual = torch.remainder(b - A.matmul(s_hat.to(A.device)), q)
    residual_center = center_mod(residual, q).float()
    return float(torch.mean(torch.abs(residual_center)).item())

