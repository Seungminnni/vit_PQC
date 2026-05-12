from __future__ import annotations

import torch
import torch.nn.functional as F

from .lwe import centered_mod_float, secret_value_table


def secret_cross_entropy_loss(s_logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
    if s_logits.dim() != 3:
        raise ValueError("s_logits must have shape (B, n, num_classes).")
    if target_labels.shape != s_logits.shape[:2]:
        raise ValueError("target_labels must have shape (B, n).")
    return F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]), target_labels.reshape(-1))


def expected_secret_from_logits(s_logits: torch.Tensor, q: int, secret_dist: str) -> torch.Tensor:
    values = secret_value_table(q, secret_dist, device=s_logits.device).to(s_logits.dtype)
    probs = torch.softmax(s_logits, dim=-1)
    return probs.matmul(values)


def hard_secret_from_logits(s_logits: torch.Tensor, q: int, secret_dist: str) -> torch.Tensor:
    values = secret_value_table(q, secret_dist, device=s_logits.device)
    labels = s_logits.argmax(dim=-1)
    return values[labels]


def residual_consistency_loss(
    A: torch.Tensor,
    b: torch.Tensor,
    s_logits: torch.Tensor,
    q: int,
    secret_dist: str,
    noise_bound: float = 3.0,
    normalize: bool = True,
) -> torch.Tensor:
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    s_expected = expected_secret_from_logits(s_logits, q=q, secret_dist=secret_dist)
    pred = torch.matmul(A.to(s_expected.dtype), s_expected.unsqueeze(-1)).squeeze(-1)
    residual = centered_mod_float(b.to(s_expected.dtype) - pred, q)
    excess = torch.relu(residual.abs() - float(noise_bound))
    if normalize:
        excess = excess / (float(q) / 2.0)
    return excess.square().mean()
