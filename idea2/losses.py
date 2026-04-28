from __future__ import annotations

import torch
import torch.nn.functional as F

from modular import circular_distance_loss


def support_bce_loss(logits: torch.Tensor, support: torch.Tensor, n: int, h: int) -> torch.Tensor:
    pos_weight = torch.tensor([(n - h) / max(h, 1)], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, support.float(), pos_weight=pos_weight)


def ternary_value_loss(value_logits: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    target = (s + 1).long()
    return F.cross_entropy(value_logits.reshape(-1, 3), target.reshape(-1))


def integer_value_loss(integer_logits: torch.Tensor, s: torch.Tensor, integer_values: tuple[int, ...]) -> torch.Tensor:
    target = torch.zeros_like(s, dtype=torch.long)
    for cls, value in enumerate(integer_values, start=1):
        target = torch.where(s == value, torch.full_like(target, cls), target)
    return F.cross_entropy(integer_logits.reshape(-1, len(integer_values) + 1), target.reshape(-1))


def circular_residual_auxiliary(A: torch.Tensor, b: torch.Tensor, support_probs: torch.Tensor, q: int) -> torch.Tensor:
    s_soft = support_probs.float()
    residual = torch.remainder(b.float() - torch.einsum("bmn,bn->bm", A.float(), s_soft), q)
    return circular_distance_loss(residual, q)

