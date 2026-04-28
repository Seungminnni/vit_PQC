from __future__ import annotations

import torch
import torch.nn.functional as F

from modular import circular_distance_loss, lwe_batch_dot


def binary_support_bce(
    logits: torch.Tensor,
    support: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    if pos_weight <= 0:
        raise ValueError(f"pos_weight must be positive, got {pos_weight}")
    if pos_weight == 1.0:
        return F.binary_cross_entropy_with_logits(logits, support.float())
    weight = torch.tensor([float(pos_weight)], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, support.float(), pos_weight=weight)


def circular_residual_auxiliary(A: torch.Tensor, b: torch.Tensor, support_probs: torch.Tensor, q: int) -> torch.Tensor:
    s_soft = support_probs.float()
    residual = torch.remainder(b.float() - lwe_batch_dot(A.float(), s_soft), q)
    return circular_distance_loss(residual, q)
