from __future__ import annotations

import torch
import torch.nn.functional as F


def support_bce_loss(logits: torch.Tensor, y_support: torch.Tensor, n: int, h: int) -> torch.Tensor:
    pos_weight = torch.tensor([(n - h) / max(h, 1)], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, y_support, pos_weight=pos_weight)


def ternary_value_loss(logits_value: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    target = s + 1
    return F.cross_entropy(logits_value.reshape(-1, 3), target.reshape(-1))

