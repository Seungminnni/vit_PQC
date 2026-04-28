from __future__ import annotations

import torch
import torch.nn as nn


class SupportHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class TernaryValueHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Linear(d_model, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def support_probs_from_ternary(logits_value: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits_value, dim=-1)
    return 1.0 - probs[..., 1]

