from __future__ import annotations

import torch
import torch.nn as nn


class AttentionColumnPooling(nn.Module):
    """Adaptive pooling over M equations for each coordinate."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=2)
        pooled = (x * weights).sum(dim=2)
        return pooled.transpose(1, 2)


class MeanColumnPooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2).transpose(1, 2)


class MeanMaxColumnPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=2).transpose(1, 2)
        maxv = x.max(dim=2).values.transpose(1, 2)
        return self.proj(torch.cat([mean, maxv], dim=-1))


def build_pooling(pooling: str, d_model: int) -> nn.Module:
    if pooling == "attention":
        return AttentionColumnPooling(d_model)
    if pooling == "mean":
        return MeanColumnPooling()
    if pooling == "meanmax":
        return MeanMaxColumnPooling(d_model)
    raise ValueError(f"unsupported pooling={pooling}")

