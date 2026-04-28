from __future__ import annotations

import torch
import torch.nn as nn


class ChannelLayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cl = x.permute(0, 2, 3, 1)
        x_cl = self.norm(x_cl)
        return x_cl.permute(0, 3, 1, 2)


class LocalRelationEmbedding(nn.Module):
    """Per-pixel channel vector embedding via 1x1 convolutions."""

    def __init__(self, in_channels: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1),
            nn.GELU(),
            ChannelLayerNorm2d(d_model),
            nn.Dropout(dropout),
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

