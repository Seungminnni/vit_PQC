from __future__ import annotations

import torch
import torch.nn as nn


class SimpleLWEImageCNN(nn.Module):
    """Small CNN that pools rows and emits coordinate-wise support logits."""

    def __init__(self, in_channels: int, hidden_dim: int = 128, output_n: int | None = None):
        super().__init__()
        self.output_n = output_n
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = z.mean(dim=2)
        logits = self.head(z).squeeze(1)
        if self.output_n is not None:
            logits = logits[:, : self.output_n]
        return logits

