from __future__ import annotations

import torch
import torch.nn as nn


class ColumnTransformerLWE(nn.Module):
    """Column-aware transformer over row-pooled embedding features."""

    def __init__(
        self,
        in_channels: int,
        n: int,
        d_model: int = 128,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        output_n: int | None = None,
    ):
        super().__init__()
        self.output_n = output_n or n
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.col_pos = nn.Parameter(torch.randn(1, n + 1, d_model) * 0.02)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = z.mean(dim=2).transpose(1, 2)
        z = z + self.col_pos[:, : z.shape[1], :]
        z = self.encoder(z)
        logits = self.head(z).squeeze(-1)
        return logits[:, : self.output_n]

