from __future__ import annotations

import torch
import torch.nn as nn


class CoordinateTransformer(nn.Module):
    """Permutation-aware coordinate interaction transformer by default."""

    def __init__(
        self,
        n: int,
        d_model: int,
        depth: int,
        heads: int,
        dropout: float,
        use_position: bool = False,
    ):
        super().__init__()
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
        self.pos = nn.Parameter(torch.randn(1, n, d_model) * 0.02) if use_position else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos is not None:
            x = x + self.pos[:, : x.shape[1], :]
        return self.encoder(x)

