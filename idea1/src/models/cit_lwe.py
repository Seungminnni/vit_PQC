from __future__ import annotations

import torch
import torch.nn as nn


class ConvColumnBranch(nn.Module):
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, output_n: int) -> torch.Tensor:
        z = self.net(x).mean(dim=2).transpose(1, 2)
        return z[:, :output_n, :]


class CITLEWEMultiBranch(nn.Module):
    """Minimal multi-branch CIT-LWE skeleton for RHIE plus optional Gram."""

    def __init__(
        self,
        rhie_channels: int,
        n: int,
        d_model: int = 128,
        gram_channels: int | None = None,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.rhie_branch = ConvColumnBranch(rhie_channels, d_model)
        self.gram_branch = ConvColumnBranch(gram_channels, d_model) if gram_channels else None
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
        self.col_pos = nn.Parameter(torch.randn(1, n, d_model) * 0.02)
        self.head = nn.Linear(d_model, 1)

    def forward(self, packet: dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.rhie_branch(packet["rhie"], self.n)
        if self.gram_branch is not None and "gram" in packet:
            z = z + self.gram_branch(packet["gram"], self.n)
        z = self.encoder(z + self.col_pos[:, : z.shape[1], :])
        return self.head(z).squeeze(-1)

