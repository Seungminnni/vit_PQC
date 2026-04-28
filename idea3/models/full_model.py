from __future__ import annotations

import torch
import torch.nn as nn

from models.axial_block import AxialLWEBlock
from models.coordinate_transformer import CoordinateTransformer
from models.decoder import CandidateDecoder
from models.local_embedding import LocalRelationEmbedding
from models.pooling import build_pooling


class RHIECGModel(nn.Module):
    """RHIE-CG vision/attention hybrid for coordinate posterior generation."""

    def __init__(
        self,
        in_channels: int,
        n: int,
        d_model: int = 96,
        depth: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        pooling: str = "attention",
        coordinate_transformer: bool = True,
        axial_mode: str = "row_column",
        use_position: bool = False,
        secret_type: str = "binary",
        integer_values: tuple[int, ...] = (-3, -2, -1, 1, 2, 3),
    ):
        super().__init__()
        self.secret_type = secret_type
        self.local = LocalRelationEmbedding(in_channels, d_model, dropout)
        self.axial_blocks = nn.ModuleList(
            [AxialLWEBlock(d_model, heads, dropout, mode=axial_mode) for _ in range(depth)]
        )
        self.pool = build_pooling(pooling, d_model)
        self.coord = (
            CoordinateTransformer(n, d_model, max(1, depth // 2), heads, dropout, use_position)
            if coordinate_transformer
            else nn.Identity()
        )
        self.decoder = CandidateDecoder(d_model, secret_type=secret_type, integer_values=integer_values)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.local(x)
        for block in self.axial_blocks:
            z = block(z)
        tokens = self.pool(z)
        tokens = self.coord(tokens)
        out = self.decoder(tokens)
        out["tokens"] = tokens
        return out

