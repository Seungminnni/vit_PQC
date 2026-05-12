from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .lwe import centered_mod
from .tokenization import RectangularPatchTokenizer


@dataclass(frozen=True)
class LWEViTConfig:
    n: int
    q: int
    in_channels: int
    num_secret_classes: int
    patch_rows: int = 4
    patch_cols: int = 4
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_patch_rows: int = 64
    max_patch_cols: int = 64

    def __post_init__(self) -> None:
        if self.n <= 0 or self.q <= 2:
            raise ValueError("n must be positive and q must be greater than 2.")
        if self.in_channels <= 0 or self.num_secret_classes <= 1:
            raise ValueError("in_channels must be positive and num_secret_classes must be greater than 1.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")


@dataclass(frozen=True)
class LWEViTOutput:
    s_logits: torch.Tensor
    residual_score: torch.Tensor


@dataclass(frozen=True)
class PairTokenLWEConfig:
    n: int
    m: int
    q: int
    num_secret_classes: int
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_m: int = 4096
    max_n: int = 1024

    def __post_init__(self) -> None:
        if self.n <= 0 or self.m <= 0 or self.q <= 2:
            raise ValueError("n and m must be positive and q must be greater than 2.")
        if self.num_secret_classes <= 1:
            raise ValueError("num_secret_classes must be greater than 1.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if self.m > self.max_m or self.n > self.max_n:
            raise ValueError("m/n exceed configured max_m/max_n.")


class PairTokenLWETransformer(nn.Module):
    """Full row-column LWE Transformer.

    One token represents one atomic coefficient/RHS pair:
    token(i, j) = phase/centered features of (A_ij, b_i).
    """

    input_kind = "pair"

    def __init__(self, config: PairTokenLWEConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_proj = nn.Linear(6, config.embed_dim)
        self.row_pos = nn.Embedding(config.max_m, config.embed_dim)
        self.col_pos = nn.Embedding(config.max_n, config.embed_dim)
        self.column_queries = nn.Parameter(torch.zeros(1, config.n, config.embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.secret_head = nn.Linear(config.embed_dim, config.num_secret_classes)
        self.residual_head = nn.Linear(config.embed_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.column_queries, std=0.02)

    def _features(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        A = torch.remainder(A.to(torch.long), self.config.q)
        b = torch.remainder(b.to(torch.long), self.config.q)
        b_grid = b.unsqueeze(-1).expand(-1, -1, A.shape[-1])

        scale = float(self.config.q) / 2.0
        A_centered = centered_mod(A, self.config.q).to(torch.float32) / scale
        b_centered = centered_mod(b_grid, self.config.q).to(torch.float32) / scale
        A_angle = A.to(torch.float32) * (2.0 * math.pi / float(self.config.q))
        b_angle = b_grid.to(torch.float32) * (2.0 * math.pi / float(self.config.q))
        return torch.stack(
            [
                A_centered,
                torch.sin(A_angle),
                torch.cos(A_angle),
                b_centered,
                torch.sin(b_angle),
                torch.cos(b_angle),
            ],
            dim=-1,
        )

    def forward(self, A: torch.Tensor, b: torch.Tensor) -> LWEViTOutput:
        if A.dim() == 2:
            A = A.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        if A.dim() != 3 or b.dim() != 2:
            raise ValueError("A must be (B,m,n), b must be (B,m).")
        batch, m, n = A.shape
        if m != self.config.m or n != self.config.n:
            raise ValueError(f"Expected A shape (*,{self.config.m},{self.config.n}), got {tuple(A.shape)}.")
        if b.shape != (batch, m):
            raise ValueError(f"Expected b shape ({batch},{m}), got {tuple(b.shape)}.")

        features = self._features(A, b)
        tokens = self.feature_proj(features).reshape(batch, m * n, self.config.embed_dim)

        row_ids = torch.arange(m, device=A.device).repeat_interleave(n)
        col_ids = torch.arange(n, device=A.device).repeat(m)
        tokens = tokens + self.row_pos(row_ids).unsqueeze(0) + self.col_pos(col_ids).unsqueeze(0)

        query_ids = torch.arange(n, device=A.device)
        queries = self.column_queries.expand(batch, -1, -1) + self.col_pos(query_ids).unsqueeze(0)
        encoded = self.encoder(torch.cat([tokens, queries], dim=1))

        pair_out = self.norm(encoded[:, : m * n])
        query_out = self.norm(encoded[:, m * n :])
        s_logits = self.secret_head(query_out)
        residual_score = self.residual_head(pair_out.mean(dim=1)).squeeze(-1)
        return LWEViTOutput(s_logits=s_logits, residual_score=residual_score)


class LWEViTForSecret(nn.Module):
    input_kind = "image"

    def __init__(self, config: LWEViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_tokenizer = RectangularPatchTokenizer(
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            patch_rows=config.patch_rows,
            patch_cols=config.patch_cols,
        )
        self.row_pos = nn.Embedding(config.max_patch_rows, config.embed_dim)
        self.col_pos = nn.Embedding(config.max_patch_cols, config.embed_dim)
        self.column_queries = nn.Parameter(torch.zeros(1, config.n, config.embed_dim))
        self.column_pos = nn.Embedding(config.n, config.embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.secret_head = nn.Linear(config.embed_dim, config.num_secret_classes)
        self.residual_head = nn.Linear(config.embed_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.column_queries, std=0.02)

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None = None) -> LWEViTOutput:
        patches = self.patch_tokenizer(image, mask)
        tokens = patches.tokens
        batch = tokens.shape[0]
        grid_h, grid_w = patches.grid_size
        if grid_h > self.config.max_patch_rows or grid_w > self.config.max_patch_cols:
            raise ValueError("Patch grid exceeds configured max_patch_rows/max_patch_cols.")

        row_ids = torch.arange(grid_h, device=image.device).repeat_interleave(grid_w)
        col_ids = torch.arange(grid_w, device=image.device).repeat(grid_h)
        tokens = tokens + self.row_pos(row_ids).unsqueeze(0) + self.col_pos(col_ids).unsqueeze(0)

        query_ids = torch.arange(self.config.n, device=image.device)
        queries = self.column_queries.expand(batch, -1, -1) + self.column_pos(query_ids).unsqueeze(0)
        x = torch.cat([tokens, queries], dim=1)

        query_mask = torch.zeros((batch, self.config.n), dtype=torch.bool, device=image.device)
        key_padding_mask = torch.cat([~patches.mask, query_mask], dim=1)
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)

        patch_out = self.norm(encoded[:, : tokens.shape[1]])
        query_out = self.norm(encoded[:, tokens.shape[1] :])
        s_logits = self.secret_head(query_out)

        valid = patches.mask.unsqueeze(-1).to(patch_out.dtype)
        pooled = (patch_out * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        residual_score = self.residual_head(pooled).squeeze(-1)
        return LWEViTOutput(s_logits=s_logits, residual_score=residual_score)
