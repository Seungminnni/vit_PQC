from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .lwe import centered_mod


@dataclass(frozen=True)
class LWEViTOutput:
    s_logits: torch.Tensor
    residual_score: torch.Tensor


def _residue_dim(residue_encoding: str) -> int:
    if residue_encoding in {"raw", "centered"}:
        return 1
    if residue_encoding == "phase":
        return 3
    raise ValueError("residue_encoding must be one of: raw, centered, phase.")


def _encode_residue(values: torch.Tensor, q: int, residue_encoding: str) -> torch.Tensor:
    values = torch.remainder(values.to(torch.long), q)
    if residue_encoding == "raw":
        return (values.to(torch.float32) / float(q - 1)).unsqueeze(-1)

    scale = float(q) / 2.0
    centered = centered_mod(values, q).to(torch.float32) / scale
    if residue_encoding == "centered":
        return centered.unsqueeze(-1)

    if residue_encoding != "phase":
        raise ValueError("residue_encoding must be one of: raw, centered, phase.")
    angle = values.to(torch.float32) * (2.0 * math.pi / float(q))
    return torch.stack([centered, torch.sin(angle), torch.cos(angle)], dim=-1)


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


@dataclass(frozen=True)
class RowBlockLWEConfig:
    n: int
    m: int
    q: int
    num_secret_classes: int
    block_rows: int = 1
    block_cols: int = 16
    residue_encoding: str = "phase"
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_row_blocks: int = 4096
    max_col_blocks: int = 1024
    max_n: int = 1024

    def __post_init__(self) -> None:
        if self.n <= 0 or self.m <= 0 or self.q <= 2:
            raise ValueError("n and m must be positive and q must be greater than 2.")
        if self.num_secret_classes <= 1:
            raise ValueError("num_secret_classes must be greater than 1.")
        if self.block_rows <= 0 or self.block_cols <= 0:
            raise ValueError("block_rows and block_cols must be positive.")
        if self.residue_encoding not in {"raw", "centered", "phase"}:
            raise ValueError("residue_encoding must be one of: raw, centered, phase.")
        if self.m % self.block_rows != 0:
            raise ValueError("m must be divisible by block_rows.")
        if self.n % self.block_cols != 0:
            raise ValueError("n must be divisible by block_cols.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if self.m // self.block_rows > self.max_row_blocks:
            raise ValueError("row block count exceeds max_row_blocks.")
        if self.n // self.block_cols > self.max_col_blocks:
            raise ValueError("column block count exceeds max_col_blocks.")
        if self.n > self.max_n:
            raise ValueError("n exceeds max_n.")


@dataclass(frozen=True)
class EquationTransformerConfig:
    n: int
    m: int
    q: int
    num_secret_classes: int
    residue_encoding: str = "raw"
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
        _residue_dim(self.residue_encoding)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if self.m > self.max_m or self.n > self.max_n:
            raise ValueError("m/n exceed configured max_m/max_n.")


@dataclass(frozen=True)
class RowLocalCNNLWEConfig:
    n: int
    m: int
    q: int
    num_secret_classes: int
    residue_encoding: str = "raw"
    embed_dim: int = 128
    depth: int = 4
    dropout: float = 0.0
    max_n: int = 1024

    def __post_init__(self) -> None:
        if self.n <= 0 or self.m <= 0 or self.q <= 2:
            raise ValueError("n and m must be positive and q must be greater than 2.")
        if self.num_secret_classes <= 1:
            raise ValueError("num_secret_classes must be greater than 1.")
        _residue_dim(self.residue_encoding)
        if self.embed_dim <= 0 or self.depth <= 0:
            raise ValueError("embed_dim and depth must be positive.")
        if self.n > self.max_n:
            raise ValueError("n exceeds max_n.")


class RowBlockLWETransformer(nn.Module):
    """Transformer over LWE row/column blocks.

    The tokenization follows the report shape:
    pixel(i, j) = [ENC(A_ij), ENC(b_i)]. ENC can be raw normalized residue,
    centered residue, or centered plus circular phase features.
    """

    input_kind = "row_block"

    def __init__(self, config: RowBlockLWEConfig) -> None:
        super().__init__()
        self.config = config
        self.residue_dim = self._residue_dim(config)
        self.pixel_dim = 2 * self.residue_dim
        self.token_feature_dim = config.block_rows * config.block_cols * self.pixel_dim
        self.feature_proj = nn.Linear(self.token_feature_dim, config.embed_dim)
        self.row_block_pos = nn.Embedding(config.max_row_blocks, config.embed_dim)
        self.col_block_pos = nn.Embedding(config.max_col_blocks, config.embed_dim)
        self.secret_col_pos = nn.Embedding(config.max_n, config.embed_dim)
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

    @staticmethod
    def _residue_dim(config: RowBlockLWEConfig) -> int:
        return _residue_dim(config.residue_encoding)

    def _encode_residue(self, values: torch.Tensor) -> torch.Tensor:
        return _encode_residue(values, q=self.config.q, residue_encoding=self.config.residue_encoding)

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

        A_enc = self._encode_residue(A)
        b_enc = self._encode_residue(b).unsqueeze(2).expand(-1, -1, n, -1)
        pixels = torch.cat([A_enc, b_enc], dim=-1)

        block_rows = self.config.block_rows
        block_cols = self.config.block_cols
        row_blocks = m // block_rows
        col_blocks = n // block_cols
        block_features = (
            pixels.reshape(batch, row_blocks, block_rows, col_blocks, block_cols, self.pixel_dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, row_blocks * col_blocks, self.token_feature_dim)
        )
        tokens = self.feature_proj(block_features)

        row_ids = torch.arange(row_blocks, device=A.device).repeat_interleave(col_blocks)
        col_ids = torch.arange(col_blocks, device=A.device).repeat(row_blocks)
        tokens = tokens + self.row_block_pos(row_ids).unsqueeze(0) + self.col_block_pos(col_ids).unsqueeze(0)

        query_ids = torch.arange(n, device=A.device)
        queries = self.column_queries.expand(batch, -1, -1) + self.secret_col_pos(query_ids).unsqueeze(0)
        encoded = self.encoder(torch.cat([tokens, queries], dim=1))

        block_out = self.norm(encoded[:, : row_blocks * col_blocks])
        query_out = self.norm(encoded[:, row_blocks * col_blocks :])
        s_logits = self.secret_head(query_out)
        residual_score = self.residual_head(block_out.mean(dim=1)).squeeze(-1)
        return LWEViTOutput(s_logits=s_logits, residual_score=residual_score)


class EquationLWETransformer(nn.Module):
    """Plain Transformer over complete LWE equations.

    One token is one public equation row:
    token_i = [ENC(A_i1), ..., ENC(A_in), ENC(b_i)].
    Unlike row_block bc16, b_i is appended once instead of being paired with
    every coefficient. This keeps the sequence length identical to bc16 while
    testing a more conventional row-token Transformer baseline.
    """

    input_kind = "row_block"

    def __init__(self, config: EquationTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.residue_dim = _residue_dim(config.residue_encoding)
        self.token_feature_dim = (config.n + 1) * self.residue_dim
        self.feature_proj = nn.Linear(self.token_feature_dim, config.embed_dim)
        self.row_pos = nn.Embedding(config.max_m, config.embed_dim)
        self.secret_col_pos = nn.Embedding(config.max_n, config.embed_dim)
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

    def _encode_residue(self, values: torch.Tensor) -> torch.Tensor:
        return _encode_residue(values, q=self.config.q, residue_encoding=self.config.residue_encoding)

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

        A_enc = self._encode_residue(A).reshape(batch, m, n * self.residue_dim)
        b_enc = self._encode_residue(b)
        equation_features = torch.cat([A_enc, b_enc], dim=-1)
        tokens = self.feature_proj(equation_features)

        row_ids = torch.arange(m, device=A.device)
        tokens = tokens + self.row_pos(row_ids).unsqueeze(0)

        query_ids = torch.arange(n, device=A.device)
        queries = self.column_queries.expand(batch, -1, -1) + self.secret_col_pos(query_ids).unsqueeze(0)
        encoded = self.encoder(torch.cat([tokens, queries], dim=1))

        equation_out = self.norm(encoded[:, :m])
        query_out = self.norm(encoded[:, m:])
        s_logits = self.secret_head(query_out)
        residual_score = self.residual_head(equation_out.mean(dim=1)).squeeze(-1)
        return LWEViTOutput(s_logits=s_logits, residual_score=residual_score)


class CNNResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = _norm_groups(channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class RowLocalCNNLWEModel(nn.Module):
    """Row-local CNN baseline over the same relation rows used by row_block.

    The row pixel is [ENC(A_ij), ENC(b_i)]. The 1D CNN processes one equation
    row at a time across the secret-coordinate axis, then pools over rows to
    produce one classifier input per secret coordinate. This avoids giving the
    CNN an artificial notion that adjacent LWE equations are image neighbors.
    """

    input_kind = "row_block"

    def __init__(self, config: RowLocalCNNLWEConfig) -> None:
        super().__init__()
        self.config = config
        self.residue_dim = _residue_dim(config.residue_encoding)
        self.in_channels = 2 * self.residue_dim
        groups = _norm_groups(config.embed_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, config.embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, config.embed_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[CNNResidualBlock(config.embed_dim, config.dropout) for _ in range(config.depth)]
        )
        self.column_pos = nn.Embedding(config.max_n, config.embed_dim)
        self.column_norm = nn.LayerNorm(config.embed_dim)
        self.secret_head = nn.Linear(config.embed_dim, config.num_secret_classes)
        self.residual_head = nn.Linear(config.embed_dim, 1)

    def _encode_residue(self, values: torch.Tensor) -> torch.Tensor:
        return _encode_residue(values, q=self.config.q, residue_encoding=self.config.residue_encoding)

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

        A_enc = self._encode_residue(A)
        b_enc = self._encode_residue(b).unsqueeze(2).expand(-1, -1, n, -1)
        row_pixels = torch.cat([A_enc, b_enc], dim=-1)
        row_inputs = row_pixels.reshape(batch * m, n, self.in_channels).permute(0, 2, 1).contiguous()

        row_features = self.blocks(self.stem(row_inputs))
        features = row_features.reshape(batch, m, self.config.embed_dim, n)
        column_features = features.mean(dim=1).permute(0, 2, 1)
        col_ids = torch.arange(n, device=A.device)
        column_features = self.column_norm(column_features + self.column_pos(col_ids).unsqueeze(0))
        s_logits = self.secret_head(column_features)

        pooled = features.mean(dim=(1, 3))
        residual_score = self.residual_head(pooled).squeeze(-1)
        return LWEViTOutput(s_logits=s_logits, residual_score=residual_score)
