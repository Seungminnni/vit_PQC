from __future__ import annotations

import torch
import torch.nn as nn


class TokenFFN(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.permute(0, 2, 3, 1)
        y = y + self.ffn(self.norm(y))
        return y.permute(0, 3, 1, 2)


class AxisAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        y = self.norm(tokens)
        y, _ = self.attn(y, y, y, need_weights=False)
        return tokens + y


class AxialLWEBlock(nn.Module):
    """Separated row/column attention over [B,D,M,n]."""

    def __init__(self, d_model: int, heads: int, dropout: float = 0.1, mode: str = "row_column"):
        super().__init__()
        self.mode = mode
        self.row_attn = AxisAttention(d_model, heads, dropout)
        self.col_attn = AxisAttention(d_model, heads, dropout)
        self.ffn = TokenFFN(d_model, dropout)

    def _row_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, D, M, n = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B * M, n, D)
        tokens = self.row_attn(tokens)
        return tokens.reshape(B, M, n, D).permute(0, 3, 1, 2)

    def _column_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, D, M, n = x.shape
        tokens = x.permute(0, 3, 2, 1).reshape(B * n, M, D)
        tokens = self.col_attn(tokens)
        return tokens.reshape(B, n, M, D).permute(0, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in ("row_only", "row_column"):
            x = self._row_attention(x)
        if self.mode in ("column_only", "row_column"):
            x = self._column_attention(x)
        return self.ffn(x)

