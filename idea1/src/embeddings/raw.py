from __future__ import annotations

import torch

from src.lwe.modular import normalize_centered


def raw_image(A: torch.Tensor, b: torch.Tensor, q: int) -> torch.Tensor:
    """Build [A|b] centered heatmap with shape [B,1,M,N+1]."""
    x = torch.cat([A, b.unsqueeze(-1)], dim=-1)
    return normalize_centered(x, q).unsqueeze(1)

