from __future__ import annotations

import torch


def mod_q(x: torch.Tensor, q: int) -> torch.Tensor:
    """Reduce tensor entries modulo q."""
    return torch.remainder(x, q)


def center_mod(x: torch.Tensor, q: int) -> torch.Tensor:
    """Convert values modulo q into centered representatives."""
    return torch.remainder(x + q // 2, q) - q // 2


def normalize_centered(x: torch.Tensor, q: int) -> torch.Tensor:
    """Normalize centered modular values into roughly [-1, 1]."""
    return center_mod(x, q).float() / (q / 2)

