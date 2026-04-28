from __future__ import annotations

import torch

from src.lwe.modular import normalize_centered


def gram_interaction(A: torch.Tensor, b: torch.Tensor, q: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build column interaction G=A^T A and c=A^T b views."""
    G = torch.remainder(torch.einsum("bmn,bmk->bnk", A, A), q)
    c = torch.remainder(torch.einsum("bmn,bm->bn", A, b), q)
    G_center = normalize_centered(G, q)
    c_center = normalize_centered(c, q)
    G_img = torch.stack([G_center, torch.abs(G_center)], dim=1)
    c_vec = torch.stack([c_center, torch.abs(c_center)], dim=1)
    return G_img, c_vec

