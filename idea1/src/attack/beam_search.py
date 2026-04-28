from __future__ import annotations

import torch

from src.attack.verify import residual_score_binary


@torch.no_grad()
def greedy_binary_support(A: torch.Tensor, b: torch.Tensor, probs: torch.Tensor, h: int, q: int) -> tuple[int, ...]:
    """Simple fallback candidate search seeded by posterior rank."""
    selected: list[int] = []
    remaining = probs.argsort(descending=True).tolist()
    while len(selected) < h:
        best_idx = None
        best_score = float("inf")
        for idx in remaining[: max(8, h * 2)]:
            if idx in selected:
                continue
            score = residual_score_binary(A, b, selected + [idx], q)
            if score < best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            best_idx = remaining[0]
        selected.append(int(best_idx))
        remaining = [idx for idx in remaining if idx != best_idx]
    return tuple(selected)

