from __future__ import annotations

import itertools

import torch

from src.attack.verify import residual_score_binary


@torch.no_grad()
def recover_binary_from_topk(
    A: torch.Tensor,
    b: torch.Tensor,
    probs: torch.Tensor,
    h: int,
    k: int,
    q: int,
) -> tuple[tuple[int, ...], float]:
    if k < h:
        raise ValueError(f"k must be >= h, got k={k}, h={h}")
    topk = probs.topk(k).indices.tolist()
    best_score = float("inf")
    best_support: tuple[int, ...] | None = None
    for support in itertools.combinations(topk, h):
        score = residual_score_binary(A, b, support, q)
        if score < best_score:
            best_score = score
            best_support = tuple(int(x) for x in support)
    if best_support is None:
        raise RuntimeError("No support candidate was evaluated")
    return best_support, best_score


@torch.no_grad()
def recover_batch_exact_rate(
    A: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    logits: torch.Tensor,
    h: int,
    k: int,
    q: int,
) -> float:
    probs = torch.sigmoid(logits)
    exact = 0
    for row in range(A.shape[0]):
        support, _ = recover_binary_from_topk(A[row], b[row], probs[row], h=h, k=k, q=q)
        true_support = set(torch.where(s[row] != 0)[0].tolist())
        exact += int(set(support) == true_support)
    return exact / A.shape[0]

