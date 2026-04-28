from __future__ import annotations

import itertools

import torch

from candidate import enumerate_binary_supports
from modular import centered_energy


def pair_residual_score(
    A: torch.Tensor,
    b: torch.Tensor,
    j: int,
    k: int,
    q: int,
    score_type: str = "squared",
    sigma: float | None = None,
) -> float:
    residual = torch.remainder(b - A[:, j] - A[:, k], q)
    return float(centered_energy(residual.unsqueeze(0), q, score_type=score_type, sigma=sigma)[0].item())


def ranked_topk_pairs(
    A: torch.Tensor,
    b: torch.Tensor,
    topk: list[int],
    q: int,
    score_type: str = "squared",
    sigma: float | None = None,
) -> list[tuple[tuple[int, int], float]]:
    pairs = []
    for j, k in itertools.combinations(topk, 2):
        pairs.append(((int(j), int(k)), pair_residual_score(A, b, int(j), int(k), q, score_type, sigma)))
    pairs.sort(key=lambda item: item[1])
    return pairs


def pair_filtered_supports(
    A: torch.Tensor,
    b: torch.Tensor,
    topk: list[int],
    h: int,
    q: int,
    pair_budget: int,
    score_type: str = "squared",
    sigma: float | None = None,
) -> list[tuple[int, ...]]:
    """Keep candidate supports that contain at least one strong top-K pair.

    For h=1 pair filtering is not meaningful, so all top-K singletons are returned.
    For h=2 this is exactly a pair ranking. For h>2 this is a second-stage
    pruning heuristic; final correctness is still decided by the full residual
    verifier.
    """
    all_supports = enumerate_binary_supports(topk, h)
    if h < 2 or not all_supports:
        return all_supports

    ranked_pairs = ranked_topk_pairs(A, b, topk, q, score_type=score_type, sigma=sigma)
    keep_pairs = {tuple(sorted(pair)) for pair, _ in ranked_pairs[: max(1, pair_budget)]}
    filtered = []
    for support in all_supports:
        support_pairs = {tuple(sorted(pair)) for pair in itertools.combinations(support, 2)}
        if support_pairs & keep_pairs:
            filtered.append(support)
    return filtered or all_supports


def mean_pair_score_for_support(
    ranked_pair_scores: dict[tuple[int, int], float],
    support: tuple[int, ...],
) -> float:
    pairs = [tuple(sorted(pair)) for pair in itertools.combinations(support, 2)]
    if not pairs:
        return 0.0
    values = [ranked_pair_scores[pair] for pair in pairs if pair in ranked_pair_scores]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def pair_score_lookup(
    A: torch.Tensor,
    b: torch.Tensor,
    topk: list[int],
    q: int,
    score_type: str = "squared",
    sigma: float | None = None,
) -> dict[tuple[int, int], float]:
    return {tuple(sorted(pair)): score for pair, score in ranked_topk_pairs(A, b, topk, q, score_type, sigma)}

