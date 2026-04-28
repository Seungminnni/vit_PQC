from __future__ import annotations

import torch

from modular import centered_energy, centered_int, lwe_dot


def residual_mod_q(A: torch.Tensor, b: torch.Tensor, s_hat: torch.Tensor, q: int) -> torch.Tensor:
    return torch.remainder(b - lwe_dot(A, s_hat.to(A.device)), q)


def residual_score(
    A: torch.Tensor,
    b: torch.Tensor,
    s_hat: torch.Tensor,
    q: int,
    score_type: str = "squared",
    sigma: float | None = None,
) -> float:
    residual = residual_mod_q(A, b, s_hat, q)
    return float(centered_energy(residual.unsqueeze(0), q, score_type=score_type, sigma=sigma)[0].item())


def residual_summary(A: torch.Tensor, b: torch.Tensor, s_hat: torch.Tensor, q: int) -> dict[str, float]:
    r = centered_int(residual_mod_q(A, b, s_hat, q), q).float()
    return {
        "mean_abs": float(r.abs().mean().item()),
        "max_abs": float(r.abs().max().item()),
        "energy": float(r.square().sum().item()),
        "std": float(r.std(unbiased=False).item()),
    }


def choose_best_candidate(
    A: torch.Tensor,
    b: torch.Tensor,
    candidates: list[torch.Tensor],
    q: int,
    score_type: str = "squared",
    sigma: float | None = None,
) -> tuple[torch.Tensor, float, int]:
    if not candidates:
        raise ValueError("candidate list is empty")
    best_idx = -1
    best_score = float("inf")
    for idx, candidate in enumerate(candidates):
        score = residual_score(A, b, candidate, q, score_type=score_type, sigma=sigma)
        if score < best_score:
            best_idx = idx
            best_score = score
    return candidates[best_idx], best_score, best_idx


def residual_gap_for_true(
    A: torch.Tensor,
    b: torch.Tensor,
    true_s: torch.Tensor,
    candidates: list[torch.Tensor],
    q: int,
    score_type: str = "squared",
    sigma: float | None = None,
) -> float:
    true_score = residual_score(A, b, true_s, q, score_type=score_type, sigma=sigma)
    wrong_scores = [
        residual_score(A, b, cand, q, score_type=score_type, sigma=sigma)
        for cand in candidates
        if not torch.equal(cand.to(true_s.device), true_s)
    ]
    if not wrong_scores:
        return float("inf")
    return min(wrong_scores) - true_score
