from __future__ import annotations

from dataclasses import dataclass

import torch

from candidate import binary_secret_from_support, enumerate_binary_supports, topk_indices_from_logits
from distinguisher import residual_score, residual_summary
from pair_residual import mean_pair_score_for_support, pair_filtered_supports, pair_score_lookup


@dataclass
class CandidateTrace:
    support: tuple[int, ...]
    residual_score: float
    posterior_score: float
    pair_score: float
    total_score: float
    is_true: bool


@dataclass
class RecoveryTrace:
    topk: tuple[int, ...]
    direct_support: tuple[int, ...]
    best_support: tuple[int, ...]
    true_support: tuple[int, ...]
    candidate_hit: bool
    direct_exact: bool
    rerank_exact: bool
    candidate_count: int
    reduction_factor: float
    residual_gap: float
    best_residual_summary: dict[str, float]
    true_residual_summary: dict[str, float]
    candidates: list[CandidateTrace]


def _posterior_score(log_probs: torch.Tensor, support: tuple[int, ...]) -> float:
    if not support:
        return 0.0
    return float(-log_probs[list(support)].sum().item())


def recover_binary_trace(
    A: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    logits: torch.Tensor,
    h: int,
    K: int,
    q: int,
    score_type: str = "squared",
    sigma: float | None = None,
    use_pair_filter: bool = False,
    pair_budget: int = 64,
    pair_score_weight: float = 0.0,
    posterior_weight: float = 0.0,
) -> RecoveryTrace:
    probs = torch.sigmoid(logits)
    log_probs = torch.log(probs.clamp_min(1e-8))
    topk = tuple(int(x) for x in topk_indices_from_logits(logits.unsqueeze(0), K)[0].tolist())
    direct_support = tuple(sorted(int(x) for x in probs.topk(h).indices.tolist()))
    true_support = tuple(sorted(int(x) for x in torch.where(s != 0)[0].tolist()))

    if use_pair_filter:
        supports = pair_filtered_supports(A, b, list(topk), h, q, pair_budget, score_type=score_type, sigma=sigma)
    else:
        supports = enumerate_binary_supports(list(topk), h)

    pair_lookup = pair_score_lookup(A, b, list(topk), q, score_type=score_type, sigma=sigma) if pair_score_weight else {}
    traces: list[CandidateTrace] = []
    for support in supports:
        support_sorted = tuple(sorted(int(x) for x in support))
        s_hat = binary_secret_from_support(A.shape[1], support_sorted, A.device)
        r_score = residual_score(A, b, s_hat, q, score_type=score_type, sigma=sigma)
        p_score = _posterior_score(log_probs, support_sorted)
        pair_score = mean_pair_score_for_support(pair_lookup, support_sorted)
        total = r_score + posterior_weight * p_score + pair_score_weight * pair_score
        traces.append(
            CandidateTrace(
                support=support_sorted,
                residual_score=r_score,
                posterior_score=p_score,
                pair_score=pair_score,
                total_score=total,
                is_true=support_sorted == true_support,
            )
        )
    traces.sort(key=lambda item: item.total_score)
    if not traces:
        raise RuntimeError("no binary candidates generated")
    best = traces[0]
    true_trace = next((trace for trace in traces if trace.is_true), None)
    best_wrong = next((trace for trace in traces if not trace.is_true), None)
    if true_trace is None:
        gap = 0.0
    elif best_wrong is None:
        gap = float("inf")
    else:
        gap = best_wrong.residual_score - true_trace.residual_score

    best_s = binary_secret_from_support(A.shape[1], best.support, A.device)
    true_s = binary_secret_from_support(A.shape[1], true_support, A.device)
    full_space = 1
    for i in range(h):
        full_space = full_space * (A.shape[1] - i) // (i + 1)
    return RecoveryTrace(
        topk=topk,
        direct_support=direct_support,
        best_support=best.support,
        true_support=true_support,
        candidate_hit=set(true_support).issubset(set(topk)),
        direct_exact=direct_support == true_support,
        rerank_exact=best.support == true_support,
        candidate_count=len(traces),
        reduction_factor=float(full_space / max(len(traces), 1)),
        residual_gap=gap,
        best_residual_summary=residual_summary(A, b, best_s, q),
        true_residual_summary=residual_summary(A, b, true_s, q),
        candidates=traces,
    )


def trace_to_row(trace: RecoveryTrace, prefix: str = "") -> dict[str, float | int | str | bool]:
    return {
        f"{prefix}topk": " ".join(str(x) for x in trace.topk),
        f"{prefix}direct_support": " ".join(str(x) for x in trace.direct_support),
        f"{prefix}best_support": " ".join(str(x) for x in trace.best_support),
        f"{prefix}true_support": " ".join(str(x) for x in trace.true_support),
        f"{prefix}candidate_hit": trace.candidate_hit,
        f"{prefix}direct_exact": trace.direct_exact,
        f"{prefix}rerank_exact": trace.rerank_exact,
        f"{prefix}candidate_count": trace.candidate_count,
        f"{prefix}reduction_factor": trace.reduction_factor,
        f"{prefix}residual_gap": trace.residual_gap,
        f"{prefix}best_mean_abs_residual": trace.best_residual_summary["mean_abs"],
        f"{prefix}true_mean_abs_residual": trace.true_residual_summary["mean_abs"],
    }

