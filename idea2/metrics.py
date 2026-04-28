from __future__ import annotations

import math

import torch

from candidate import enumerate_integer_candidates_from_logits, enumerate_ternary_candidates, reduction_factor, topk_indices_from_logits
from distinguisher import choose_best_candidate, residual_gap_for_true
from models.decoder import support_logits_from_output
from recovery import recover_binary_trace


@torch.no_grad()
def coord_acc_from_logits(logits: torch.Tensor, support: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return float((pred == support.float()).float().mean().item())


@torch.no_grad()
def top_h_support_mask(logits: torch.Tensor, h: int) -> torch.Tensor:
    idx = torch.sigmoid(logits).topk(h, dim=-1).indices
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


@torch.no_grad()
def pre_rerank_full_match(logits: torch.Tensor, support: torch.Tensor, h: int) -> float:
    pred_mask = top_h_support_mask(logits, h)
    true_mask = support.bool()
    return float((pred_mask == true_mask).all(dim=1).float().mean().item())


@torch.no_grad()
def candidate_hit_rate(logits: torch.Tensor, support: torch.Tensor, h: int, K: int) -> float:
    topk = topk_indices_from_logits(logits, K)
    true_idx = support.topk(h, dim=1).indices
    hit = (topk.unsqueeze(-1) == true_idx.unsqueeze(1)).any(dim=1).all(dim=1)
    return float(hit.float().mean().item())


@torch.no_grad()
def evaluate_binary_candidates(
    batch,
    output: dict[str, torch.Tensor],
    h: int,
    K: int,
    score_type: str = "squared",
    use_pair_filter: bool = False,
    pair_budget: int = 64,
    pair_score_weight: float = 0.0,
    posterior_weight: float = 0.0,
) -> dict[str, float]:
    logits = support_logits_from_output(output, "binary")
    support = batch.support
    direct_exact = 0
    hit = 0
    rerank_exact = 0
    rerank_given_hit = 0
    gaps = []
    candidate_counts = []
    reduction_factors = []
    for row in range(batch.A.shape[0]):
        trace = recover_binary_trace(
            batch.A[row],
            batch.b[row],
            batch.s[row],
            logits[row],
            h=h,
            K=K,
            q=batch.q,
            score_type=score_type,
            use_pair_filter=use_pair_filter,
            pair_budget=pair_budget,
            pair_score_weight=pair_score_weight,
            posterior_weight=posterior_weight,
        )
        direct_exact += int(trace.direct_exact)
        hit += int(trace.candidate_hit)
        rerank_exact += int(trace.rerank_exact)
        rerank_given_hit += int(trace.rerank_exact and trace.candidate_hit)
        if trace.candidate_hit:
            gaps.append(trace.residual_gap)
        candidate_counts.append(trace.candidate_count)
        reduction_factors.append(trace.reduction_factor)
    B = batch.A.shape[0]
    hit_rate = hit / B
    return {
        "coord_acc": coord_acc_from_logits(logits, support),
        "pre_rerank_full_match": direct_exact / B,
        "candidate_hit_rate": hit_rate,
        "post_rerank_full_match": rerank_exact / B,
        "rerank_success_given_hit": rerank_given_hit / max(hit, 1),
        "residual_gap": float(sum(gaps) / max(len(gaps), 1)),
        "candidate_count": float(sum(candidate_counts) / max(len(candidate_counts), 1)),
        "reduction_factor": float(sum(reduction_factors) / max(len(reduction_factors), 1)),
        "pair_filter_enabled": float(use_pair_filter),
    }


@torch.no_grad()
def evaluate_ternary_candidates(batch, output: dict[str, torch.Tensor], h: int, K: int, score_type: str = "squared") -> dict[str, float]:
    logits = support_logits_from_output(output, "ternary")
    value_pred = output["value_logits"].argmax(dim=-1) - 1
    support = batch.support
    topk = topk_indices_from_logits(logits, K)
    exact = 0
    hit = 0
    nz_mask = batch.s != 0
    sign_acc = (value_pred[nz_mask] == batch.s[nz_mask]).float().mean().item() if nz_mask.any() else torch.tensor(1.0).item()
    for row in range(batch.A.shape[0]):
        topk_list = topk[row].tolist()
        true_support = set(torch.where(batch.s[row] != 0)[0].tolist())
        is_hit = true_support.issubset(set(topk_list))
        hit += int(is_hit)
        candidates = enumerate_ternary_candidates(batch.A.shape[-1], topk_list, h, batch.A.device)
        best, _, _ = choose_best_candidate(batch.A[row], batch.b[row], candidates, batch.q, score_type=score_type)
        exact += int(torch.equal(best, batch.s[row]))
    return {
        "coord_acc": coord_acc_from_logits(logits, support),
        "candidate_hit_rate": hit / batch.A.shape[0],
        "post_rerank_full_match": exact / batch.A.shape[0],
        "support_pre_rerank_full_match": pre_rerank_full_match(logits, support, h),
        "value_sign_acc_nonzero": float(sign_acc),
        "reduction_factor": reduction_factor(batch.A.shape[-1], h, K, secret_type="ternary"),
    }


@torch.no_grad()
def evaluate_integer_candidates(
    batch,
    output: dict[str, torch.Tensor],
    h: int,
    K: int,
    value_topr: int,
    integer_values: tuple[int, ...],
    score_type: str = "squared",
) -> dict[str, float]:
    logits = support_logits_from_output(output, "integer")
    support = batch.support
    topk = topk_indices_from_logits(logits, K)
    exact = 0
    hit = 0
    value_pred_cls = output["integer_logits"].argmax(dim=-1)
    value_table = torch.tensor((0, *integer_values), dtype=torch.long, device=batch.A.device)
    value_pred = value_table[value_pred_cls]
    nz_mask = batch.s != 0
    value_acc = (value_pred[nz_mask] == batch.s[nz_mask]).float().mean().item() if nz_mask.any() else torch.tensor(1.0).item()
    for row in range(batch.A.shape[0]):
        true_support = set(torch.where(batch.s[row] != 0)[0].tolist())
        is_hit = true_support.issubset(set(topk[row].tolist()))
        hit += int(is_hit)
        candidates = enumerate_integer_candidates_from_logits(
            logits[row],
            output["integer_logits"][row],
            h=h,
            K=K,
            value_topr=value_topr,
            integer_values=integer_values,
        )
        best, _, _ = choose_best_candidate(batch.A[row], batch.b[row], candidates, batch.q, score_type=score_type)
        exact += int(torch.equal(best, batch.s[row]))
    return {
        "coord_acc": coord_acc_from_logits(logits, support),
        "candidate_hit_rate": hit / batch.A.shape[0],
        "post_rerank_full_match": exact / batch.A.shape[0],
        "support_pre_rerank_full_match": pre_rerank_full_match(logits, support, h),
        "integer_value_acc_nonzero": float(value_acc),
        "reduction_factor": reduction_factor(batch.A.shape[-1], h, K, secret_type="integer", value_width=len(integer_values)),
    }


def random_candidate_hit_probability(n: int, h: int, K: int) -> float:
    if K < h:
        return 0.0
    return math.comb(K, h) / math.comb(n, h)
