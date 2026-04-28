from __future__ import annotations

import torch

from candidate import binary_hfree_candidates_from_logits, binary_posterior_nll
from distinguisher import residual_score
from models.decoder import support_logits_from_output


@torch.no_grad()
def evaluate_binary_hfree_candidates(
    batch,
    output: dict[str, torch.Tensor],
    uncertain_K: int,
    threshold: float = 0.5,
    score_type: str = "squared",
    posterior_weight: float = 0.0,
) -> dict[str, float]:
    logits = support_logits_from_output(output, "binary")
    probs = torch.sigmoid(logits)
    direct_s = (probs >= threshold).long()
    true_s = batch.s.long()

    tp = ((direct_s == 1) & (true_s == 1)).sum().float()
    fp = ((direct_s == 1) & (true_s == 0)).sum().float()
    fn = ((direct_s == 0) & (true_s == 1)).sum().float()
    precision = float((tp / (tp + fp).clamp_min(1)).item())
    recall = float((tp / (tp + fn).clamp_min(1)).item())
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)

    pred_h = direct_s.sum(dim=1).float()
    true_h = true_s.sum(dim=1).float()
    post_exact = 0
    contains_true = 0
    candidate_counts: list[int] = []
    best_scores: list[float] = []
    true_scores: list[float] = []
    gaps: list[float] = []

    for row in range(batch.A.shape[0]):
        candidates = binary_hfree_candidates_from_logits(logits[row], uncertain_K=uncertain_K, threshold=threshold)
        best_s = candidates[0]
        best_score = float("inf")
        true_score = residual_score(batch.A[row], batch.b[row], true_s[row], batch.q, score_type=score_type)
        wrong_scores = []
        row_contains_true = False

        for candidate in candidates:
            residual = residual_score(batch.A[row], batch.b[row], candidate, batch.q, score_type=score_type)
            score = residual
            if posterior_weight:
                score += posterior_weight * binary_posterior_nll(logits[row], candidate)
            if torch.equal(candidate.to(true_s.device), true_s[row]):
                row_contains_true = True
            else:
                wrong_scores.append(residual)
            if score < best_score:
                best_s = candidate
                best_score = score

        post_exact += int(torch.equal(best_s.to(true_s.device), true_s[row]))
        contains_true += int(row_contains_true)
        candidate_counts.append(len(candidates))
        best_scores.append(best_score)
        true_scores.append(true_score)
        if wrong_scores:
            gaps.append(min(wrong_scores) - true_score)

    B = batch.A.shape[0]
    return {
        "coord_acc": float((direct_s == true_s).float().mean().item()),
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
        "direct_full_match": float((direct_s == true_s).all(dim=1).float().mean().item()),
        "h_abs_error": float((pred_h - true_h).abs().mean().item()),
        "pred_h_mean": float(pred_h.mean().item()),
        "true_h_mean": float(true_h.mean().item()),
        "candidate_contains_true": contains_true / B,
        "post_verifier_full_match": post_exact / B,
        "candidate_count": float(sum(candidate_counts) / max(len(candidate_counts), 1)),
        "best_score": float(sum(best_scores) / max(len(best_scores), 1)),
        "true_residual_score": float(sum(true_scores) / max(len(true_scores), 1)),
        "residual_gap": float(sum(gaps) / max(len(gaps), 1)),
        "h_free_enabled": 1.0,
    }
