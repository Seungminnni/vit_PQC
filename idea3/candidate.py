from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F


def binary_hfree_candidates_from_logits(
    logits: torch.Tensor,
    uncertain_K: int = 10,
    threshold: float = 0.5,
) -> list[torch.Tensor]:
    """Generate binary candidates without top-h or combinations(topK, h).

    The base secret is formed by thresholding Pr(s_j=1). The search then flips
    only the coordinates closest to the threshold, so each candidate can have a
    different Hamming weight.
    """
    probs = torch.sigmoid(logits)
    base_s = (probs >= threshold).long()
    K_eff = min(max(int(uncertain_K), 0), logits.numel())
    if K_eff == 0:
        return [base_s]

    uncertainty = (probs - threshold).abs()
    idxs = uncertainty.topk(K_eff, largest=False).indices.tolist()
    candidates: list[torch.Tensor] = []
    seen: set[tuple[int, ...]] = set()
    for bits in itertools.product((0, 1), repeat=len(idxs)):
        s_hat = base_s.clone()
        for idx, bit in zip(idxs, bits):
            s_hat[idx] = bit
        key = tuple(int(x) for x in s_hat.tolist())
        if key not in seen:
            seen.add(key)
            candidates.append(s_hat)
    return candidates


def binary_posterior_nll(logits: torch.Tensor, s_hat: torch.Tensor) -> float:
    target = s_hat.to(device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
    return float(loss.item())
