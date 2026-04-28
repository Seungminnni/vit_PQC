from __future__ import annotations

import itertools

import torch


@torch.no_grad()
def top_h_recall(logits: torch.Tensor, y_support: torch.Tensor, h: int) -> float:
    probs = torch.sigmoid(logits)
    top_idx = probs.topk(h, dim=1).indices
    true_idx = y_support.topk(h, dim=1).indices
    hits = (top_idx.unsqueeze(-1) == true_idx.unsqueeze(1)).any(dim=-1).float().sum(dim=1)
    return float((hits / h).mean().item())


@torch.no_grad()
def exact_support_recovery(logits: torch.Tensor, y_support: torch.Tensor, h: int) -> float:
    probs = torch.sigmoid(logits)
    top_idx = probs.topk(h, dim=1).indices
    true_idx = y_support.topk(h, dim=1).indices
    pred_mask = torch.zeros_like(y_support, dtype=torch.bool)
    true_mask = torch.zeros_like(y_support, dtype=torch.bool)
    pred_mask.scatter_(1, top_idx, True)
    true_mask.scatter_(1, true_idx, True)
    return float((pred_mask == true_mask).all(dim=1).float().mean().item())


@torch.no_grad()
def top_k_contains_support(logits: torch.Tensor, y_support: torch.Tensor, h: int, k: int) -> float:
    top_idx = torch.sigmoid(logits).topk(k, dim=1).indices
    true_idx = y_support.topk(h, dim=1).indices
    contains = (top_idx.unsqueeze(-1) == true_idx.unsqueeze(1)).any(dim=1).all(dim=1)
    return float(contains.float().mean().item())


def random_top_h_recall(n: int, h: int) -> float:
    return h / n

