from __future__ import annotations

import itertools
from math import comb

import torch


def topk_indices_from_logits(logits: torch.Tensor, K: int) -> torch.Tensor:
    K_eff = min(K, logits.shape[-1])
    return torch.sigmoid(logits).topk(K_eff, dim=-1).indices


def enumerate_binary_supports(topk: list[int], h: int) -> list[tuple[int, ...]]:
    return [tuple(int(x) for x in support) for support in itertools.combinations(topk, h)]


def binary_secret_from_support(n: int, support: tuple[int, ...], device) -> torch.Tensor:
    s = torch.zeros(n, dtype=torch.long, device=device)
    s[list(support)] = 1
    return s


def enumerate_binary_candidates(n: int, topk: list[int], h: int, device) -> list[torch.Tensor]:
    return [binary_secret_from_support(n, support, device) for support in enumerate_binary_supports(topk, h)]


def binary_candidate_matrix(n: int, supports: list[tuple[int, ...]], device) -> torch.Tensor:
    candidates = torch.zeros(len(supports), n, dtype=torch.long, device=device)
    for row, support in enumerate(supports):
        candidates[row, list(support)] = 1
    return candidates


def binary_support_from_secret(s: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(x) for x in torch.where(s != 0)[0].tolist())


def enumerate_ternary_candidates(n: int, topk: list[int], h: int, device) -> list[torch.Tensor]:
    candidates: list[torch.Tensor] = []
    for support in itertools.combinations(topk, h):
        for signs in itertools.product((-1, 1), repeat=h):
            s = torch.zeros(n, dtype=torch.long, device=device)
            s[list(support)] = torch.tensor(signs, dtype=torch.long, device=device)
            candidates.append(s)
    return candidates


def integer_class_values(integer_values: tuple[int, ...], device) -> torch.Tensor:
    return torch.tensor((0, *integer_values), dtype=torch.long, device=device)


def enumerate_integer_candidates_from_logits(
    support_logits: torch.Tensor,
    integer_logits: torch.Tensor,
    h: int,
    K: int,
    value_topr: int,
    integer_values: tuple[int, ...],
) -> list[torch.Tensor]:
    n = support_logits.shape[-1]
    topk = topk_indices_from_logits(support_logits.unsqueeze(0), K)[0].tolist()
    value_table = integer_class_values(integer_values, support_logits.device)
    candidates: list[torch.Tensor] = []
    for support in itertools.combinations(topk, h):
        per_coord_values: list[list[int]] = []
        for idx in support:
            probs = torch.softmax(integer_logits[idx], dim=-1)
            nz_scores = probs[1:]
            class_idx = nz_scores.topk(min(value_topr, nz_scores.numel())).indices + 1
            per_coord_values.append([int(value_table[c].item()) for c in class_idx])
        for values in itertools.product(*per_coord_values):
            s = torch.zeros(n, dtype=torch.long, device=support_logits.device)
            s[list(support)] = torch.tensor(values, dtype=torch.long, device=support_logits.device)
            candidates.append(s)
    return candidates


def greedy_beam_integer_candidates(
    support_logits: torch.Tensor,
    integer_logits: torch.Tensor,
    h: int,
    K: int,
    beam_width: int,
    integer_values: tuple[int, ...],
) -> list[torch.Tensor]:
    n = support_logits.shape[-1]
    topk = topk_indices_from_logits(support_logits.unsqueeze(0), K)[0].tolist()
    class_values = integer_class_values(integer_values, support_logits.device)
    beams: list[tuple[float, torch.Tensor, set[int]]] = [(0.0, torch.zeros(n, dtype=torch.long, device=support_logits.device), set())]
    support_logp = torch.log(torch.sigmoid(support_logits).clamp_min(1e-8))
    value_logp = torch.log_softmax(integer_logits, dim=-1)
    for _ in range(h):
        expanded = []
        for score, s, used in beams:
            for idx in topk:
                if idx in used:
                    continue
                for cls in range(1, len(class_values)):
                    s_new = s.clone()
                    s_new[idx] = class_values[cls]
                    expanded.append((score + float(support_logp[idx] + value_logp[idx, cls]), s_new, used | {idx}))
        expanded.sort(key=lambda item: item[0], reverse=True)
        beams = expanded[:beam_width]
    return [s for _, s, _ in beams]


def reduction_factor(n: int, h: int, K: int, secret_type: str = "binary", value_width: int = 2) -> float:
    if K < h:
        return 0.0
    full = comb(n, h)
    reduced = comb(K, h)
    if secret_type == "ternary":
        full *= 2**h
        reduced *= 2**h
    if secret_type == "integer":
        full *= value_width**h
        reduced *= value_width**h
    return float(full / max(reduced, 1))
