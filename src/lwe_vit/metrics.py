from __future__ import annotations

import torch

from .lwe import residual_from_secret, secret_value_table


@torch.no_grad()
def decode_predictions(logits: torch.Tensor, q: int, secret_dist: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    labels = probs.argmax(dim=-1)
    values = secret_value_table(q, secret_dist, device=logits.device)
    secret_hat = values[labels].to(torch.long)
    support_scores = 1.0 - probs[..., values == 0].sum(dim=-1) if (values == 0).any() else probs.max(dim=-1).values
    return secret_hat, labels.to(torch.long), support_scores


@torch.no_grad()
def batch_statistics(
    logits: torch.Tensor,
    target_labels: torch.Tensor,
    secret: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    oracle_residual: torch.Tensor,
    q: int,
    noise_width: float,
    residual_success_factor: float,
    secret_dist: str,
) -> dict[str, float]:
    secret_hat, pred_labels, _ = decode_predictions(logits, q=q, secret_dist=secret_dist)
    target_labels = target_labels.to(torch.long)
    secret = secret.to(torch.long)
    secret_hat = secret_hat.to(torch.long)

    coord_correct = (pred_labels == target_labels).sum().item()
    exact_match = (secret_hat == secret).all(dim=1)
    true_support = secret != 0
    pred_support = secret_hat != 0

    tp = torch.logical_and(true_support, pred_support).sum().item()
    fp = torch.logical_and(~true_support, pred_support).sum().item()
    fn = torch.logical_and(true_support, ~pred_support).sum().item()
    tn = torch.logical_and(~true_support, ~pred_support).sum().item()

    sample_tp = torch.logical_and(true_support, pred_support).sum(dim=1).to(torch.float32)
    sample_fp = torch.logical_and(~true_support, pred_support).sum(dim=1).to(torch.float32)
    sample_fn = torch.logical_and(true_support, ~pred_support).sum(dim=1).to(torch.float32)
    sample_precision = sample_tp / (sample_tp + sample_fp).clamp(min=1.0)
    sample_recall = sample_tp / (sample_tp + sample_fn).clamp(min=1.0)
    sample_f1_den = sample_precision + sample_recall
    sample_empty_match = torch.logical_and(true_support.sum(dim=1) == 0, pred_support.sum(dim=1) == 0)
    sample_f1 = torch.where(
        sample_f1_den > 0,
        2.0 * sample_precision * sample_recall / sample_f1_den,
        sample_empty_match.to(torch.float32),
    )

    residual = residual_from_secret(A, b, secret_hat, q=q).to(torch.float32)
    oracle_residual = oracle_residual.to(torch.float32)
    pred_resid_std = residual.std(dim=1, unbiased=False)
    oracle_resid_std = oracle_residual.std(dim=1, unbiased=False)

    if noise_width == 0.0:
        residual_success = residual.eq(0).all(dim=1)
        oracle_success = oracle_residual.eq(0).all(dim=1)
    else:
        threshold = residual_success_factor * noise_width
        residual_success = pred_resid_std.le(threshold)
        oracle_success = oracle_resid_std.le(threshold)

    stats = {
        "sample_count": float(secret.shape[0]),
        "coord_count": float(secret.numel()),
        "coord_correct": float(coord_correct),
        "exact_correct": float(exact_match.sum().item()),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "support_tp": float(tp),
        "support_fp": float(fp),
        "support_fn": float(fn),
        "support_tn": float(tn),
        "support_exact_count": float((true_support == pred_support).all(dim=1).sum().item()),
        "sample_support_f1_sum": float(sample_f1.sum().item()),
        "pred_support_count": float(pred_support.sum().item()),
        "true_support_count": float(true_support.sum().item()),
        "pred_residual_std_sum": float(pred_resid_std.sum().item()),
        "oracle_residual_std_sum": float(oracle_resid_std.sum().item()),
        "pred_residual_std_gap_sum": float((pred_resid_std - noise_width).abs().sum().item()),
        "residual_success_count": float(residual_success.sum().item()),
        "oracle_success_count": float(oracle_success.sum().item()),
    }

    for class_idx in range(logits.shape[-1]):
        true_now = target_labels == class_idx
        pred_now = pred_labels == class_idx
        stats[f"class_{class_idx}_tp"] = float(torch.logical_and(true_now, pred_now).sum().item())
        stats[f"class_{class_idx}_fp"] = float(torch.logical_and(~true_now, pred_now).sum().item())
        stats[f"class_{class_idx}_fn"] = float(torch.logical_and(true_now, ~pred_now).sum().item())
    return stats


def merge_statistics(stat_list: list[dict[str, float]]) -> dict[str, float]:
    if not stat_list:
        raise ValueError("stat_list must not be empty.")
    keys = set().union(*(stats.keys() for stats in stat_list))
    merged = {key: 0.0 for key in keys}
    for stats in stat_list:
        for key, value in stats.items():
            merged[key] += value
    return merged


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1.0)


def finalize_statistics(stats: dict[str, float], num_classes: int) -> dict[str, float]:
    precision = _safe_div(stats["tp"], stats["tp"] + stats["fp"])
    recall = _safe_div(stats["tp"], stats["tp"] + stats["fn"])
    support_f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)

    class_f1_values = []
    for class_idx in range(num_classes):
        class_tp = stats.get(f"class_{class_idx}_tp", 0.0)
        class_fp = stats.get(f"class_{class_idx}_fp", 0.0)
        class_fn = stats.get(f"class_{class_idx}_fn", 0.0)
        class_precision = _safe_div(class_tp, class_tp + class_fp)
        class_recall = _safe_div(class_tp, class_tp + class_fn)
        class_f1 = 0.0 if class_precision + class_recall == 0.0 else 2.0 * class_precision * class_recall / (class_precision + class_recall)
        class_f1_values.append(class_f1)

    coord_acc = _safe_div(stats["coord_correct"], stats["coord_count"])
    exact_match = _safe_div(stats["exact_correct"], stats["sample_count"])
    return {
        "coord_acc": coord_acc,
        "coordinate_accuracy": coord_acc,
        "exact_match": exact_match,
        "exact_recovery_rate": exact_match,
        "support_precision": precision,
        "support_recall": recall,
        "active_recall": recall,
        "support_f1": support_f1,
        "support_f1_micro": support_f1,
        "support_f1_sample_mean": _safe_div(stats["sample_support_f1_sum"], stats["sample_count"]),
        "macro_f1": sum(class_f1_values) / max(len(class_f1_values), 1),
        "normalized_hamming_error": 1.0 - coord_acc,
        "support_exact_match": _safe_div(stats["support_exact_count"], stats["sample_count"]),
        "support_tp": stats["support_tp"],
        "support_fp": stats["support_fp"],
        "support_fn": stats["support_fn"],
        "support_tn": stats["support_tn"],
        "support_pred_rate": _safe_div(stats["pred_support_count"], stats["coord_count"]),
        "support_true_rate": _safe_div(stats["true_support_count"], stats["coord_count"]),
        "pred_residual_std_mean": _safe_div(stats["pred_residual_std_sum"], stats["sample_count"]),
        "oracle_residual_std_mean": _safe_div(stats["oracle_residual_std_sum"], stats["sample_count"]),
        "pred_residual_std_gap_mean": _safe_div(stats["pred_residual_std_gap_sum"], stats["sample_count"]),
        "residual_success_rate": _safe_div(stats["residual_success_count"], stats["sample_count"]),
        "oracle_residual_success_rate": _safe_div(stats["oracle_success_count"], stats["sample_count"]),
    }
