import torch

from .data import centered_mod


@torch.no_grad()
def decode_predictions(logits: torch.Tensor, secret_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    if secret_type != "binary":
        raise ValueError(f"Only binary secret_type is supported, got {secret_type}")
    probs = torch.softmax(logits, dim=-1)
    pred_class = probs.argmax(dim=-1)
    secret_hat = pred_class.to(torch.int64)
    support_scores = probs[..., 1]

    return secret_hat, support_scores


@torch.no_grad()
def centered_residual(A: torch.Tensor, b: torch.Tensor, secret_hat: torch.Tensor, q: int) -> torch.Tensor:
    A_int = A.to(torch.int64)
    secret_int = secret_hat.to(torch.int64)
    pred_b = (A_int * secret_int.unsqueeze(1)).sum(dim=-1)
    return centered_mod(pred_b - b.to(torch.int64), q).to(torch.float32)


@torch.no_grad()
def batch_statistics(
    logits: torch.Tensor,
    secret: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    oracle_residual: torch.Tensor,
    q: int,
    sigma: float,
    residual_success_factor: float,
    secret_type: str,
) -> dict[str, float]:
    if secret_type != "binary":
        raise ValueError(f"Only binary secret_type is supported, got {secret_type}")
    secret_hat, _ = decode_predictions(logits, secret_type=secret_type)
    secret = secret.to(torch.int64)
    secret_hat = secret_hat.to(torch.int64)
    true_class = secret
    pred_class = secret_hat
    num_classes = 2

    coord_correct = (secret_hat == secret).sum().item()
    exact_match_direct = (secret_hat == secret).all(dim=1)
    exact_correct = exact_match_direct.sum().item()

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
    sample_true_empty = true_support.sum(dim=1) == 0
    sample_pred_empty = pred_support.sum(dim=1) == 0
    sample_empty_match = torch.logical_and(sample_true_empty, sample_pred_empty)
    sample_f1 = torch.where(
        sample_f1_den > 0,
        2.0 * sample_precision * sample_recall / sample_f1_den,
        sample_empty_match.to(torch.float32),
    )
    support_exact = (true_support == pred_support).all(dim=1)
    pred_support_count = pred_support.sum().item()
    true_support_count = true_support.sum().item()

    residual = centered_residual(A, b, secret_hat, q=q)
    pred_resid_std = residual.std(dim=1, unbiased=False)
    oracle_resid_std = oracle_residual.to(torch.float32).std(dim=1, unbiased=False)

    if sigma == 0.0:
        residual_success = residual.eq(0).all(dim=1)
        oracle_success = oracle_residual.eq(0).all(dim=1)
    else:
        threshold = residual_success_factor * sigma
        residual_success = pred_resid_std.le(threshold)
        oracle_success = oracle_resid_std.le(threshold)

    stats = {
        "sample_count": float(secret.shape[0]),
        "coord_count": float(secret.numel()),
        "coord_correct": float(coord_correct),
        "exact_correct": float(exact_correct),
        "exact_match_direct_count": float(exact_correct),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "support_tp": float(tp),
        "support_fp": float(fp),
        "support_fn": float(fn),
        "support_tn": float(tn),
        "support_exact_count": float(support_exact.sum().item()),
        "sample_support_f1_sum": float(sample_f1.sum().item()),
        "pred_support_count": float(pred_support_count),
        "true_support_count": float(true_support_count),
        "pred_residual_std_sum": float(pred_resid_std.sum().item()),
        "oracle_residual_std_sum": float(oracle_resid_std.sum().item()),
        "pred_residual_std_gap_sum": float((pred_resid_std - sigma).abs().sum().item()),
        "residual_success_count": float(residual_success.sum().item()),
        "oracle_success_count": float(oracle_success.sum().item()),
    }
    for class_idx in range(num_classes):
        true_now = true_class == class_idx
        pred_now = pred_class == class_idx
        stats[f"class_{class_idx}_tp"] = float(torch.logical_and(true_now, pred_now).sum().item())
        stats[f"class_{class_idx}_fp"] = float(torch.logical_and(~true_now, pred_now).sum().item())
        stats[f"class_{class_idx}_fn"] = float(torch.logical_and(true_now, ~pred_now).sum().item())

    return stats


def merge_statistics(stat_list: list[dict[str, float]]) -> dict[str, float]:
    if not stat_list:
        raise ValueError("stat_list must not be empty")
    keys = set().union(*(stats.keys() for stats in stat_list))
    merged = {key: 0.0 for key in keys}
    for stats in stat_list:
        for key, value in stats.items():
            merged[key] += value
    return merged


def finalize_statistics(stats: dict[str, float], secret_type: str) -> dict[str, float]:
    if secret_type != "binary":
        raise ValueError(f"Only binary secret_type is supported, got {secret_type}")
    precision = stats["tp"] / max(stats["tp"] + stats["fp"], 1.0)
    recall = stats["tp"] / max(stats["tp"] + stats["fn"], 1.0)
    support_f1 = 0.0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    num_classes = 2
    class_f1_values = []
    for class_idx in range(num_classes):
        class_tp = stats.get(f"class_{class_idx}_tp", 0.0)
        class_fp = stats.get(f"class_{class_idx}_fp", 0.0)
        class_fn = stats.get(f"class_{class_idx}_fn", 0.0)
        class_precision = class_tp / max(class_tp + class_fp, 1.0)
        class_recall = class_tp / max(class_tp + class_fn, 1.0)
        class_f1 = 0.0 if (class_precision + class_recall) == 0 else 2.0 * class_precision * class_recall / (class_precision + class_recall)
        class_f1_values.append(class_f1)
    coord_acc = stats["coord_correct"] / max(stats["coord_count"], 1.0)
    exact_match = stats["exact_correct"] / max(stats["sample_count"], 1.0)

    metrics = {
        "coord_acc": coord_acc,
        "coordinate_accuracy": coord_acc,
        "exact_match": exact_match,
        "exact_recovery_rate": exact_match,
        "exact_match_direct": stats["exact_match_direct_count"] / max(stats["sample_count"], 1.0),
        "support_precision": precision,
        "support_recall": recall,
        "active_recall": recall,
        "support_f1": support_f1,
        "support_f1_micro": support_f1,
        "support_f1_sample_mean": stats["sample_support_f1_sum"] / max(stats["sample_count"], 1.0),
        "macro_f1": sum(class_f1_values) / max(len(class_f1_values), 1),
        "normalized_hamming_error": 1.0 - coord_acc,
        "support_exact_match": stats["support_exact_count"] / max(stats["sample_count"], 1.0),
        "support_tp": stats["support_tp"],
        "support_fp": stats["support_fp"],
        "support_fn": stats["support_fn"],
        "support_tn": stats["support_tn"],
        "support_pred_rate": stats["pred_support_count"] / max(stats["coord_count"], 1.0),
        "support_true_rate": stats["true_support_count"] / max(stats["coord_count"], 1.0),
        "pred_residual_std_mean": stats["pred_residual_std_sum"] / max(stats["sample_count"], 1.0),
        "oracle_residual_std_mean": stats["oracle_residual_std_sum"] / max(stats["sample_count"], 1.0),
        "pred_residual_std_gap_mean": stats["pred_residual_std_gap_sum"] / max(stats["sample_count"], 1.0),
        "residual_success_rate": stats["residual_success_count"] / max(stats["sample_count"], 1.0),
        "oracle_residual_success_rate": stats["oracle_success_count"] / max(stats["sample_count"], 1.0),
    }

    return metrics
