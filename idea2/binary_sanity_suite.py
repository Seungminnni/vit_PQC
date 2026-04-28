from __future__ import annotations

import argparse
import math

import torch

from candidate import topk_indices_from_logits
from configs import add_common_args, build_config
from data import b_shuffle_batch, column_permute_batch, generate_lwe_batch, split_binary_supports
from features.factory import encode_features
from metrics import evaluate_binary_candidates, random_candidate_hit_probability
from models.full_model import RHIECGModel
from recovery import recover_binary_trace
from utils import resolve_device, seed_everything


def oracle_logits(s: torch.Tensor, high: float = 12.0, low: float = -12.0) -> torch.Tensor:
    return torch.where(s != 0, torch.full_like(s, high, dtype=torch.float), torch.full_like(s, low, dtype=torch.float))


def random_logits_like(s: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(s.float())


@torch.no_grad()
def oracle_recovery_check(cfg, device) -> dict[str, float]:
    batch = generate_lwe_batch(cfg.train.batch_size, cfg.lwe.n, cfg.lwe.M, cfg.lwe.q, cfg.lwe.h, cfg.lwe.sigma_e, device, "binary")
    output = {"support_logits": oracle_logits(batch.s)}
    return evaluate_binary_candidates(batch, output, cfg.lwe.h, cfg.candidate.topK, cfg.candidate.score_type)


@torch.no_grad()
def random_posterior_check(cfg, device, trials: int) -> dict[str, float]:
    total = 0.0
    for _ in range(trials):
        batch = generate_lwe_batch(cfg.train.batch_size, cfg.lwe.n, cfg.lwe.M, cfg.lwe.q, cfg.lwe.h, cfg.lwe.sigma_e, device, "binary")
        output = {"support_logits": random_logits_like(batch.s)}
        total += evaluate_binary_candidates(batch, output, cfg.lwe.h, cfg.candidate.topK, cfg.candidate.score_type)["candidate_hit_rate"]
    empirical = total / trials
    expected = random_candidate_hit_probability(cfg.lwe.n, cfg.lwe.h, cfg.candidate.topK)
    return {"empirical_random_hit": empirical, "expected_random_hit": expected, "abs_error": abs(empirical - expected)}


@torch.no_grad()
def pair_filter_check(cfg, device) -> dict[str, float]:
    h_original = cfg.lwe.h
    cfg.lwe.h = max(2, h_original)
    batch = generate_lwe_batch(cfg.train.batch_size, cfg.lwe.n, cfg.lwe.M, cfg.lwe.q, cfg.lwe.h, cfg.lwe.sigma_e, device, "binary")
    output = {"support_logits": oracle_logits(batch.s)}
    plain = evaluate_binary_candidates(batch, output, cfg.lwe.h, cfg.candidate.topK, cfg.candidate.score_type)
    paired = evaluate_binary_candidates(
        batch,
        output,
        cfg.lwe.h,
        cfg.candidate.topK,
        cfg.candidate.score_type,
        use_pair_filter=True,
        pair_budget=cfg.candidate.pair_budget,
    )
    cfg.lwe.h = h_original
    return {
        "plain_candidate_hit": plain["candidate_hit_rate"],
        "pair_candidate_hit": paired["candidate_hit_rate"],
        "plain_candidate_count": plain["candidate_count"],
        "pair_candidate_count": paired["candidate_count"],
        "pair_post_rerank": paired["post_rerank_full_match"],
    }


@torch.no_grad()
def permutation_equivariance_check(cfg, device) -> dict[str, float]:
    batch = generate_lwe_batch(2, cfg.lwe.n, cfg.lwe.M, cfg.lwe.q, cfg.lwe.h, 0.0, device, "binary")
    X = encode_features(batch, cfg.features, cfg.lwe.integer_values)
    model = RHIECGModel(
        in_channels=X.shape[1],
        n=cfg.lwe.n,
        d_model=min(32, cfg.model.d_model),
        depth=1,
        heads=4,
        dropout=0.0,
        pooling=cfg.model.pooling,
        coordinate_transformer=True,
        axial_mode=cfg.model.axial_mode,
        use_position=False,
    ).to(device)
    model.eval()
    logits = model(X)["support_logits"]
    perm_batch, perm = column_permute_batch(batch)
    logits_p = model(encode_features(perm_batch, cfg.features, cfg.lwe.integer_values))["support_logits"]
    b_bad = b_shuffle_batch(batch)
    delta_b = (encode_features(batch, cfg.features, cfg.lwe.integer_values) - encode_features(b_bad, cfg.features, cfg.lwe.integer_values)).abs().mean()
    return {
        "perm_equivariance_max_error": float((logits[:, perm] - logits_p).abs().max().item()),
        "b_shuffle_feature_delta": float(delta_b.item()),
    }


def secret_split_check(cfg) -> dict[str, int | bool]:
    train_pool, eval_pool = split_binary_supports(cfg.lwe.n, cfg.lwe.h, cfg.lwe.train_secret_fraction, cfg.lwe.split_seed)
    overlap = set(train_pool) & set(eval_pool)
    return {
        "train_secret_count": len(train_pool),
        "eval_secret_count": len(eval_pool),
        "overlap_count": len(overlap),
        "is_disjoint": len(overlap) == 0,
    }


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Binary-only sanity suite"))
    parser.add_argument("--random_trials", type=int, default=10)
    args = parser.parse_args()
    cfg = build_config(args, secret_type="binary")
    seed_everything(cfg.train.seed + 500)
    device = resolve_device(cfg.train.device)
    print({"oracle": oracle_recovery_check(cfg, device)})
    print({"random": random_posterior_check(cfg, device, args.random_trials)})
    print({"pair_filter": pair_filter_check(cfg, device)})
    print({"permutation_and_b_shuffle": permutation_equivariance_check(cfg, device)})
    print({"secret_split": secret_split_check(cfg)})


if __name__ == "__main__":
    main()
