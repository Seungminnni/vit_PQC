from __future__ import annotations

from pathlib import Path

import torch

from data import generate_lwe_batch, split_binary_supports
from features.factory import encode_features
from logger import JSONLLogger
from losses import circular_residual_auxiliary, integer_value_loss, support_bce_loss, ternary_value_loss
from metrics import evaluate_binary_candidates, evaluate_integer_candidates, evaluate_ternary_candidates
from models.decoder import support_logits_from_output
from models.full_model import RHIECGModel
from utils import count_parameters, make_run_dir, resolve_device, save_checkpoint, save_json, seed_everything


def build_secret_pools(cfg):
    if cfg.lwe.secret_type != "binary" or not cfg.lwe.secret_split:
        return None
    return split_binary_supports(cfg.lwe.n, cfg.lwe.h, cfg.lwe.train_secret_fraction, cfg.lwe.split_seed)


def sample_batch(cfg, device, split: str = "train", secret_pools=None):
    support_pool = None
    if secret_pools is not None:
        train_pool, eval_pool = secret_pools
        support_pool = train_pool if split == "train" else eval_pool
    return generate_lwe_batch(
        batch_size=cfg.train.batch_size,
        n=cfg.lwe.n,
        M=cfg.lwe.M,
        q=cfg.lwe.q,
        h=cfg.lwe.h,
        sigma_e=cfg.lwe.sigma_e,
        device=device,
        secret_type=cfg.lwe.secret_type,
        noise_type=cfg.lwe.noise_type,
        integer_values=cfg.lwe.integer_values,
        binary_support_pool=support_pool,
    )


def build_model_from_probe(cfg, device) -> tuple[RHIECGModel, int]:
    probe = generate_lwe_batch(
        batch_size=2,
        n=cfg.lwe.n,
        M=cfg.lwe.M,
        q=cfg.lwe.q,
        h=cfg.lwe.h,
        sigma_e=cfg.lwe.sigma_e,
        device=device,
        secret_type=cfg.lwe.secret_type,
        noise_type=cfg.lwe.noise_type,
        integer_values=cfg.lwe.integer_values,
    )
    X = encode_features(probe, cfg.features, cfg.lwe.integer_values)
    model = RHIECGModel(
        in_channels=X.shape[1],
        n=cfg.lwe.n,
        d_model=cfg.model.d_model,
        depth=cfg.model.depth,
        heads=cfg.model.heads,
        dropout=cfg.model.dropout,
        pooling=cfg.model.pooling,
        coordinate_transformer=cfg.model.coordinate_transformer,
        axial_mode=cfg.model.axial_mode,
        use_position=cfg.model.use_position,
        secret_type=cfg.lwe.secret_type,
        integer_values=cfg.lwe.integer_values,
    ).to(device)
    return model, X.shape[1]


def compute_loss(cfg, batch, output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    support_logits = support_logits_from_output(output, cfg.lwe.secret_type)
    support_loss = support_bce_loss(support_logits, batch.support, cfg.lwe.n, cfg.lwe.h)
    total = support_loss
    parts = {"loss_support": float(support_loss.item())}

    if cfg.lwe.secret_type == "ternary":
        value_loss = ternary_value_loss(output["value_logits"], batch.s)
        total = total + value_loss
        parts["loss_value"] = float(value_loss.item())
    elif cfg.lwe.secret_type == "integer":
        value_loss = integer_value_loss(output["integer_logits"], batch.s, cfg.lwe.integer_values)
        total = total + value_loss
        parts["loss_value"] = float(value_loss.item())

    if cfg.train.aux_residual_weight > 0:
        aux = circular_residual_auxiliary(batch.A, batch.b, torch.sigmoid(support_logits), batch.q)
        total = total + cfg.train.aux_residual_weight * aux
        parts["loss_circular_aux"] = float(aux.item())
    parts["loss"] = float(total.item())
    return total, parts


@torch.no_grad()
def evaluate_model(cfg, model, device, batches: int | None = None, secret_pools=None) -> dict[str, float]:
    model.eval()
    n_batches = batches or cfg.train.eval_batches
    total: dict[str, float] = {}
    for _ in range(n_batches):
        batch = sample_batch(cfg, device, split="eval", secret_pools=secret_pools)
        X = encode_features(batch, cfg.features, cfg.lwe.integer_values)
        output = model(X)
        if cfg.lwe.secret_type == "binary":
            metrics = evaluate_binary_candidates(
                batch,
                output,
                cfg.lwe.h,
                cfg.candidate.topK,
                cfg.candidate.score_type,
                use_pair_filter=cfg.candidate.use_pair_filter,
                pair_budget=cfg.candidate.pair_budget,
                pair_score_weight=cfg.candidate.pair_score_weight,
                posterior_weight=cfg.candidate.posterior_weight,
            )
        elif cfg.lwe.secret_type == "ternary":
            metrics = evaluate_ternary_candidates(batch, output, cfg.lwe.h, cfg.candidate.topK, cfg.candidate.score_type)
        else:
            metrics = evaluate_integer_candidates(
                batch,
                output,
                cfg.lwe.h,
                cfg.candidate.topK,
                cfg.candidate.value_topr,
                cfg.lwe.integer_values,
                cfg.candidate.score_type,
            )
        for key, value in metrics.items():
            total[key] = total.get(key, 0.0) + float(value)
    return {key: value / n_batches for key, value in total.items()}


def train(cfg) -> tuple[Path, dict[str, float]]:
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    run_dir = make_run_dir(cfg.run_name)
    save_json(run_dir / "config.json", cfg.to_dict())
    model, in_channels = build_model_from_probe(cfg, device)
    secret_pools = build_secret_pools(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    logger = JSONLLogger(run_dir / "metrics.jsonl")
    best_metric = -1.0
    best_eval: dict[str, float] = {}
    print({"run_dir": str(run_dir), "device": str(device), "in_channels": in_channels, "params": count_parameters(model)}, flush=True)

    for step in range(1, cfg.train.steps + 1):
        model.train()
        batch = sample_batch(cfg, device, split="train", secret_pools=secret_pools)
        X = encode_features(batch, cfg.features, cfg.lwe.integer_values)
        output = model(X)
        loss, loss_parts = compute_loss(cfg, batch, output)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if step == 1 or step % cfg.train.log_every == 0:
            row = {"step": step, "split": "train", **loss_parts}
            logger.log(row)
            print(row, flush=True)

        if step == cfg.train.steps or step % cfg.train.eval_every == 0:
            eval_metrics = evaluate_model(cfg, model, device, secret_pools=secret_pools)
            row = {"step": step, "split": "eval", **eval_metrics}
            logger.log(row)
            print(row, flush=True)
            score = eval_metrics.get("post_rerank_full_match", 0.0) + eval_metrics.get("candidate_hit_rate", 0.0)
            if score > best_metric:
                best_metric = score
                best_eval = eval_metrics
                save_checkpoint(run_dir / "best.pt", model, optimizer, cfg.to_dict(), step, eval_metrics)
    save_checkpoint(run_dir / "last.pt", model, optimizer, cfg.to_dict(), cfg.train.steps, best_eval)
    logger.close()
    return run_dir, best_eval
