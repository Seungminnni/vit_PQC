from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch

from configs import expected_nonzero_probability
from data import generate_lwe_batch
from features.factory import encode_features
from logger import JSONLLogger
from losses import binary_support_bce, circular_residual_auxiliary
from metrics import evaluate_binary_hfree_candidates
from models.decoder import support_logits_from_output
from models.full_model import RHIECGModel
from utils import count_parameters, make_run_dir, resolve_device, save_checkpoint, save_json, seed_everything


def amp_dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported amp_dtype={name}")


def autocast_context(cfg, device):
    enabled = bool(cfg.train.amp and device.type == "cuda")
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype_from_name(cfg.train.amp_dtype), enabled=True)


def make_grad_scaler(cfg, device):
    enabled = bool(cfg.train.amp and cfg.train.amp_dtype == "fp16" and device.type == "cuda")
    return torch.amp.GradScaler("cuda", enabled=enabled)


def batch_distribution(cfg, split: str) -> tuple[str, int]:
    if split == "train" and cfg.lwe.fixed_train_h is not None:
        return "fixed", int(cfg.lwe.fixed_train_h)
    if split == "eval" and cfg.lwe.fixed_eval_h is not None:
        return "fixed", int(cfg.lwe.fixed_eval_h)
    return cfg.lwe.secret_distribution, cfg.lwe.h


def sample_batch(cfg, device, split: str = "train"):
    distribution, h = batch_distribution(cfg, split)
    return generate_lwe_batch(
        batch_size=cfg.train.batch_size,
        n=cfg.lwe.n,
        M=cfg.lwe.M,
        q=cfg.lwe.q,
        h=h,
        sigma_e=cfg.lwe.sigma_e,
        device=device,
        secret_distribution=distribution,
        h_min=cfg.lwe.h_min,
        h_max=cfg.lwe.h_max,
        p_nonzero=cfg.lwe.p_nonzero,
        noise_type=cfg.lwe.noise_type,
    )


def build_model_from_probe(cfg, device) -> tuple[RHIECGModel, int]:
    probe = sample_batch(cfg, device, split="train")
    X = encode_features(probe, cfg.features)
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
        secret_type="binary",
    ).to(device)
    return model, X.shape[1]


def pos_weight_for_loss(cfg) -> float:
    if cfg.train.loss_pos_weight_mode == "none":
        return 1.0
    if cfg.train.loss_pos_weight_mode == "manual":
        return float(cfg.train.pos_weight)
    if cfg.train.loss_pos_weight_mode == "prior":
        p = expected_nonzero_probability(cfg.lwe)
        return (1.0 - p) / p
    raise ValueError(f"unsupported loss_pos_weight_mode={cfg.train.loss_pos_weight_mode}")


def compute_loss(cfg, batch, output: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    support_logits = support_logits_from_output(output, "binary")
    weight = pos_weight_for_loss(cfg)
    support_loss = binary_support_bce(support_logits, batch.support, pos_weight=weight)
    total = support_loss
    parts = {
        "loss_support": float(support_loss.item()),
        "loss_pos_weight": float(weight),
    }

    if cfg.train.aux_residual_weight > 0:
        aux = circular_residual_auxiliary(batch.A, batch.b, torch.sigmoid(support_logits), batch.q)
        total = total + cfg.train.aux_residual_weight * aux
        parts["loss_circular_aux"] = float(aux.item())
    parts["loss"] = float(total.item())
    return total, parts


@torch.no_grad()
def evaluate_model(cfg, model, device, batches: int | None = None) -> dict[str, float]:
    model.eval()
    n_batches = batches or cfg.train.eval_batches
    total: dict[str, float] = {}
    for _ in range(n_batches):
        batch = sample_batch(cfg, device, split="eval")
        X = encode_features(batch, cfg.features)
        with autocast_context(cfg, device):
            output = model(X)
        metrics = evaluate_binary_hfree_candidates(
            batch,
            output,
            cfg.candidate.hfree_uncertain_K,
            cfg.candidate.hfree_threshold,
            cfg.candidate.score_type,
            cfg.candidate.posterior_weight,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler = make_grad_scaler(cfg, device)
    logger = JSONLLogger(run_dir / "metrics.jsonl")
    best_metric = -1.0
    best_eval: dict[str, float] = {}
    print(
        {
            "run_dir": str(run_dir),
            "device": str(device),
            "in_channels": in_channels,
            "params": count_parameters(model),
            "train_distribution": batch_distribution(cfg, "train")[0],
            "eval_distribution": batch_distribution(cfg, "eval")[0],
        },
        flush=True,
    )

    for step in range(1, cfg.train.steps + 1):
        model.train()
        batch = sample_batch(cfg, device, split="train")
        X = encode_features(batch, cfg.features)
        with autocast_context(cfg, device):
            output = model(X)
            loss, loss_parts = compute_loss(cfg, batch, output)
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if step == 1 or step % cfg.train.log_every == 0:
            row = {"step": step, "split": "train", **loss_parts}
            logger.log(row)
            print(row, flush=True)

        if step == cfg.train.steps or step % cfg.train.eval_every == 0:
            eval_metrics = evaluate_model(cfg, model, device)
            row = {"step": step, "split": "eval", **eval_metrics}
            logger.log(row)
            print(row, flush=True)
            score = eval_metrics.get("post_verifier_full_match", 0.0) + eval_metrics.get("candidate_contains_true", 0.0)
            if score > best_metric:
                best_metric = score
                best_eval = eval_metrics
                save_checkpoint(run_dir / "best.pt", model, optimizer, cfg.to_dict(), step, eval_metrics)
    save_checkpoint(run_dir / "last.pt", model, optimizer, cfg.to_dict(), cfg.train.steps, best_eval)
    logger.close()
    return run_dir, best_eval
