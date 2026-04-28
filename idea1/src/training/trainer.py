from __future__ import annotations

from typing import Any

import torch

from src.attack.recover import recover_batch_exact_rate
from src.training.losses import support_bce_loss
from src.training.metrics import exact_support_recovery, top_h_recall, top_k_contains_support
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import JSONLLogger


def model_input(model: torch.nn.Module, packet: dict[str, torch.Tensor], embedding_key: str):
    if getattr(model, "consumes_packet", False):
        return packet
    return packet[embedding_key]


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().data.norm(2).item() ** 2)
    return total**0.5


def train_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset,
    batch_size: int,
    n: int,
    h: int,
    embedding_key: str = "rhie",
) -> dict[str, float]:
    model.train()
    batch = dataset.sample_batch(batch_size)
    x = model_input(model, batch["packet"], embedding_key)
    y = batch["y_support"]
    logits = model(x)
    loss = support_bce_loss(logits, y, n=n, h=h)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = grad_norm(model.parameters())
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "top_h_recall": top_h_recall(logits, y, h),
        "exact_support": exact_support_recovery(logits, y, h),
        "grad_norm": gnorm,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    batches: int,
    h: int,
    embedding_key: str,
    top_k: int | None = None,
    verify_k: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"top_h_recall": 0.0, "exact_support": 0.0}
    if top_k:
        totals[f"top_{top_k}_contains_support"] = 0.0
    verify_total = 0.0
    for _ in range(batches):
        batch = dataset.sample_batch(batch_size)
        x = model_input(model, batch["packet"], embedding_key)
        y = batch["y_support"]
        logits = model(x)
        totals["top_h_recall"] += top_h_recall(logits, y, h)
        totals["exact_support"] += exact_support_recovery(logits, y, h)
        if top_k:
            totals[f"top_{top_k}_contains_support"] += top_k_contains_support(logits, y, h, top_k)
        if verify_k:
            verify_total += recover_batch_exact_rate(
                A=batch["A"],
                b=batch["b"],
                s=batch["s"],
                logits=logits,
                h=h,
                k=verify_k,
                q=dataset.q,
            )
    out = {key: val / batches for key, val in totals.items()}
    if verify_k:
        out["verify_exact_support"] = verify_total / batches
    return out


def train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset,
    config: dict[str, Any],
    run_dir,
    embedding_key: str,
) -> dict[str, float]:
    train_cfg = config["train"]
    eval_cfg = config.get("eval", {})
    attack_cfg = config.get("attack", {})
    n = config["lwe"]["n"]
    h = config["lwe"]["h"]
    logger = JSONLLogger(run_dir / "metrics.jsonl")
    best_recall = -1.0
    best_metrics: dict[str, float] = {}

    for step in range(1, int(train_cfg["steps"]) + 1):
        metrics = train_one_step(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            batch_size=int(train_cfg["batch_size"]),
            n=n,
            h=h,
            embedding_key=embedding_key,
        )
        if step % int(train_cfg.get("log_every", 100)) == 0 or step == 1:
            row = {"step": step, "split": "train", **metrics}
            logger.log(row)
            print(row, flush=True)
        if step % int(train_cfg.get("eval_every", 1000)) == 0 or step == int(train_cfg["steps"]):
            verify_k = int(attack_cfg.get("candidate_k", 0)) if attack_cfg.get("use_verification", False) else None
            eval_metrics = evaluate(
                model=model,
                dataset=dataset,
                batch_size=int(train_cfg["batch_size"]),
                batches=int(eval_cfg.get("batches", 10)),
                h=h,
                embedding_key=embedding_key,
                top_k=int(eval_cfg.get("top_k", h)),
                verify_k=verify_k,
            )
            row = {"step": step, "split": "eval", **eval_metrics}
            logger.log(row)
            print(row, flush=True)
            if eval_metrics["top_h_recall"] > best_recall:
                best_recall = eval_metrics["top_h_recall"]
                best_metrics = eval_metrics
                save_checkpoint(run_dir / "best.pt", model, optimizer, config, step, eval_metrics)
    logger.close()
    save_checkpoint(run_dir / "last.pt", model, optimizer, config, int(train_cfg["steps"]), best_metrics)
    return best_metrics
