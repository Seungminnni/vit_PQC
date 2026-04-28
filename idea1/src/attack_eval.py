from __future__ import annotations

import argparse

import torch

from src.attack.recover import recover_batch_exact_rate
from src.data.dataset import OnTheFlyLWEDataset
from src.models.factory import build_model
from src.train import active_embedding_key
from src.training.metrics import exact_support_recovery, top_h_recall, top_k_contains_support
from src.training.trainer import model_input
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.seed import resolve_device, seed_everything


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--candidate-k", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)) + 2)
    device = resolve_device(str(config.get("device", "cpu")))
    lwe = config["lwe"]
    h = int(lwe["h"])
    k = args.candidate_k or int(config.get("attack", {}).get("candidate_k", max(2 * h, h)))
    batches = args.batches or int(config.get("eval", {}).get("batches", 20))
    dataset = OnTheFlyLWEDataset(
        n=int(lwe["n"]),
        m=int(lwe["m"]),
        q=int(lwe["q"]),
        h=h,
        noise_bound=int(lwe["noise_bound"]),
        secret_type=str(lwe.get("secret_type", "binary")),
        noise_type=str(lwe.get("noise_type", "uniform_small")),
        embedding_config=config["embedding"],
        device=device,
    )
    model = build_model(config).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    totals = {"top_h_recall": 0.0, "direct_exact": 0.0, f"top_{k}_contains": 0.0, "verify_exact": 0.0}
    embedding_key = active_embedding_key(config)
    for _ in range(batches):
        batch = dataset.sample_batch(int(config["train"]["batch_size"]))
        logits = model(model_input(model, batch["packet"], embedding_key))
        totals["top_h_recall"] += top_h_recall(logits, batch["y_support"], h)
        totals["direct_exact"] += exact_support_recovery(logits, batch["y_support"], h)
        totals[f"top_{k}_contains"] += top_k_contains_support(logits, batch["y_support"], h, k)
        totals["verify_exact"] += recover_batch_exact_rate(
            A=batch["A"],
            b=batch["b"],
            s=batch["s"],
            logits=logits,
            h=h,
            k=k,
            q=int(lwe["q"]),
        )
    print({key: val / batches for key, val in totals.items()})


if __name__ == "__main__":
    main()
