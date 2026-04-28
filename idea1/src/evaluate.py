from __future__ import annotations

import argparse

from src.data.dataset import OnTheFlyLWEDataset
from src.models.factory import build_model
from src.training.trainer import evaluate
from src.train import active_embedding_key
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.seed import resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)) + 1)
    device = resolve_device(str(config.get("device", "cpu")))
    lwe = config["lwe"]
    dataset = OnTheFlyLWEDataset(
        n=int(lwe["n"]),
        m=int(lwe["m"]),
        q=int(lwe["q"]),
        h=int(lwe["h"]),
        noise_bound=int(lwe["noise_bound"]),
        secret_type=str(lwe.get("secret_type", "binary")),
        noise_type=str(lwe.get("noise_type", "uniform_small")),
        embedding_config=config["embedding"],
        device=device,
    )
    model = build_model(config).to(device)
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device)
    metrics = evaluate(
        model=model,
        dataset=dataset,
        batch_size=int(config["train"]["batch_size"]),
        batches=int(config.get("eval", {}).get("batches", 50)),
        h=int(lwe["h"]),
        embedding_key=active_embedding_key(config),
        top_k=int(config.get("eval", {}).get("top_k", lwe["h"])),
    )
    print(metrics)


if __name__ == "__main__":
    main()

