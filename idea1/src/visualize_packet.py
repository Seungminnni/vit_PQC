from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.dataset import OnTheFlyLWEDataset
from src.models.factory import build_model
from src.train import active_embedding_key
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.seed import resolve_device, seed_everything
from src.utils.visualization import save_heatmap, save_scores_bar
from src.training.trainer import model_input


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out-dir", default="outputs/figures/packet")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)) + 3)
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
    batch = dataset.sample_batch(1)
    out_dir = Path(args.out_dir)
    for key, tensor in batch["packet"].items():
        if tensor.ndim == 4:
            save_heatmap(tensor[0, 0], out_dir / f"{key}_channel0.png", title=f"{key} channel 0")
    if args.checkpoint:
        model = build_model(config).to(device)
        load_checkpoint(args.checkpoint, model, map_location=device)
        model.eval()
        embedding_key = active_embedding_key(config)
        logits = model(model_input(model, batch["packet"], embedding_key))
        save_scores_bar(logits[0], batch["y_support"][0], out_dir / "predicted_support_scores.png")
    print({"out_dir": str(out_dir)})


if __name__ == "__main__":
    main()
