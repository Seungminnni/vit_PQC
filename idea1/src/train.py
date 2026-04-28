from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from src.data.dataset import OnTheFlyLWEDataset
from src.lwe.generator import generator_sanity_check
from src.models.factory import build_model
from src.training.trainer import train_loop
from src.utils.config import load_config
from src.utils.seed import resolve_device, seed_everything


def active_embedding_key(config: dict) -> str:
    emb = config["embedding"]
    keys = [key for key in ("rhie", "phase", "raw") if emb.get(f"use_{key}", False)]
    if not keys:
        raise ValueError("At least one trainable embedding view must be enabled")
    return keys[0]


def make_run_dir(config_path: str, output_root: str | Path = "outputs") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = Path(config_path).stem
    run_dir = Path(output_root) / "logs" / f"{name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "cpu")))
    config["resolved_device"] = str(device)
    run_dir = make_run_dir(args.config, args.output_root)
    with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2, sort_keys=True)

    lwe = config["lwe"]
    sanity = generator_sanity_check(
        batch_size=8,
        n=int(lwe["n"]),
        m=int(lwe["m"]),
        q=int(lwe["q"]),
        h=int(lwe["h"]),
        noise_bound=int(lwe["noise_bound"]),
        device=device,
        secret_type=str(lwe.get("secret_type", "binary")),
        noise_type=str(lwe.get("noise_type", "uniform_small")),
    )
    print({"sanity": sanity, "run_dir": str(run_dir)}, flush=True)
    if not sanity["passed"]:
        raise RuntimeError(f"generator sanity check failed: {sanity}")

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    best = train_loop(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        config=config,
        run_dir=run_dir,
        embedding_key=active_embedding_key(config),
    )
    print({"best_eval": best, "run_dir": str(run_dir)}, flush=True)


if __name__ == "__main__":
    main()

