from __future__ import annotations

import argparse

from src.lwe.generator import generator_sanity_check
from src.utils.config import load_config
from src.utils.seed import resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "cpu")))
    lwe = cfg["lwe"]
    print(
        generator_sanity_check(
            batch_size=8,
            n=int(lwe["n"]),
            m=int(lwe["m"]),
            q=int(lwe["q"]),
            h=int(lwe["h"]),
            noise_bound=int(lwe["noise_bound"]),
            secret_type=str(lwe.get("secret_type", "binary")),
            noise_type=str(lwe.get("noise_type", "uniform_small")),
            device=device,
        )
    )


if __name__ == "__main__":
    main()

