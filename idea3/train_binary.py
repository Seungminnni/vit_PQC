from __future__ import annotations

import argparse

from configs import add_common_args, build_config
from train_common import train


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Train binary RHIE-CG"))
    args = parser.parse_args()
    cfg = build_config(args, secret_type="binary")
    run_dir, best = train(cfg)
    print({"best_eval": best, "run_dir": str(run_dir)}, flush=True)


if __name__ == "__main__":
    main()

