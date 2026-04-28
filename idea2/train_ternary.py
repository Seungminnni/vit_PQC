from __future__ import annotations

import argparse

from configs import add_common_args, build_config
from train_common import train


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Train ternary RHIE-CG"))
    args = parser.parse_args()
    if args.encoder_type == "rhie_cip":
        args.encoder_type = "rhie_cip_ternary"
    cfg = build_config(args, secret_type="ternary")
    run_dir, best = train(cfg)
    print({"best_eval": best, "run_dir": str(run_dir)}, flush=True)


if __name__ == "__main__":
    main()

