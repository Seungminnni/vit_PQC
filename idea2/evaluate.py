from __future__ import annotations

import argparse

import torch

from configs import add_common_args, build_config, config_from_dict
from train_common import build_model_from_probe, build_secret_pools, evaluate_model
from utils import load_checkpoint, resolve_device, seed_everything


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Evaluate RHIE-CG checkpoint"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--secret_type", default="binary", choices=["binary", "ternary", "integer"])
    args = parser.parse_args()
    cfg = build_config(args, secret_type=args.secret_type)
    seed_everything(cfg.train.seed + 100)
    device = resolve_device(cfg.train.device)
    model, _ = build_model_from_probe(cfg, device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device)
        if "config" in payload:
            cfg = config_from_dict(payload["config"])
            cfg.train.device = args.device
            cfg.train.batch_size = args.batch_size
            cfg.train.eval_batches = args.eval_batches
            if args.topK is not None:
                cfg.candidate.topK = args.topK
            if args.use_pair_filter:
                cfg.candidate.use_pair_filter = True
                cfg.candidate.pair_budget = args.pair_budget
            elif args.pair_budget != 64:
                cfg.candidate.pair_budget = args.pair_budget
            if args.pair_score_weight != 0.0:
                cfg.candidate.pair_score_weight = args.pair_score_weight
            if args.posterior_weight != 0.0:
                cfg.candidate.posterior_weight = args.posterior_weight
            model, _ = build_model_from_probe(cfg, device)
        load_checkpoint(args.checkpoint, model, map_location=device)
    metrics = evaluate_model(cfg, model, device, batches=cfg.train.eval_batches, secret_pools=build_secret_pools(cfg))
    print(metrics)


if __name__ == "__main__":
    main()
