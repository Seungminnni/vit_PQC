from __future__ import annotations

import argparse
from pathlib import Path

import torch

from configs import add_common_args, build_config, config_from_dict
from data import generate_lwe_batch
from features.factory import encode_features
from logger import write_csv
from recovery import recover_binary_trace, trace_to_row
from train_common import build_model_from_probe
from utils import load_checkpoint, resolve_device, seed_everything
from visualization import save_candidate_score_plot, save_feature_heatmaps, save_residual_histograms, save_support_bar


def load_cfg_and_model(args, device):
    cfg = build_config(args, secret_type="binary")
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
    model.eval()
    return cfg, model


@torch.no_grad()
def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Binary RHIE-CG failure analysis"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out_dir", default="results/figures/failure_analysis")
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--max_cases", type=int, default=8)
    parser.add_argument("--include_successes", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed + 200)
    device = resolve_device(args.device)
    cfg, model = load_cfg_and_model(args, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    saved = 0
    global_idx = 0
    for _ in range(args.num_batches):
        batch = generate_lwe_batch(
            batch_size=cfg.train.batch_size,
            n=cfg.lwe.n,
            M=cfg.lwe.M,
            q=cfg.lwe.q,
            h=cfg.lwe.h,
            sigma_e=cfg.lwe.sigma_e,
            device=device,
            secret_type="binary",
            noise_type=cfg.lwe.noise_type,
        )
        X = encode_features(batch, cfg.features, cfg.lwe.integer_values)
        output = model(X)
        logits = output["support_logits"]
        for row in range(batch.A.shape[0]):
            trace = recover_binary_trace(
                batch.A[row],
                batch.b[row],
                batch.s[row],
                logits[row],
                h=cfg.lwe.h,
                K=cfg.candidate.topK,
                q=cfg.lwe.q,
                score_type=cfg.candidate.score_type,
                use_pair_filter=cfg.candidate.use_pair_filter,
                pair_budget=cfg.candidate.pair_budget,
                pair_score_weight=cfg.candidate.pair_score_weight,
                posterior_weight=cfg.candidate.posterior_weight,
            )
            case_row = {"case": global_idx, **trace_to_row(trace)}
            rows.append(case_row)
            should_save = args.include_successes or not trace.rerank_exact
            if should_save and saved < args.max_cases:
                prefix = out_dir / f"case_{global_idx:04d}"
                save_support_bar(logits[row], trace, prefix.with_name(prefix.name + "_posterior.png"))
                save_candidate_score_plot(trace, prefix.with_name(prefix.name + "_candidate_scores.png"))
                save_residual_histograms(batch.A[row], batch.b[row], trace, cfg.lwe.q, prefix.with_name(prefix.name + "_residual_hist.png"))
                save_feature_heatmaps(X[row], prefix.with_name(prefix.name + "_features"), max_channels=6)
                cand_rows = [
                    {
                        "rank": rank,
                        "support": " ".join(str(x) for x in cand.support),
                        "is_true": cand.is_true,
                        "residual_score": cand.residual_score,
                        "posterior_score": cand.posterior_score,
                        "pair_score": cand.pair_score,
                        "total_score": cand.total_score,
                    }
                    for rank, cand in enumerate(trace.candidates)
                ]
                write_csv(prefix.with_name(prefix.name + "_candidates.csv"), cand_rows)
                saved += 1
            global_idx += 1
    write_csv(out_dir / "summary.csv", rows)
    print({"out_dir": str(out_dir), "cases": len(rows), "saved_visual_cases": saved})


if __name__ == "__main__":
    main()
