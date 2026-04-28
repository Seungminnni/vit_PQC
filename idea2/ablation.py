from __future__ import annotations

import argparse
import copy

from configs import add_common_args, build_config
from logger import write_csv
from train_common import train


def feature_ablation(base_cfg, steps: int) -> list[dict]:
    variants = [
        ("a_only", {"encoder_type": "a_only"}),
        ("raw_only", {"encoder_type": "rhie_cip", "include_raw": True, "include_phase": False, "include_rhie": False, "include_interaction": False, "include_stats": False}),
        ("baseline_8ch", {"encoder_type": "baseline_8ch"}),
        ("baseline_10ch", {"encoder_type": "baseline_10ch"}),
        ("baseline_14ch", {"encoder_type": "baseline_14ch"}),
        ("rhie_only", {"encoder_type": "rhie_cip", "include_raw": False, "include_phase": False, "include_rhie": True, "include_interaction": False, "include_stats": False}),
        ("rhie_phase", {"encoder_type": "rhie_cip", "include_raw": False, "include_phase": True, "include_rhie": True, "include_interaction": False, "include_stats": False}),
        ("rhie_interaction", {"encoder_type": "rhie_cip", "include_raw": False, "include_phase": False, "include_rhie": True, "include_interaction": True, "include_stats": True}),
        ("rhie_cip_full", {"encoder_type": "rhie_cip", "include_raw": True, "include_phase": True, "include_rhie": True, "include_interaction": True, "include_stats": True}),
    ]
    rows = []
    for name, changes in variants:
        cfg = copy.deepcopy(base_cfg)
        cfg.run_name = f"ablation_feature_{name}"
        cfg.train.steps = steps
        for key, value in changes.items():
            setattr(cfg.features, key, value)
        run_dir, best = train(cfg)
        rows.append({"ablation": "feature", "variant": name, "run_dir": str(run_dir), **best})
    return rows


def model_ablation(base_cfg, steps: int) -> list[dict]:
    variants = [
        ("no_axial", {"axial_mode": "none"}),
        ("row_only", {"axial_mode": "row_only"}),
        ("column_only", {"axial_mode": "column_only"}),
        ("row_column", {"axial_mode": "row_column"}),
        ("mean_pool", {"pooling": "mean"}),
        ("meanmax_pool", {"pooling": "meanmax"}),
        ("attention_pool", {"pooling": "attention"}),
        ("no_coord_transformer", {"coordinate_transformer": False}),
        ("coord_transformer", {"coordinate_transformer": True}),
        ("with_position", {"use_position": True}),
        ("no_position", {"use_position": False}),
    ]
    rows = []
    for name, changes in variants:
        cfg = copy.deepcopy(base_cfg)
        cfg.run_name = f"ablation_model_{name}"
        cfg.train.steps = steps
        for key, value in changes.items():
            setattr(cfg.model, key, value)
        run_dir, best = train(cfg)
        rows.append({"ablation": "model", "variant": name, "run_dir": str(run_dir), **best})
    return rows


def candidate_ablation(base_cfg, steps: int) -> list[dict]:
    rows = []
    variants = []
    for K in sorted(set([base_cfg.lwe.h, base_cfg.lwe.h + 1, base_cfg.candidate.topK, min(base_cfg.lwe.n, base_cfg.candidate.topK + 4)])):
        variants.append((f"K={K}", {"topK": K, "use_pair_filter": False, "pair_score_weight": 0.0, "posterior_weight": 0.0}))
    if base_cfg.lwe.h >= 2:
        variants.append(("pair_filter", {"topK": base_cfg.candidate.topK, "use_pair_filter": True, "pair_budget": base_cfg.candidate.pair_budget}))
    variants.append(("posterior_tiebreak", {"topK": base_cfg.candidate.topK, "posterior_weight": 1.0}))
    for name, changes in variants:
        cfg = copy.deepcopy(base_cfg)
        cfg.run_name = f"ablation_candidate_{name.replace('=', '')}"
        cfg.train.steps = steps
        for key, value in changes.items():
            setattr(cfg.candidate, key, value)
        run_dir, best = train(cfg)
        rows.append({"ablation": "candidate", "variant": name, "run_dir": str(run_dir), **best})
    return rows


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="RHIE-CG ablations"))
    parser.add_argument("--experiment", default="feature", choices=["feature", "model", "candidate"])
    args = parser.parse_args()
    cfg = build_config(args, secret_type="binary")
    if args.experiment == "feature":
        rows = feature_ablation(cfg, args.steps)
    elif args.experiment == "model":
        rows = model_ablation(cfg, args.steps)
    else:
        rows = candidate_ablation(cfg, args.steps)
    write_csv(f"results/tables/ablation_{args.experiment}.csv", rows)
    print(rows)


if __name__ == "__main__":
    main()
