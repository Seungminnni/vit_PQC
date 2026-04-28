from __future__ import annotations

import argparse
import json
from pathlib import Path

from configs import add_common_args, build_config, config_from_dict
from data import generate_lwe_batch
from features.factory import encode_features
from utils import resolve_device, seed_everything
from visualization import save_feature_grid, save_feature_heatmaps


def rhie_cip_binary_channel_names(freqs: tuple[int, ...]) -> list[str]:
    names = ["A_center", "b_center", "|A-b|", "A*b"]
    for prefix in ("A", "b"):
        for freq in freqs:
            names.extend([f"{prefix}_sin_f{freq}", f"{prefix}_cos_f{freq}"])
    names.extend(["r=b-A center", "|r|", "r^2"])
    for freq in freqs:
        names.extend([f"r_sin_f{freq}", f"r_cos_f{freq}"])
    names.append("A*b centered")
    for freq in freqs:
        names.extend([f"(b-A)_sin_f{freq}", f"(b-A)_cos_f{freq}"])
    names.extend(["ATb_broadcast", "A_col_mean", "A_col_var", "A_col_energy"])
    return names


def load_cfg(args):
    if args.config_json:
        with Path(args.config_json).open("r", encoding="utf-8") as fp:
            return config_from_dict(json.load(fp))
    return build_config(args, secret_type="binary")


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Visualize binary RHIE-CG embeddings"))
    parser.add_argument("--config_json", default=None)
    parser.add_argument("--out_dir", default="results/figures/embeddings")
    parser.add_argument("--num_cases", type=int, default=1)
    parser.add_argument("--max_channels", type=int, default=36)
    args = parser.parse_args()

    cfg = load_cfg(args)
    cfg.train.device = args.device
    seed_everything(cfg.train.seed + 700)
    device = resolve_device(cfg.train.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = generate_lwe_batch(
        batch_size=args.num_cases,
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
    names = rhie_cip_binary_channel_names(cfg.features.freqs) if cfg.features.encoder_type == "rhie_cip" else None
    summary = []
    for row in range(args.num_cases):
        support = tuple(int(x) for x in (batch.s[row] != 0).nonzero().flatten().tolist())
        prefix = out_dir / f"case_{row:03d}"
        save_feature_grid(
            X[row],
            prefix.with_name(prefix.name + "_grid.png"),
            channel_names=names,
            max_channels=args.max_channels,
            support=support,
        )
        save_feature_heatmaps(X[row], prefix.with_name(prefix.name + "_channel"), max_channels=args.max_channels)
        summary.append({"case": row, "support": support, "shape": tuple(X[row].shape)})
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print({"out_dir": str(out_dir), "embedding_shape": tuple(X.shape), "cases": summary})


if __name__ == "__main__":
    main()
