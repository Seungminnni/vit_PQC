from __future__ import annotations

import torch

from configs import PRESETS, FeatureConfig, LWEConfig, ModelConfig, TrainConfig, CandidateConfig, ExperimentConfig
from data import b_shuffle_batch, column_permute_batch, generate_lwe_batch
from features.factory import encode_features
from modular import centered_int, lwe_batch_dot
from models.full_model import RHIECGModel
from utils import resolve_device, seed_everything


def check_generator(device) -> dict[str, float]:
    batch = generate_lwe_batch(8, n=8, M=32, q=127, h=2, sigma_e=1.0, device=device)
    residual = centered_int(batch.b - lwe_batch_dot(batch.A, batch.s), batch.q)
    return {
        "generator_max_abs_residual": float(residual.abs().max().item()),
        "generator_mean_abs_residual": float(residual.float().abs().mean().item()),
    }


def check_feature_shapes(device) -> dict[str, tuple[int, ...]]:
    out = {}
    batch = generate_lwe_batch(4, n=8, M=32, q=127, h=2, sigma_e=0.0, device=device)
    for encoder in ("a_only", "baseline_8ch", "baseline_10ch", "baseline_14ch", "rhie_cip"):
        cfg = FeatureConfig(encoder_type=encoder)
        out[encoder] = tuple(encode_features(batch, cfg).shape)
    ternary = generate_lwe_batch(4, n=8, M=32, q=127, h=2, sigma_e=0.0, device=device, secret_type="ternary")
    out["rhie_cip_ternary"] = tuple(encode_features(ternary, FeatureConfig(encoder_type="rhie_cip_ternary")).shape)
    integer = generate_lwe_batch(4, n=8, M=32, q=127, h=2, sigma_e=0.0, device=device, secret_type="integer")
    out["rhie_cip_integer"] = tuple(encode_features(integer, FeatureConfig(encoder_type="rhie_cip_integer")).shape)
    return out


def check_model_forward(device) -> dict[str, tuple[int, ...]]:
    batch = generate_lwe_batch(2, n=8, M=32, q=127, h=2, sigma_e=0.0, device=device)
    feature_cfg = FeatureConfig(encoder_type="rhie_cip")
    X = encode_features(batch, feature_cfg)
    model = RHIECGModel(in_channels=X.shape[1], n=8, d_model=32, depth=1, heads=4, dropout=0.0).to(device)
    model.eval()
    out = model(X)
    return {"support_logits": tuple(out["support_logits"].shape), "tokens": tuple(out["tokens"].shape)}


def check_column_permutation_equivariance(device) -> float:
    batch = generate_lwe_batch(2, n=8, M=32, q=127, h=2, sigma_e=0.0, device=device)
    cfg = FeatureConfig(encoder_type="rhie_cip")
    X = encode_features(batch, cfg)
    model = RHIECGModel(in_channels=X.shape[1], n=8, d_model=32, depth=1, heads=4, dropout=0.0, use_position=False).to(device)
    model.eval()
    logits = model(X)["support_logits"]
    perm_batch, perm = column_permute_batch(batch)
    Xp = encode_features(perm_batch, cfg)
    logits_p = model(Xp)["support_logits"]
    return float((logits[:, perm] - logits_p).abs().max().item())


def check_b_shuffle_changes_features(device) -> float:
    batch = generate_lwe_batch(2, n=8, M=32, q=127, h=2, sigma_e=0.0, device=device)
    cfg = FeatureConfig(encoder_type="rhie_cip")
    X = encode_features(batch, cfg)
    X_bad = encode_features(b_shuffle_batch(batch), cfg)
    return float((X - X_bad).abs().mean().item())


def main() -> None:
    seed_everything(123)
    device = resolve_device("cuda")
    print({"generator": check_generator(device)})
    print({"feature_shapes": check_feature_shapes(device)})
    print({"model_forward": check_model_forward(device)})
    print({"column_permutation_max_error": check_column_permutation_equivariance(device)})
    print({"b_shuffle_mean_feature_delta": check_b_shuffle_changes_features(device)})


if __name__ == "__main__":
    main()
