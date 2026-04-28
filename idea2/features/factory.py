from __future__ import annotations

import torch

from features import a_only, baseline_10ch, baseline_14ch, baseline_8ch
from features.rhie_cip_binary import encode_binary
from features.rhie_cip_integer import encode_integer
from features.rhie_cip_ternary import encode_ternary


def encode_features(batch, feature_cfg, integer_values: tuple[int, ...] = (-3, -2, -1, 1, 2, 3)) -> torch.Tensor:
    A, b, q = batch.A, batch.b, batch.q
    kwargs = {
        "freqs": feature_cfg.freqs,
        "include_raw": feature_cfg.include_raw,
        "include_phase": feature_cfg.include_phase,
        "include_rhie": feature_cfg.include_rhie,
        "include_interaction": feature_cfg.include_interaction,
        "include_stats": feature_cfg.include_stats,
    }
    if feature_cfg.encoder_type == "baseline_8ch":
        return baseline_8ch.encode(A, b, q, freqs=feature_cfg.freqs[:1])
    if feature_cfg.encoder_type == "baseline_10ch":
        return baseline_10ch.encode(A, b, q, freqs=feature_cfg.freqs[:1])
    if feature_cfg.encoder_type == "baseline_14ch":
        return baseline_14ch.encode(A, b, q, freqs=feature_cfg.freqs[:1])
    if feature_cfg.encoder_type == "a_only":
        return a_only.encode(A, b, q, freqs=feature_cfg.freqs)
    if feature_cfg.encoder_type == "rhie_cip_ternary":
        return encode_ternary(A, b, q, **kwargs)
    if feature_cfg.encoder_type == "rhie_cip_integer":
        return encode_integer(A, b, q, values=integer_values, **kwargs)
    if feature_cfg.encoder_type == "rhie_cip":
        if batch.s.dtype == torch.long and batch.s.min() >= 0 and batch.s.max() <= 1:
            return encode_binary(A, b, q, **kwargs)
        if batch.s.min() < 0 and batch.s.max() <= 1:
            return encode_ternary(A, b, q, **kwargs)
        if batch.s.abs().max() > 1:
            return encode_integer(A, b, q, values=integer_values, **kwargs)
        return encode_binary(A, b, q, **kwargs)
    raise ValueError(f"unsupported encoder_type={feature_cfg.encoder_type}")
