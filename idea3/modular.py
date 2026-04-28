from __future__ import annotations

import math

import torch


def mod_q(x: torch.Tensor, q: int) -> torch.Tensor:
    return torch.remainder(x, q)


def centered_int(x: torch.Tensor, q: int) -> torch.Tensor:
    return torch.remainder(x + q // 2, q) - q // 2


def centered_float(x: torch.Tensor, q: int, scale: str = "half") -> torch.Tensor:
    denom = q / 2 if scale == "half" else q
    return centered_int(x, q).float() / float(denom)


def phase(x: torch.Tensor, q: int, freq: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 2.0 * math.pi * float(freq) * x.float() / float(q)
    return torch.sin(theta), torch.cos(theta)


def multi_phase_channels(x: torch.Tensor, q: int, freqs: tuple[int, ...] = (1, 2, 4)) -> list[torch.Tensor]:
    channels: list[torch.Tensor] = []
    for freq in freqs:
        sin_x, cos_x = phase(x, q, freq)
        channels.extend([sin_x, cos_x])
    return channels


def circular_distance_loss(residual_mod_q: torch.Tensor, q: int) -> torch.Tensor:
    theta = 2.0 * math.pi * residual_mod_q.float() / float(q)
    return (1.0 - torch.cos(theta)).mean()


def centered_energy(residual_mod_q: torch.Tensor, q: int, score_type: str = "squared", sigma: float | None = None) -> torch.Tensor:
    r = centered_int(residual_mod_q, q).float()
    if score_type == "absolute":
        return r.abs().sum(dim=-1)
    if score_type == "gaussian":
        sigma_eff = max(float(sigma or 1.0), 1e-6)
        return (r.square() / (2.0 * sigma_eff * sigma_eff)).sum(dim=-1)
    return r.square().sum(dim=-1)


def lwe_batch_dot(A: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Compute batched A@s without CUDA integer matmul/einsum kernels."""
    return (A * s.unsqueeze(1)).sum(dim=-1)


def lwe_dot(A: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Compute A@s without CUDA integer matmul kernels."""
    return (A * s.unsqueeze(0)).sum(dim=-1)
