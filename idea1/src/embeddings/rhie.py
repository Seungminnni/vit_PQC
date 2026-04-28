from __future__ import annotations

import math

import torch

from src.lwe.modular import normalize_centered


def _residual_channels(
    R: torch.Tensor,
    q: int,
    freqs: tuple[int, ...] | list[int],
    include_magnitude: bool,
) -> list[torch.Tensor]:
    channels: list[torch.Tensor] = []
    if include_magnitude:
        R_center = normalize_centered(R, q)
        channels.append(R_center)
        channels.append(torch.abs(R_center))
    R_float = R.float()
    for freq in freqs:
        theta = 2.0 * math.pi * float(freq) * R_float / float(q)
        channels.append(torch.cos(theta))
        channels.append(torch.sin(theta))
    return channels


def rhie_binary(
    A: torch.Tensor,
    b: torch.Tensor,
    q: int,
    freqs: tuple[int, ...] | list[int] = (1, 2, 4),
    include_magnitude: bool = True,
) -> torch.Tensor:
    """Residual-hypothesis image for binary hypothesis s_j=1."""
    R = torch.remainder(b.unsqueeze(-1) - A, q)
    return torch.stack(_residual_channels(R, q, freqs, include_magnitude), dim=1)


def rhie_ternary(
    A: torch.Tensor,
    b: torch.Tensor,
    q: int,
    freqs: tuple[int, ...] | list[int] = (1, 2, 4),
    include_magnitude: bool = True,
) -> torch.Tensor:
    """Residual-hypothesis image for ternary hypotheses s_j=+1 and s_j=-1."""
    residuals = (torch.remainder(b.unsqueeze(-1) - A, q), torch.remainder(b.unsqueeze(-1) + A, q))
    channels: list[torch.Tensor] = []
    for R in residuals:
        channels.extend(_residual_channels(R, q, freqs, include_magnitude))
    return torch.stack(channels, dim=1)

