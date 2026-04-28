from __future__ import annotations

import math

import torch


def phase_image(
    A: torch.Tensor,
    b: torch.Tensor,
    q: int,
    freqs: tuple[int, ...] | list[int] = (1, 2, 4),
) -> torch.Tensor:
    """Map modular values to multi-frequency cos/sin channels."""
    x = torch.cat([A, b.unsqueeze(-1)], dim=-1).float()
    channels = []
    for freq in freqs:
        theta = 2.0 * math.pi * float(freq) * x / float(q)
        channels.append(torch.cos(theta))
        channels.append(torch.sin(theta))
    return torch.stack(channels, dim=1)

