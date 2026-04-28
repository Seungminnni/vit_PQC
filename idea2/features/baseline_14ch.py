from __future__ import annotations

import torch

from features.baseline_10ch import encode as encode10
from features.interaction import interaction_channels


def encode(A: torch.Tensor, b: torch.Tensor, q: int, freqs: tuple[int, ...] = (1,)) -> torch.Tensor:
    base = encode10(A, b, q, freqs=freqs)
    inter = interaction_channels(A, b, q, include_stats=True, freqs=(freqs[0],))
    extra = torch.stack(inter[:4], dim=1)
    return torch.cat([base, extra], dim=1)

