from __future__ import annotations

import torch

from features.baseline_8ch import encode as encode8
from modular import centered_float


def encode(A: torch.Tensor, b: torch.Tensor, q: int, freqs: tuple[int, ...] = (1,)) -> torch.Tensor:
    base = encode8(A, b, q, freqs=freqs)
    b_exp = b.unsqueeze(-1).expand_as(A)
    r_plus = torch.remainder(b_exp - A, q)
    r_center = centered_float(r_plus, q)
    extra = torch.stack([r_center, r_center.abs()], dim=1)
    return torch.cat([base, extra], dim=1)

