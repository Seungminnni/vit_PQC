from __future__ import annotations

import torch

from modular import centered_float, multi_phase_channels


def encode(A: torch.Tensor, b: torch.Tensor, q: int, freqs: tuple[int, ...] = (1,)) -> torch.Tensor:
    b_exp = b.unsqueeze(-1).expand_as(A)
    A_c = centered_float(A, q)
    b_c = centered_float(b_exp, q)
    A_sin, A_cos = multi_phase_channels(A, q, (freqs[0],))
    b_sin, b_cos = multi_phase_channels(b_exp, q, (freqs[0],))
    prod = A_c * b_c
    abs_diff = torch.abs(A_c - b_c)
    channels = [A_c, b_c, A_sin, A_cos, b_sin, b_cos, prod, abs_diff]
    return torch.stack(channels, dim=1)

