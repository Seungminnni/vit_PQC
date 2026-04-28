from __future__ import annotations

import torch

from features.rhie_cip_binary import encode_binary
from modular import centered_float, multi_phase_channels


def encode_ternary(
    A: torch.Tensor,
    b: torch.Tensor,
    q: int,
    freqs: tuple[int, ...] = (1, 2, 4),
    include_raw: bool = True,
    include_phase: bool = True,
    include_rhie: bool = True,
    include_interaction: bool = True,
    include_stats: bool = True,
) -> torch.Tensor:
    base = encode_binary(A, b, q, freqs, include_raw, include_phase, False, include_interaction, include_stats)
    if not include_rhie:
        return base
    b_exp = b.unsqueeze(-1).expand_as(A)
    channels = [base]
    for residual in (torch.remainder(b_exp - A, q), torch.remainder(b_exp + A, q)):
        r_c = centered_float(residual, q)
        view = [r_c, r_c.abs(), r_c.square(), *multi_phase_channels(residual, q, freqs)]
        channels.append(torch.stack(view, dim=1))
    return torch.cat(channels, dim=1)

