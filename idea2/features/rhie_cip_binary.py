from __future__ import annotations

import torch

from features.interaction import interaction_channels
from modular import centered_float, multi_phase_channels


def encode_binary(
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
    b_exp = b.unsqueeze(-1).expand_as(A)
    channels: list[torch.Tensor] = []

    if include_raw:
        A_c = centered_float(A, q)
        b_c = centered_float(b_exp, q)
        channels.extend([A_c, b_c, (A_c - b_c).abs(), A_c * b_c])

    if include_phase:
        channels.extend(multi_phase_channels(A, q, freqs))
        channels.extend(multi_phase_channels(b_exp, q, freqs))

    if include_rhie:
        r_plus = torch.remainder(b_exp - A, q)
        r_c = centered_float(r_plus, q)
        channels.extend([r_c, r_c.abs(), r_c.square()])
        channels.extend(multi_phase_channels(r_plus, q, freqs))

    if include_interaction:
        channels.extend(interaction_channels(A, b, q, include_stats=include_stats, freqs=freqs))

    if not channels:
        raise ValueError("at least one feature view must be enabled")
    return torch.stack(channels, dim=1)

