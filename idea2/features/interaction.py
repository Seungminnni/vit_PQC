from __future__ import annotations

import torch

from modular import centered_float, centered_int, multi_phase_channels


def interaction_channels(
    A: torch.Tensor,
    b: torch.Tensor,
    q: int,
    include_stats: bool = True,
    freqs: tuple[int, ...] = (1, 2, 4),
) -> list[torch.Tensor]:
    A_c = centered_float(A, q)
    b_exp = b.unsqueeze(-1).expand_as(A)
    b_c = centered_float(b_exp, q)
    channels = [A_c * b_c]

    diff = torch.remainder(b_exp - A, q)
    for sin_d, cos_d in [tuple(multi_phase_channels(diff, q, (freq,))) for freq in freqs]:
        channels.extend([sin_d, cos_d])

    if include_stats:
        c = (centered_int(A, q).float() * centered_int(b_exp, q).float()).mean(dim=1)
        col_mean = A_c.mean(dim=1)
        col_var = A_c.var(dim=1, unbiased=False)
        col_energy = A_c.square().mean(dim=1)
        for vec in (c / max(q, 1), col_mean, col_var, col_energy):
            channels.append(vec.unsqueeze(1).expand(-1, A.shape[1], -1))
    return channels

