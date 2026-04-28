from __future__ import annotations

import torch

from modular import centered_float, multi_phase_channels


def encode(A: torch.Tensor, b: torch.Tensor, q: int, freqs: tuple[int, ...] = (1, 2, 4)) -> torch.Tensor:
    """A-only negative control. The public response b is intentionally unused."""
    A_c = centered_float(A, q)
    channels = [A_c, A_c.abs(), A_c.square()]
    channels.extend(multi_phase_channels(A, q, freqs))
    return torch.stack(channels, dim=1)

