from __future__ import annotations

import torch


def sample_uniform_small_error(
    batch_size: int,
    m: int,
    bound: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample e_i uniformly from {-bound, ..., bound}."""
    if bound < 0:
        raise ValueError(f"noise bound must be non-negative, got {bound}")
    return torch.randint(-bound, bound + 1, (batch_size, m), device=device)


def sample_error(
    batch_size: int,
    m: int,
    noise_type: str,
    noise_bound: int,
    device: torch.device | str,
) -> torch.Tensor:
    if noise_type == "uniform_small":
        return sample_uniform_small_error(batch_size, m, noise_bound, device)
    raise ValueError(f"Unsupported noise_type: {noise_type}")

