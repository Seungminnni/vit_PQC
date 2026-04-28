from __future__ import annotations

import torch


def sample_sparse_binary_secret(
    batch_size: int,
    n: int,
    h: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample sparse binary secrets in {0,1}^n with Hamming weight h."""
    if not 0 <= h <= n:
        raise ValueError(f"h must satisfy 0 <= h <= n, got h={h}, n={n}")
    s = torch.zeros(batch_size, n, dtype=torch.long, device=device)
    for row in range(batch_size):
        idx = torch.randperm(n, device=device)[:h]
        s[row, idx] = 1
    return s


def sample_sparse_ternary_secret(
    batch_size: int,
    n: int,
    h: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample sparse ternary secrets in {-1,0,1}^n with Hamming weight h."""
    if not 0 <= h <= n:
        raise ValueError(f"h must satisfy 0 <= h <= n, got h={h}, n={n}")
    s = torch.zeros(batch_size, n, dtype=torch.long, device=device)
    for row in range(batch_size):
        idx = torch.randperm(n, device=device)[:h]
        signs = torch.randint(0, 2, (h,), device=device, dtype=torch.long) * 2 - 1
        s[row, idx] = signs
    return s


def sample_secret(
    batch_size: int,
    n: int,
    h: int,
    secret_type: str,
    device: torch.device | str,
) -> torch.Tensor:
    if secret_type == "binary":
        return sample_sparse_binary_secret(batch_size, n, h, device)
    if secret_type == "ternary":
        return sample_sparse_ternary_secret(batch_size, n, h, device)
    raise ValueError(f"Unsupported secret_type: {secret_type}")

