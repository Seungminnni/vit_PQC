from __future__ import annotations

from dataclasses import dataclass
import itertools
import random

import torch

from modular import lwe_batch_dot


@dataclass
class LWEBatch:
    A: torch.Tensor
    b: torch.Tensor
    s: torch.Tensor
    e: torch.Tensor
    q: int

    @property
    def support(self) -> torch.Tensor:
        return (self.s != 0).float()


def sample_sparse_binary(batch_size: int, n: int, h: int, device) -> torch.Tensor:
    s = torch.zeros(batch_size, n, dtype=torch.long, device=device)
    for row in range(batch_size):
        idx = torch.randperm(n, device=device)[:h]
        s[row, idx] = 1
    return s


def all_binary_supports(n: int, h: int) -> list[tuple[int, ...]]:
    return [tuple(int(x) for x in support) for support in itertools.combinations(range(n), h)]


def split_binary_supports(
    n: int,
    h: int,
    train_fraction: float = 0.8,
    seed: int = 1234,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    supports = all_binary_supports(n, h)
    rng = random.Random(seed)
    rng.shuffle(supports)
    cut = max(1, min(len(supports) - 1, int(round(len(supports) * train_fraction))))
    return supports[:cut], supports[cut:]


def sample_sparse_binary_from_support_pool(
    batch_size: int,
    n: int,
    support_pool: list[tuple[int, ...]],
    device,
) -> torch.Tensor:
    if not support_pool:
        raise ValueError("support_pool is empty")
    choices = torch.randint(0, len(support_pool), (batch_size,), device=device)
    s = torch.zeros(batch_size, n, dtype=torch.long, device=device)
    for row, pool_idx in enumerate(choices.tolist()):
        s[row, list(support_pool[pool_idx])] = 1
    return s


def sample_sparse_ternary(batch_size: int, n: int, h: int, device) -> torch.Tensor:
    s = torch.zeros(batch_size, n, dtype=torch.long, device=device)
    for row in range(batch_size):
        idx = torch.randperm(n, device=device)[:h]
        signs = torch.randint(0, 2, (h,), dtype=torch.long, device=device) * 2 - 1
        s[row, idx] = signs
    return s


def sample_sparse_integer(batch_size: int, n: int, h: int, values: tuple[int, ...], device) -> torch.Tensor:
    if 0 in values:
        raise ValueError("integer secret candidate values must exclude zero")
    value_tensor = torch.tensor(values, dtype=torch.long, device=device)
    s = torch.zeros(batch_size, n, dtype=torch.long, device=device)
    for row in range(batch_size):
        idx = torch.randperm(n, device=device)[:h]
        chosen = value_tensor[torch.randint(0, len(values), (h,), device=device)]
        s[row, idx] = chosen
    return s


def sample_secret(batch_size: int, n: int, h: int, secret_type: str, integer_values: tuple[int, ...], device) -> torch.Tensor:
    if secret_type == "binary":
        return sample_sparse_binary(batch_size, n, h, device)
    if secret_type == "ternary":
        return sample_sparse_ternary(batch_size, n, h, device)
    if secret_type == "integer":
        return sample_sparse_integer(batch_size, n, h, integer_values, device)
    raise ValueError(f"unsupported secret_type={secret_type}")


def sample_noise(batch_size: int, M: int, sigma_e: float, noise_type: str, device) -> torch.Tensor:
    if sigma_e == 0:
        return torch.zeros(batch_size, M, dtype=torch.long, device=device)
    if noise_type == "rounded_gaussian":
        return torch.round(torch.randn(batch_size, M, device=device) * sigma_e).long()
    if noise_type == "uniform_small":
        bound = int(round(sigma_e))
        return torch.randint(-bound, bound + 1, (batch_size, M), dtype=torch.long, device=device)
    raise ValueError(f"unsupported noise_type={noise_type}")


def generate_lwe_batch(
    batch_size: int,
    n: int,
    M: int,
    q: int,
    h: int,
    sigma_e: float,
    device,
    secret_type: str = "binary",
    noise_type: str = "rounded_gaussian",
    integer_values: tuple[int, ...] = (-3, -2, -1, 1, 2, 3),
    binary_support_pool: list[tuple[int, ...]] | None = None,
) -> LWEBatch:
    A = torch.randint(0, q, (batch_size, M, n), dtype=torch.long, device=device)
    if binary_support_pool is not None:
        if secret_type != "binary":
            raise ValueError("binary_support_pool is only valid for binary secrets")
        s = sample_sparse_binary_from_support_pool(batch_size, n, binary_support_pool, device)
    else:
        s = sample_secret(batch_size, n, h, secret_type, integer_values, device)
    e = sample_noise(batch_size, M, sigma_e, noise_type, device)
    As = lwe_batch_dot(A, s)
    b = torch.remainder(As + e, q)
    return LWEBatch(A=A, b=b, s=s, e=e, q=q)


def column_permute_batch(batch: LWEBatch) -> tuple[LWEBatch, torch.Tensor]:
    n = batch.A.shape[-1]
    perm = torch.randperm(n, device=batch.A.device)
    return LWEBatch(A=batch.A[:, :, perm], b=batch.b, s=batch.s[:, perm], e=batch.e, q=batch.q), perm


def b_shuffle_batch(batch: LWEBatch) -> LWEBatch:
    perm = torch.randperm(batch.b.shape[0], device=batch.b.device)
    return LWEBatch(A=batch.A, b=batch.b[perm], s=batch.s, e=batch.e[perm], q=batch.q)
