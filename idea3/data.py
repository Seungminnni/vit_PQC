from __future__ import annotations

from dataclasses import dataclass

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
        return self.s.float()


def sample_fixed_h_binary(batch_size: int, n: int, h: int, device) -> torch.Tensor:
    h = max(0, min(int(h), n))
    s = torch.zeros(batch_size, n, dtype=torch.long, device=device)
    for row in range(batch_size):
        if h > 0:
            idx = torch.randperm(n, device=device)[:h]
            s[row, idx] = 1
    return s


def sample_h_range_binary(batch_size: int, n: int, h_min: int, h_max: int, device) -> torch.Tensor:
    h_min = max(0, min(int(h_min), n))
    h_max = max(h_min, min(int(h_max), n))
    hs = torch.randint(h_min, h_max + 1, (batch_size,), device=device)
    s = torch.zeros(batch_size, n, dtype=torch.long, device=device)
    for row, h in enumerate(hs.tolist()):
        if h > 0:
            idx = torch.randperm(n, device=device)[:h]
            s[row, idx] = 1
    return s


def sample_bernoulli_binary(batch_size: int, n: int, p_nonzero: float, device) -> torch.Tensor:
    p = min(max(float(p_nonzero), 0.0), 1.0)
    return (torch.rand(batch_size, n, device=device) < p).long()


def sample_binary_secret(
    batch_size: int,
    n: int,
    distribution: str,
    h: int,
    h_min: int,
    h_max: int,
    p_nonzero: float,
    device,
) -> torch.Tensor:
    if distribution == "fixed":
        return sample_fixed_h_binary(batch_size, n, h, device)
    if distribution == "h_range":
        return sample_h_range_binary(batch_size, n, h_min, h_max, device)
    if distribution == "bernoulli":
        return sample_bernoulli_binary(batch_size, n, p_nonzero, device)
    raise ValueError(f"unsupported binary secret distribution={distribution}")


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
    secret_distribution: str = "bernoulli",
    h_min: int = 2,
    h_max: int = 4,
    p_nonzero: float = 0.1875,
    noise_type: str = "rounded_gaussian",
) -> LWEBatch:
    A = torch.randint(0, q, (batch_size, M, n), dtype=torch.long, device=device)
    s = sample_binary_secret(batch_size, n, secret_distribution, h, h_min, h_max, p_nonzero, device)
    e = sample_noise(batch_size, M, sigma_e, noise_type, device)
    b = torch.remainder(lwe_batch_dot(A, s) + e, q)
    return LWEBatch(A=A, b=b, s=s, e=e, q=q)
