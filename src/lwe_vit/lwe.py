from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class LWEParams:
    n: int
    m: int
    q: int
    secret_dist: str = "binary"
    noise_dist: str = "discrete_gaussian"
    seed: Optional[int] = None
    noise_width: float = 2.0

    def __post_init__(self) -> None:
        if self.n <= 0 or self.m <= 0:
            raise ValueError("n and m must be positive.")
        if self.q <= 2:
            raise ValueError("q must be greater than 2.")
        if self.secret_dist not in {"binary", "ternary", "uniform"}:
            raise ValueError("secret_dist must be one of: binary, ternary, uniform.")
        if self.noise_dist not in {"zero", "uniform_small", "discrete_gaussian"}:
            raise ValueError("noise_dist must be one of: zero, uniform_small, discrete_gaussian.")
        if self.noise_width < 0:
            raise ValueError("noise_width must be non-negative.")


@dataclass(frozen=True)
class LWESample:
    A: torch.Tensor
    b: torch.Tensor
    s: torch.Tensor
    e: torch.Tensor
    s_labels: torch.Tensor


def _generator(seed: Optional[int], device: torch.device | str) -> Optional[torch.Generator]:
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def centered_mod(x: torch.Tensor, q: int) -> torch.Tensor:
    """Map residues to the centered interval [-floor(q/2), ceil(q/2)-1]."""
    return torch.remainder(x + q // 2, q) - q // 2


def centered_mod_float(x: torch.Tensor, q: int) -> torch.Tensor:
    half_q = float(q) / 2.0
    return torch.remainder(x + half_q, float(q)) - half_q


def secret_value_table(q: int, secret_dist: str, device: torch.device | str | None = None) -> torch.Tensor:
    if secret_dist == "binary":
        return torch.tensor([0, 1], dtype=torch.long, device=device)
    if secret_dist == "ternary":
        return torch.tensor([q - 1, 0, 1], dtype=torch.long, device=device)
    if secret_dist == "uniform":
        return torch.arange(q, dtype=torch.long, device=device)
    raise ValueError(f"Unsupported secret_dist: {secret_dist}")


def num_secret_classes(params: LWEParams | str, q: Optional[int] = None) -> int:
    if isinstance(params, LWEParams):
        secret_dist = params.secret_dist
        q_value = params.q
    else:
        if q is None:
            raise ValueError("q is required when params is a secret distribution string.")
        secret_dist = params
        q_value = q
    return int(secret_value_table(q_value, secret_dist).numel())


def _sample_secret(
    params: LWEParams,
    batch_size: int,
    device: torch.device | str,
    gen: Optional[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor]:
    if params.secret_dist == "uniform":
        labels = torch.randint(params.q, (batch_size, params.n), generator=gen, device=device)
        return labels.to(torch.long), labels.to(torch.long)

    classes = num_secret_classes(params)
    labels = torch.randint(classes, (batch_size, params.n), generator=gen, device=device)
    values = secret_value_table(params.q, params.secret_dist, device=device)
    return values[labels].to(torch.long), labels.to(torch.long)


def _sample_noise(
    params: LWEParams,
    batch_size: int,
    device: torch.device | str,
    gen: Optional[torch.Generator],
) -> torch.Tensor:
    shape = (batch_size, params.m)
    if params.noise_dist == "zero":
        return torch.zeros(shape, dtype=torch.long, device=device)
    if params.noise_dist == "uniform_small":
        bound = int(round(params.noise_width))
        return torch.randint(-bound, bound + 1, shape, generator=gen, device=device).to(torch.long)
    noise = torch.randn(shape, generator=gen, device=device) * float(params.noise_width)
    return noise.round().to(torch.long)


def sample_lwe_batch(
    params: LWEParams,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> LWESample:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    gen = _generator(params.seed, device)
    A = torch.randint(
        params.q,
        (batch_size, params.m, params.n),
        generator=gen,
        device=device,
        dtype=torch.long,
    )
    s, s_labels = _sample_secret(params, batch_size, device, gen)
    e = _sample_noise(params, batch_size, device, gen)
    b = torch.remainder(torch.matmul(A, s.unsqueeze(-1)).squeeze(-1) + e, params.q).to(torch.long)
    return LWESample(A=A, b=b, s=s, e=e, s_labels=s_labels)


def residual_from_secret(A: torch.Tensor, b: torch.Tensor, s: torch.Tensor, q: int) -> torch.Tensor:
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if s.dim() == 1:
        s = s.unsqueeze(0)
    pred = (A.to(torch.long) * s.to(torch.long).unsqueeze(1)).sum(dim=-1)
    return centered_mod(b.to(torch.long) - pred, q)
