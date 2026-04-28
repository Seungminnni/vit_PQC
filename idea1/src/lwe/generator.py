from __future__ import annotations

import torch

from .modular import center_mod
from .noise import sample_error
from .secret import sample_secret


def generate_lwe_batch(
    batch_size: int,
    n: int,
    m: int,
    q: int,
    h: int,
    noise_bound: int,
    device: torch.device | str,
    secret_type: str = "binary",
    noise_type: str = "uniform_small",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate batched raw LWE instances b = A s + e mod q."""
    A = torch.randint(0, q, (batch_size, m, n), dtype=torch.long, device=device)
    s = sample_secret(batch_size, n, h, secret_type=secret_type, device=device)
    e = sample_error(batch_size, m, noise_type=noise_type, noise_bound=noise_bound, device=device)
    As = torch.einsum("bmn,bn->bm", A, s)
    b = torch.remainder(As + e, q)
    return A, b, s, e


@torch.no_grad()
def generator_sanity_check(
    batch_size: int,
    n: int,
    m: int,
    q: int,
    h: int,
    noise_bound: int,
    device: torch.device | str,
    secret_type: str = "binary",
    noise_type: str = "uniform_small",
) -> dict[str, float]:
    A, b, s, e = generate_lwe_batch(
        batch_size=batch_size,
        n=n,
        m=m,
        q=q,
        h=h,
        noise_bound=noise_bound,
        device=device,
        secret_type=secret_type,
        noise_type=noise_type,
    )
    As = torch.einsum("bmn,bn->bm", A, s)
    residual_center = center_mod(b - As, q)
    max_abs = float(residual_center.abs().max().item())
    return {
        "max_abs_centered_residual": max_abs,
        "mean_abs_centered_residual": float(residual_center.float().abs().mean().item()),
        "expected_noise_bound": float(noise_bound),
        "passed": float(max_abs <= noise_bound),
        "mean_abs_error": float(e.float().abs().mean().item()),
    }

