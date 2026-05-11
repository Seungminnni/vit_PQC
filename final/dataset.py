import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


def centered_mod(x: torch.Tensor, q: int) -> torch.Tensor:
    half = q // 2
    return torch.remainder(x + half, q) - half


def _sample_fixed_h_support(num_samples: int, n: int, fixed_h: int, generator: torch.Generator) -> torch.Tensor:
    if fixed_h < 0 or fixed_h > n:
        raise ValueError(f"fixed_h must be in [0, n], got fixed_h={fixed_h}, n={n}")
    if fixed_h == 0:
        return torch.zeros((num_samples, n), dtype=torch.bool)

    scores = torch.rand((num_samples, n), generator=generator)
    indices = torch.topk(scores, k=fixed_h, dim=1).indices
    support = torch.zeros((num_samples, n), dtype=torch.bool)
    support.scatter_(1, indices, True)
    return support


def _sample_variable_h_support(
    num_samples: int,
    n: int,
    h_min: int,
    h_max: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if h_min < 0 or h_max < h_min or h_max > n:
        raise ValueError(f"Expected 0 <= h_min <= h_max <= n, got h_min={h_min}, h_max={h_max}, n={n}")

    if h_min == h_max:
        h_values = torch.full((num_samples,), h_min, dtype=torch.int64)
    else:
        h_values = torch.randint(h_min, h_max + 1, (num_samples,), generator=generator, dtype=torch.int64)

    scores = torch.rand((num_samples, n), generator=generator)
    sorted_indices = scores.argsort(dim=1, descending=True)
    rank = sorted_indices.argsort(dim=1)
    support = rank < h_values.unsqueeze(1)
    return support, h_values


def sample_support_mask(
    num_samples: int,
    n: int,
    h_setting: str,
    generator: torch.Generator,
    p_nonzero: float | None = None,
    fixed_h: int | None = None,
    h_min: int | None = None,
    h_max: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if h_setting == "bernoulli":
        if p_nonzero is None:
            raise ValueError("p_nonzero is required when h_setting='bernoulli'")
        if not 0.0 <= p_nonzero <= 1.0:
            raise ValueError(f"p_nonzero must be in [0, 1], got {p_nonzero}")
        support = torch.rand((num_samples, n), generator=generator) < p_nonzero
        return support, support.sum(dim=1).to(torch.int64)

    if h_setting == "fixed_h":
        if fixed_h is None:
            raise ValueError("fixed_h is required when h_setting='fixed_h'")
        support = _sample_fixed_h_support(num_samples, n, fixed_h, generator)
        h_values = torch.full((num_samples,), fixed_h, dtype=torch.int64)
        return support, h_values

    if h_setting == "variable_h":
        if h_min is None or h_max is None:
            raise ValueError("h_min and h_max are required when h_setting='variable_h'")
        return _sample_variable_h_support(num_samples, n, h_min, h_max, generator)

    raise ValueError(f"Unknown h_setting: {h_setting}")


def sample_binary_secret(
    num_samples: int,
    n: int,
    h_setting: str,
    generator: torch.Generator,
    p_nonzero: float | None = None,
    fixed_h: int | None = None,
    h_min: int | None = None,
    h_max: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    support, h_values = sample_support_mask(
        num_samples=num_samples,
        n=n,
        h_setting=h_setting,
        generator=generator,
        p_nonzero=p_nonzero,
        fixed_h=fixed_h,
        h_min=h_min,
        h_max=h_max,
    )
    return support.to(torch.int64), h_values


def _bounded_integer_radius_from_sigma(sigma: float) -> int:
    if sigma <= 0:
        return 0
    radius = int(round((math.sqrt(1.0 + 12.0 * sigma * sigma) - 1.0) / 2.0))
    return max(radius, 1)


def make_raw_lwe_image(A: torch.Tensor, b: torch.Tensor, q: int) -> torch.Tensor:
    """Build a single-channel raw [A | b] image scaled to [0, 1)."""
    squeeze_batch = False
    if A.dim() == 2:
        A = A.unsqueeze(0)
        b = b.unsqueeze(0)
        squeeze_batch = True

    if A.dim() != 3 or b.dim() != 2:
        raise ValueError(f"Expected A=[B,m,n] and b=[B,m], got A={A.shape}, b={b.shape}")

    matrix = torch.cat([A.to(torch.float32), b.unsqueeze(-1).to(torch.float32)], dim=2)
    image = (matrix / float(q)).unsqueeze(1)
    return image.squeeze(0) if squeeze_batch else image


@dataclass(frozen=True)
class SyntheticLWESpec:
    num_samples: int
    m: int
    n: int
    q: int
    h_setting: str
    p_nonzero: float | None
    fixed_h: int | None
    h_min: int | None
    h_max: int | None
    sigma: float
    noise_distribution: str
    noise_bound: int | None
    shared_a: bool
    seed: int


class SyntheticLWERawImageDataset(Dataset):
    """Leakage-free synthetic binary LWE dataset using raw [A | b] images.

    This follows the sample-generation path from lwe_image_experiment but keeps
    the final/ input representation as one channel: [A | b] / q.
    """

    def __init__(self, spec: SyntheticLWESpec):
        self.spec = spec
        self.num_samples = spec.num_samples
        self.m = spec.m
        self.n = spec.n
        self.q = spec.q
        self.shared_a = spec.shared_a
        self.in_channels = 1
        self.num_classes = 2
        self.A: torch.Tensor | None = None
        self.A_shared: torch.Tensor | None = None
        self.b: torch.Tensor | None = None
        self.noise: torch.Tensor | None = None

        generator = torch.Generator().manual_seed(spec.seed)
        self.secret, self.h_values = self._generate_secret(generator)
        self._generate_public_instance(generator)

    def _generate_secret(self, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        secret, h_values = sample_binary_secret(
            num_samples=self.num_samples,
            n=self.n,
            h_setting=self.spec.h_setting,
            generator=generator,
            p_nonzero=self.spec.p_nonzero,
            fixed_h=self.spec.fixed_h,
            h_min=self.spec.h_min,
            h_max=self.spec.h_max,
        )
        return secret.to(torch.int8), h_values.to(torch.int16)

    def _generate_noise(self, generator: torch.Generator, num_samples: int | None = None) -> torch.Tensor:
        sample_count = self.num_samples if num_samples is None else num_samples
        sigma = float(self.spec.sigma)
        if sigma == 0.0:
            return torch.zeros((sample_count, self.m), dtype=torch.int32)

        if self.spec.noise_distribution == "discrete_gaussian":
            noise = torch.round(torch.randn((sample_count, self.m), generator=generator) * sigma)
            return noise.to(torch.int32)

        if self.spec.noise_distribution == "bounded_integer":
            radius = self.spec.noise_bound
            if radius is None:
                radius = _bounded_integer_radius_from_sigma(sigma)
            noise = torch.randint(
                low=-radius,
                high=radius + 1,
                size=(sample_count, self.m),
                generator=generator,
                dtype=torch.int64,
            )
            return noise.to(torch.int32)

        raise ValueError(f"Unknown noise_distribution: {self.spec.noise_distribution}")

    def _generate_public_instance(self, generator: torch.Generator) -> None:
        secret_int = self.secret.to(torch.int64)
        self.noise = self._generate_noise(generator)
        noise_int = self.noise.to(torch.int64)

        if self.shared_a:
            self.A_shared = torch.randint(
                low=0,
                high=self.q,
                size=(self.m, self.n),
                generator=generator,
                dtype=torch.int64,
            ).to(torch.int32)
            prod = torch.einsum("mn,bn->bm", self.A_shared.to(torch.int64), secret_int)
        else:
            self.A = torch.randint(
                low=0,
                high=self.q,
                size=(self.num_samples, self.m, self.n),
                generator=generator,
                dtype=torch.int64,
            ).to(torch.int32)
            prod = torch.matmul(self.A.to(torch.int64), secret_int.unsqueeze(-1)).squeeze(-1)

        self.b = torch.remainder(prod + noise_int, self.q).to(torch.int32)

    def __len__(self) -> int:
        return self.num_samples

    def _get_A(self, idx: int) -> torch.Tensor:
        if self.A_shared is not None:
            return self.A_shared
        if self.A is None:
            raise RuntimeError("Expected per-sample A to be initialized.")
        return self.A[idx]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.b is None or self.noise is None:
            raise RuntimeError("Expected public LWE instance to be initialized.")

        A = self._get_A(idx)
        b = self.b[idx]
        secret = self.secret[idx].to(torch.int64)
        return {
            "image": make_raw_lwe_image(A, b, self.q),
            "secret": secret,
            "target": secret,
            "A": A.to(torch.int64),
            "b": b.to(torch.int64),
            "noise": self.noise[idx].to(torch.int64),
            "h": self.h_values[idx].to(torch.int64),
        }


def apply_row_permutation(dataset: SyntheticLWERawImageDataset, mode: str, seed: int) -> None:
    if mode == "none":
        return

    if dataset.b is None or dataset.noise is None:
        raise RuntimeError("Expected public LWE instance to be initialized.")

    generator = torch.Generator().manual_seed(seed)
    if mode == "global":
        permutation = torch.randperm(dataset.m, generator=generator)
        if dataset.A_shared is not None:
            dataset.A_shared = dataset.A_shared[permutation]
        elif dataset.A is not None:
            dataset.A = dataset.A[:, permutation, :]
        else:
            raise RuntimeError("Expected A or A_shared to be initialized.")
        dataset.b = dataset.b[:, permutation]
        dataset.noise = dataset.noise[:, permutation]
        return

    if mode == "per_sample":
        if dataset.A_shared is not None:
            raise ValueError("row_permutation='per_sample' is not supported for shared-A datasets")
        if dataset.A is None:
            raise RuntimeError("Expected per-sample A tensor to be initialized.")
        scores = torch.rand((dataset.num_samples, dataset.m), generator=generator)
        permutations = scores.argsort(dim=1)
        row_index = torch.arange(dataset.num_samples).unsqueeze(1)
        dataset.A = dataset.A[row_index, permutations, :]
        dataset.b = dataset.b[row_index, permutations]
        dataset.noise = dataset.noise[row_index, permutations]
        return

    raise ValueError(f"Unknown row_permutation: {mode}")


def build_synthetic_lwe_datasets(args, run_seed: int) -> tuple[SyntheticLWERawImageDataset, SyntheticLWERawImageDataset, SyntheticLWERawImageDataset]:
    def make_spec(num_samples: int, split_offset: int) -> SyntheticLWESpec:
        return SyntheticLWESpec(
            num_samples=num_samples,
            m=args.m,
            n=args.n,
            q=args.q,
            h_setting=args.h_setting,
            p_nonzero=args.p_nonzero,
            fixed_h=args.fixed_h,
            h_min=args.h_min,
            h_max=args.h_max,
            sigma=args.sigma,
            noise_distribution=args.noise_distribution,
            noise_bound=args.noise_bound,
            shared_a=args.shared_a,
            seed=run_seed * 1000 + split_offset,
        )

    datasets = (
        SyntheticLWERawImageDataset(make_spec(args.num_train, 11)),
        SyntheticLWERawImageDataset(make_spec(args.num_val, 23)),
        SyntheticLWERawImageDataset(make_spec(args.num_test, 37)),
    )
    for dataset, split_offset in zip(datasets, (11, 23, 37)):
        apply_row_permutation(dataset, mode=args.row_permutation, seed=run_seed * 1000 + split_offset + 101)
    return datasets
