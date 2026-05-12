import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


def centered_mod(x: torch.Tensor, q: int) -> torch.Tensor:
    half = q // 2
    return torch.remainder(x + half, q) - half


def normalize_centered(x: torch.Tensor, q: int) -> torch.Tensor:
    scale = max(q / 2.0, 1.0)
    return centered_mod(x.to(torch.int64), q).to(torch.float32) / scale


def _sample_fixed_h_support(
    num_samples: int,
    n: int,
    fixed_h: int,
    generator: torch.Generator,
) -> torch.Tensor:
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
        fixed_h = torch.full((num_samples,), h_min, dtype=torch.int64)
    else:
        fixed_h = torch.randint(h_min, h_max + 1, (num_samples,), generator=generator, dtype=torch.int64)

    scores = torch.rand((num_samples, n), generator=generator)
    sorted_indices = scores.argsort(dim=1, descending=True)
    rank = sorted_indices.argsort(dim=1)
    support = rank < fixed_h.unsqueeze(1)
    return support, fixed_h


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
        support = torch.rand((num_samples, n), generator=generator) < p_nonzero
        h_values = support.sum(dim=1).to(torch.int64)
        return support, h_values

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


def make_global_lwe_image(A: torch.Tensor, b: torch.Tensor, q: int, encoding: str) -> torch.Tensor:
    if encoding != "phase6":
        raise ValueError(f"Only phase6 encoding is supported, got {encoding}")

    squeeze_batch = False
    if A.dim() == 2:
        A = A.unsqueeze(0)
        b = b.unsqueeze(0)
        squeeze_batch = True

    if A.dim() != 3 or b.dim() != 2:
        raise ValueError(f"Expected A=[B,m,n] and b=[B,m], got A={A.shape}, b={b.shape}")

    _, _, n = A.shape
    Bmat = b.unsqueeze(-1).expand(-1, -1, n)
    A_c = normalize_centered(A, q)
    B_c = normalize_centered(Bmat, q)
    A_f = A.to(torch.float32)
    B_f = Bmat.to(torch.float32)
    image = torch.stack(
        [
            A_c,
            B_c,
            torch.sin(2.0 * math.pi * A_f / q),
            torch.cos(2.0 * math.pi * A_f / q),
            torch.sin(2.0 * math.pi * B_f / q),
            torch.cos(2.0 * math.pi * B_f / q),
        ],
        dim=1,
    )

    return image.squeeze(0) if squeeze_batch else image


@dataclass(frozen=True)
class DatasetSpec:
    num_samples: int
    m: int
    n: int
    q: int
    secret_type: str
    h_setting: str
    p_nonzero: float | None
    fixed_h: int | None
    h_min: int | None
    h_max: int | None
    sigma: float
    noise_distribution: str
    noise_bound: int | None
    encoding: str
    shared_a: bool
    seed: int


class SyntheticLWEDataset(Dataset):
    MAX_MATERIALIZED_A_BYTES = 32 * 1024**3

    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        self.num_samples = spec.num_samples
        self.m = spec.m
        self.n = spec.n
        self.q = spec.q
        self.secret_type = spec.secret_type
        if self.secret_type != "binary":
            raise ValueError(f"Only binary secret_type is supported, got {self.secret_type}")
        self.encoding = spec.encoding
        self.shared_a = spec.shared_a
        self.in_channels = self._resolve_in_channels(spec.encoding)
        self.num_classes = 2
        self.materialize_public = self._should_materialize_public()
        self.A = None
        self.A_shared = None
        self.b = None
        self.noise = None
        self.oracle_residual = None

        generator = torch.Generator().manual_seed(spec.seed)
        self.secret, self.h_values = self._generate_secret(generator)
        self._generate_public_instance(generator)

    def _should_materialize_public(self) -> bool:
        if self.shared_a:
            return True
        estimated_a_bytes = self.num_samples * self.m * self.n * 4
        return estimated_a_bytes <= self.MAX_MATERIALIZED_A_BYTES

    @staticmethod
    def _resolve_in_channels(encoding: str) -> int:
        if encoding == "phase6":
            return 6
        raise ValueError(f"Only phase6 encoding is supported, got {encoding}")

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

        if self.shared_a:
            self.noise = self._generate_noise(generator)
            self.oracle_residual = self.noise.clone()
            noise_int = self.noise.to(torch.int64)
            self.A_shared = torch.randint(
                low=0,
                high=self.q,
                size=(self.m, self.n),
                generator=generator,
                dtype=torch.int64,
            ).to(torch.int32)
            prod = torch.einsum("mn,bn->bm", self.A_shared.to(torch.int64), secret_int)
        else:
            if not self.materialize_public:
                return
            self.noise = self._generate_noise(generator)
            self.oracle_residual = self.noise.clone()
            noise_int = self.noise.to(torch.int64)
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
            return self._generate_lazy_public(idx)[0]
        return self.A[idx]

    def _generate_lazy_public(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.spec.seed * 1_000_003 + idx)
        A = torch.randint(
            low=0,
            high=self.q,
            size=(self.m, self.n),
            generator=generator,
            dtype=torch.int64,
        ).to(torch.int32)
        noise = self._generate_noise(generator, num_samples=1).squeeze(0)
        secret = self.secret[idx].to(torch.int64)
        prod = torch.matmul(A.to(torch.int64), secret)
        b = torch.remainder(prod + noise.to(torch.int64), self.q).to(torch.int32)
        return A, b, noise

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.b is None:
            A, b, noise = self._generate_lazy_public(idx)
            oracle_residual = noise
        else:
            A = self._get_A(idx)
            b = self.b[idx]
            noise = self.noise[idx]
            oracle_residual = self.oracle_residual[idx]
        secret = self.secret[idx].to(torch.int64)

        item = {
            "secret": secret,
            "target": secret.to(torch.int64),
            "A": A.to(torch.int64),
            "b": b.to(torch.int64),
            "noise": noise.to(torch.int64),
            "oracle_residual": oracle_residual.to(torch.int64),
            "h": self.h_values[idx].to(torch.int64),
        }

        item["image"] = make_global_lwe_image(A, b, self.q, self.encoding)

        return item
