from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .lwe import LWEParams, LWESample, secret_value_table
from .representations import LWEImageEncoder, RepresentationConfig


@dataclass(frozen=True)
class LWEDatasetSpec:
    num_samples: int
    params: LWEParams
    representation: RepresentationConfig
    h_setting: str = "iid"
    p_nonzero: float | None = None
    fixed_h: int | None = None
    h_min: int | None = None
    h_max: int | None = None

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self.h_setting not in {"iid", "fixed_h", "variable_h", "bernoulli"}:
            raise ValueError("h_setting must be one of: iid, fixed_h, variable_h, bernoulli.")
        if self.params.secret_dist != "binary" and self.h_setting != "iid":
            raise ValueError("Sparse h_setting options are currently supported only for binary secrets.")


class SyntheticLWEDataset(Dataset):
    def __init__(self, spec: LWEDatasetSpec) -> None:
        self.spec = spec
        self.params = spec.params
        self.encoder = LWEImageEncoder(spec.representation, q=self.params.q)
        self.in_channels = self.encoder.num_channels(n=self.params.n)
        self.sample = self._generate()

    def __len__(self) -> int:
        return self.spec.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image, mask = self.encoder.encode(self.sample.A[idx], self.sample.b[idx])
        return {
            "image": image.squeeze(0),
            "mask": mask.squeeze(0),
            "target": self.sample.s_labels[idx],
            "secret": self.sample.s[idx],
            "A": self.sample.A[idx],
            "b": self.sample.b[idx],
            "noise": self.sample.e[idx],
            "oracle_residual": self.sample.e[idx],
            "h": (self.sample.s[idx] != 0).sum().to(torch.long),
        }

    def _generate(self) -> LWESample:
        device = torch.device("cpu")
        gen = torch.Generator(device=device)
        if self.params.seed is not None:
            gen.manual_seed(self.params.seed)

        A = torch.randint(
            self.params.q,
            (self.spec.num_samples, self.params.m, self.params.n),
            generator=gen,
            dtype=torch.long,
            device=device,
        )
        s, s_labels = self._sample_secret(gen)
        e = self._sample_noise(gen)
        b = torch.remainder(torch.matmul(A, s.unsqueeze(-1)).squeeze(-1) + e, self.params.q).to(torch.long)
        return LWESample(A=A, b=b, s=s, e=e, s_labels=s_labels)

    def _sample_secret(self, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        if self.params.secret_dist == "binary" and self.spec.h_setting != "iid":
            support = self._sample_binary_support(gen)
            labels = support.to(torch.long)
            return labels.clone(), labels

        values = secret_value_table(self.params.q, self.params.secret_dist)
        labels = torch.randint(
            values.numel(),
            (self.spec.num_samples, self.params.n),
            generator=gen,
            dtype=torch.long,
        )
        return values[labels].to(torch.long), labels

    def _sample_binary_support(self, gen: torch.Generator) -> torch.Tensor:
        if self.spec.h_setting == "bernoulli":
            p_nonzero = self.spec.p_nonzero
            if p_nonzero is None:
                p_nonzero = min(1.0, 3.0 / max(self.params.n, 1))
            return torch.rand((self.spec.num_samples, self.params.n), generator=gen) < p_nonzero

        if self.spec.h_setting == "fixed_h":
            if self.spec.fixed_h is None:
                raise ValueError("fixed_h is required when h_setting='fixed_h'.")
            h_values = torch.full((self.spec.num_samples,), self.spec.fixed_h, dtype=torch.long)
        elif self.spec.h_setting == "variable_h":
            h_min = 1 if self.spec.h_min is None else self.spec.h_min
            h_max = max(2, self.params.n // 8) if self.spec.h_max is None else self.spec.h_max
            if h_min < 0 or h_max < h_min or h_max > self.params.n:
                raise ValueError("Expected 0 <= h_min <= h_max <= n.")
            h_values = torch.randint(h_min, h_max + 1, (self.spec.num_samples,), generator=gen)
        else:
            raise ValueError(f"Unsupported sparse h_setting: {self.spec.h_setting}")

        if h_values.max().item() > self.params.n or h_values.min().item() < 0:
            raise ValueError("h values must be in [0, n].")
        scores = torch.rand((self.spec.num_samples, self.params.n), generator=gen)
        rank = scores.argsort(dim=1, descending=True).argsort(dim=1)
        return rank < h_values.unsqueeze(1)

    def _sample_noise(self, gen: torch.Generator) -> torch.Tensor:
        shape = (self.spec.num_samples, self.params.m)
        if self.params.noise_dist == "zero":
            return torch.zeros(shape, dtype=torch.long)
        if self.params.noise_dist == "uniform_small":
            radius = int(round(self.params.noise_width))
            return torch.randint(-radius, radius + 1, shape, generator=gen, dtype=torch.long)
        if self.params.noise_dist == "discrete_gaussian":
            return torch.round(torch.randn(shape, generator=gen) * float(self.params.noise_width)).to(torch.long)
        raise ValueError(f"Unsupported noise_dist: {self.params.noise_dist}")


def dataset_statistics(dataset: SyntheticLWEDataset) -> dict[str, float]:
    sample = dataset.sample
    target = sample.s_labels.to(torch.long)
    secret = sample.s.to(torch.long)
    support = secret != 0
    h_values = support.sum(dim=1).to(torch.float32)
    num_classes = int(secret_value_table(dataset.params.q, dataset.params.secret_dist).numel())
    one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).to(torch.float32)
    per_coord_probs = one_hot.mean(dim=0)
    overall_probs = one_hot.mean(dim=(0, 1))
    random_coord_acc = per_coord_probs.pow(2).sum(dim=1).mean().item()
    random_exact_match = math.prod(per_coord_probs.pow(2).sum(dim=1).tolist())

    values = secret_value_table(dataset.params.q, dataset.params.secret_dist)
    zero_label_matches = (values == 0).nonzero(as_tuple=False)
    zero_label = int(zero_label_matches[0].item()) if zero_label_matches.numel() else 0
    zero_pred = torch.full_like(target, zero_label)

    stats = {
        "avg_h": h_values.mean().item(),
        "std_h": h_values.std(unbiased=False).item(),
        "nonzero_rate": support.to(torch.float32).mean().item(),
        "random_guess_coord_acc": random_coord_acc,
        "random_guess_exact_match": float(random_exact_match),
        "all_zero_coord_acc": (zero_pred == target).to(torch.float32).mean().item(),
        "all_zero_exact_match": (zero_pred == target).all(dim=1).to(torch.float32).mean().item(),
        "all_zero_active_recall": 0.0 if support.any().item() else 1.0,
    }
    for class_idx in range(num_classes):
        stats[f"class_prob_{class_idx}"] = overall_probs[class_idx].item()
    return stats
