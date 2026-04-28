from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.embeddings.packet import build_crypto_image_packet
from src.lwe.generator import generate_lwe_batch


@dataclass
class OnTheFlyLWEDataset:
    n: int
    m: int
    q: int
    h: int
    noise_bound: int
    secret_type: str = "binary"
    noise_type: str = "uniform_small"
    embedding_config: dict[str, Any] = field(default_factory=dict)
    device: torch.device | str = "cpu"

    def sample_batch(self, batch_size: int) -> dict[str, Any]:
        A, b, s, e = generate_lwe_batch(
            batch_size=batch_size,
            n=self.n,
            m=self.m,
            q=self.q,
            h=self.h,
            noise_bound=self.noise_bound,
            device=self.device,
            secret_type=self.secret_type,
            noise_type=self.noise_type,
        )
        packet = build_crypto_image_packet(
            A=A,
            b=b,
            q=self.q,
            secret_type=self.secret_type,
            **self.embedding_config,
        )
        return {
            "A": A,
            "b": b,
            "s": s,
            "e": e,
            "packet": packet,
            "y_support": (s != 0).float(),
        }

