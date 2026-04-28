from __future__ import annotations

import torch

from .gram import gram_interaction
from .phase import phase_image
from .raw import raw_image
from .rhie import rhie_binary, rhie_ternary


def build_crypto_image_packet(
    A: torch.Tensor,
    b: torch.Tensor,
    q: int,
    secret_type: str = "binary",
    freqs: tuple[int, ...] | list[int] = (1, 2, 4),
    use_raw: bool = True,
    use_phase: bool = True,
    use_rhie: bool = True,
    use_gram: bool = False,
    include_magnitude: bool = True,
) -> dict[str, torch.Tensor]:
    """Bundle enabled raw/phase/RHIE/Gram views for ablations."""
    packet: dict[str, torch.Tensor] = {}
    if use_raw:
        packet["raw"] = raw_image(A, b, q)
    if use_phase:
        packet["phase"] = phase_image(A, b, q, freqs=freqs)
    if use_rhie:
        if secret_type == "binary":
            packet["rhie"] = rhie_binary(A, b, q, freqs=freqs, include_magnitude=include_magnitude)
        elif secret_type == "ternary":
            packet["rhie"] = rhie_ternary(A, b, q, freqs=freqs, include_magnitude=include_magnitude)
        else:
            raise ValueError(f"Unsupported secret_type: {secret_type}")
    if use_gram:
        packet["gram"], packet["gram_vec"] = gram_interaction(A, b, q)
    return packet

