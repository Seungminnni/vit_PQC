from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .lwe import centered_mod


GRID_NAMES = {"relation_grid", "phase_grid"}
REPRESENTATION_NAMES = GRID_NAMES | {"row_equation_tokens"}


@dataclass(frozen=True)
class RepresentationConfig:
    name: str = "relation_grid"
    patch_rows: int = 4
    patch_cols: int = 4
    use_phase: bool = True
    broadcast_b: bool = True
    add_rhs_column: bool = True

    def __post_init__(self) -> None:
        if self.name not in REPRESENTATION_NAMES:
            raise ValueError(f"name must be one of: {sorted(REPRESENTATION_NAMES)}")
        if self.patch_rows <= 0 or self.patch_cols <= 0:
            raise ValueError("patch_rows and patch_cols must be positive.")
        if self.name == "phase_grid" and not self.use_phase:
            raise ValueError("phase_grid requires use_phase=True.")


class LWEImageEncoder:
    """Encode LWE samples as relation-preserving image-like tensors.

    The public encode method always returns batched tensors:
    image: (B, C, H_padded, W_padded)
    mask:  (B, H_padded, W_padded), true on original non-padding cells.
    """

    def __init__(self, config: RepresentationConfig, q: int) -> None:
        if q <= 2:
            raise ValueError("q must be greater than 2.")
        self.config = config
        self.q = q

    def channel_names(self, n: int | None = None) -> list[str]:
        if self.config.name == "row_equation_tokens":
            if n is None:
                raise ValueError("n is required for row_equation_tokens channel names.")
            base = [f"row_x{idx}_centered" for idx in range(n + 1)]
            if not self.config.use_phase:
                return base
            return (
                base
                + [f"row_x{idx}_sin" for idx in range(n + 1)]
                + [f"row_x{idx}_cos" for idx in range(n + 1)]
            )

        if not self.config.use_phase:
            return ["A_centered", "b_centered_broadcast", "is_rhs_column"]
        return [
            "A_centered",
            "sin(A)",
            "cos(A)",
            "b_centered_broadcast",
            "sin(b)",
            "cos(b)",
            "is_rhs_column",
        ]

    def num_channels(self, n: int | None = None) -> int:
        return len(self.channel_names(n=n))

    def encode(self, A: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        A, b = self._as_batch(A, b)
        if self.config.name == "row_equation_tokens":
            return self._encode_row_equation_tokens(A, b)
        return self._encode_grid(A, b)

    def _as_batch(self, A: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if A.dim() == 2:
            A = A.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        if A.dim() != 3 or b.dim() != 2:
            raise ValueError("A must have shape (m, n) or (B, m, n); b must have shape (m) or (B, m).")
        if A.shape[0] != b.shape[0] or A.shape[1] != b.shape[1]:
            raise ValueError("A and b batch/equation dimensions do not match.")
        return torch.remainder(A.to(torch.long), self.q), torch.remainder(b.to(torch.long), self.q)

    def _centered_unit(self, x: torch.Tensor) -> torch.Tensor:
        return centered_mod(x, self.q).to(torch.float32) / (float(self.q) / 2.0)

    def _phase(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angle = x.to(torch.float32) * (2.0 * math.pi / float(self.q))
        return torch.sin(angle), torch.cos(angle)

    def _encode_grid(self, A: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, m, n = A.shape
        width = n + 1 if self.config.add_rhs_column else n
        values = torch.zeros((batch, m, width), dtype=torch.long, device=A.device)
        values[:, :, :n] = A
        if self.config.add_rhs_column:
            values[:, :, n] = b

        b_grid = b.unsqueeze(-1).expand(batch, m, width)
        if not self.config.broadcast_b:
            b_grid = torch.zeros_like(b_grid)

        channels = [self._centered_unit(values)]
        if self.config.use_phase:
            sin_a, cos_a = self._phase(values)
            channels.extend([sin_a, cos_a])

        channels.append(self._centered_unit(b_grid))
        if self.config.use_phase:
            sin_b, cos_b = self._phase(b_grid)
            channels.extend([sin_b, cos_b])

        rhs = torch.zeros((batch, m, width), dtype=torch.float32, device=A.device)
        if self.config.add_rhs_column:
            rhs[:, :, n] = 1.0
        channels.append(rhs)

        image = torch.stack(channels, dim=1)
        mask = torch.ones((batch, m, width), dtype=torch.bool, device=A.device)
        return self._pad(image, mask)

    def _encode_row_equation_tokens(self, A: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.cat([A, b.unsqueeze(-1)], dim=-1)
        features = [self._centered_unit(values)]
        if self.config.use_phase:
            sin_x, cos_x = self._phase(values)
            features.extend([sin_x, cos_x])
        row_features = torch.cat(features, dim=-1)
        image = row_features.transpose(1, 2).unsqueeze(-1).contiguous()
        mask = torch.ones((A.shape[0], A.shape[1], 1), dtype=torch.bool, device=A.device)
        return self._pad(image, mask)

    def _pad(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = image.shape
        pad_h = (self.config.patch_rows - height % self.config.patch_rows) % self.config.patch_rows
        pad_w = (self.config.patch_cols - width % self.config.patch_cols) % self.config.patch_cols
        if pad_h == 0 and pad_w == 0:
            return image, mask
        image = F.pad(image, (0, pad_w, 0, pad_h), value=0.0)
        mask = F.pad(mask, (0, pad_w, 0, pad_h), value=False)
        return image, mask
