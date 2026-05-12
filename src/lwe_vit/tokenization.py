from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PatchTokens:
    tokens: torch.Tensor
    mask: torch.Tensor
    grid_size: tuple[int, int]


class RectangularPatchTokenizer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_rows: int = 4, patch_cols: int = 4) -> None:
        super().__init__()
        if in_channels <= 0 or embed_dim <= 0:
            raise ValueError("in_channels and embed_dim must be positive.")
        if patch_rows <= 0 or patch_cols <= 0:
            raise ValueError("patch_rows and patch_cols must be positive.")
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_rows, patch_cols),
            stride=(patch_rows, patch_cols),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None = None) -> PatchTokens:
        if image.dim() != 4:
            raise ValueError("image must have shape (B, C, H, W).")
        batch, _, height, width = image.shape
        if mask is None:
            mask = torch.ones((batch, height, width), dtype=torch.bool, device=image.device)
        if mask.shape != (batch, height, width):
            raise ValueError("mask must have shape (B, H, W).")

        image, mask = self._pad_if_needed(image, mask)
        patches = self.proj(image)
        grid_h, grid_w = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2).contiguous()
        patch_mask = F.max_pool2d(
            mask.to(image.dtype).unsqueeze(1),
            kernel_size=(self.patch_rows, self.patch_cols),
            stride=(self.patch_rows, self.patch_cols),
        ).squeeze(1)
        patch_mask = patch_mask.flatten(1).to(torch.bool)
        return PatchTokens(tokens=tokens, mask=patch_mask, grid_size=(grid_h, grid_w))

    def _pad_if_needed(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image.shape[-2:]
        pad_h = (self.patch_rows - height % self.patch_rows) % self.patch_rows
        pad_w = (self.patch_cols - width % self.patch_cols) % self.patch_cols
        if pad_h == 0 and pad_w == 0:
            return image, mask
        return (
            F.pad(image, (0, pad_w, 0, pad_h), value=0.0),
            F.pad(mask, (0, pad_w, 0, pad_h), value=False),
        )
