from __future__ import annotations

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, name: str | None, steps: int):
    if not name or name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    raise ValueError(f"Unsupported scheduler: {name}")

