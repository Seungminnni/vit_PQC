from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    step: int,
    metrics: dict[str, float] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": config,
        "step": step,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location=None) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    return payload

