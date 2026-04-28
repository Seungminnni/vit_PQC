from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def make_run_dir(run_name: str, root: str = "results") -> Path:
    path = Path(root) / "csv" / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def save_checkpoint(path: str | Path, model, optimizer, config: dict[str, Any], step: int, metrics: dict[str, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "config": config,
            "step": step,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, model, map_location=None) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    return payload


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
