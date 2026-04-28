from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp) or {}
    if "defaults" in cfg:
        base_path = path.parent / cfg.pop("defaults")
        with base_path.open("r", encoding="utf-8") as fp:
            base = yaml.safe_load(fp) or {}
        cfg = deep_update(base, cfg)
    return cfg

