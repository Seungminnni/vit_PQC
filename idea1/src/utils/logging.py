from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JSONLLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = self.path.open("a", encoding="utf-8")

    def log(self, row: dict[str, Any]) -> None:
        self.fp.write(json.dumps(row, sort_keys=True) + "\n")
        self.fp.flush()

    def close(self) -> None:
        self.fp.close()

