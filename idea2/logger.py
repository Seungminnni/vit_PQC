from __future__ import annotations

import csv
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


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

