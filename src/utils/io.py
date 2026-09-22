"""Small IO helpers that always create parent directories and use UTF-8."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    path = ensure_parent(path)
    frame.to_csv(path, index=False)
    return path


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
