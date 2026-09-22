"""Typed access to ``params.yaml``.

Every stage (pipeline, tests, API) reads its settings through this module so a
parameter is declared exactly once and DVC can track it as a dependency.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_FILE = PROJECT_ROOT / "params.yaml"


class Config(dict):
    """A ``dict`` that also supports dotted lookups: ``cfg.get_path("data.raw_path")``."""

    def section(self, name: str) -> dict[str, Any]:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Missing '{name}' section in params.yaml") from exc

    def get_nested(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def get_path(self, dotted_key: str) -> Path:
        """Resolve a path parameter relative to the project root."""
        value = self.get_nested(dotted_key)
        if value is None:
            raise KeyError(f"'{dotted_key}' is not defined in params.yaml")
        return resolve_path(value)


def resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(params_file: str | os.PathLike[str] | None = None) -> Config:
    """Read ``params.yaml`` (or an override) into a :class:`Config`."""
    path = Path(params_file) if params_file else Path(os.getenv("PARAMS_FILE", DEFAULT_PARAMS_FILE))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Config(data)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached configuration for processes that read it repeatedly (e.g. the API)."""
    return load_config()
