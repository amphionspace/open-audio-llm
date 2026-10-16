"""YAML config with attribute access and dotted CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """dict with attribute access; nested dicts become Config."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = _wrap(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _wrap(value)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, Config) else
                    [x.to_dict() if isinstance(x, Config) else x for x in v] if isinstance(v, list) else v)
                for k, v in self.items()}


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _deep_update(base: dict, other: dict) -> dict:
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    """Load YAML. ``_base_: other.yaml`` is merged first (relative to the file).

    ``overrides`` are ``a.b.c=value`` strings; values are parsed as YAML.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    base_ref = data.pop("_base_", None)
    if base_ref:
        base = load_config(path.parent / base_ref).to_dict()
        data = _deep_update(base, data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return Config(data)
