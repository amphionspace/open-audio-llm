"""Machine-local root alias configuration."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from .errors import ContractError, ResolutionError

ROOTS_ENV = "AUDIO_DATA_ROOTS_FILE"


def is_url(value: str | Path) -> bool:
    return "://" in str(value)


def load_roots(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    allow_urls: bool = False,
) -> dict[str, Path | str]:
    """Load root aliases; object-store URLs (s3://...) only where allow_urls is set."""
    environment = os.environ if environ is None else environ
    selected = path or environment.get(ROOTS_ENV)
    if not selected:
        raise ResolutionError(
            f"no roots file configured; pass one explicitly or set {ROOTS_ENV}"
        )
    roots_path = Path(selected)
    try:
        data = json.loads(roots_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResolutionError(f"failed to load roots file {roots_path}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ContractError("roots file must contain a non-empty JSON object")
    roots: dict[str, Path | str] = {}
    for alias, value in data.items():
        if not isinstance(alias, str) or not alias.strip():
            raise ContractError("root aliases must be non-empty strings")
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"root {alias!r} must be a non-empty path string")
        if allow_urls and is_url(value):
            roots[alias] = value.rstrip("/")
            continue
        root = Path(value).expanduser()
        if not root.is_absolute():
            raise ContractError(f"root {alias!r} must be an absolute path")
        roots[alias] = root.resolve(strict=False)
    return roots


def resolve_root_path(
    root_alias: str, relative_path: str, roots: Mapping[str, str | Path], where: str
) -> Path:
    if root_alias not in roots:
        raise ResolutionError(
            f"root alias {root_alias!r} is not configured for {where}"
        )
    root = Path(roots[root_alias]).expanduser().resolve(strict=False)
    resolved = (root / relative_path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ResolutionError(f"artifact escaped configured root: {resolved}") from exc
    return resolved


def portable_path(path: str | Path, roots: Mapping[str, str | Path]) -> dict[str, str]:
    """Express a local path under its most specific configured root alias."""
    path = Path(path).resolve(strict=False)
    matches = []
    for alias, value in roots.items():
        if is_url(value):
            continue
        root = Path(value).expanduser().resolve(strict=False)
        if path == root or root in path.parents:
            matches.append((len(root.parts), alias, root))
    if not matches:
        raise ResolutionError(f"path is outside every configured root: {path}")
    _, alias, root = max(matches)
    return {"root_alias": alias, "relative_path": path.relative_to(root).as_posix()}
