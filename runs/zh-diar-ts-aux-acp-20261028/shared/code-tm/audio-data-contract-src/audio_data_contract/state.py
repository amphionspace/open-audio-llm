"""Mutable local download and preparation state."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .errors import ContractError, StateTransitionError

STATE_SCHEMA_VERSION = "dataset-state/1.0"
_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class DownloadState(str, Enum):
    PLANNED = "planned"
    WAITING_AUTH = "waiting_auth"
    DOWNLOADING = "downloading"
    PARTIAL = "partial"
    DOWNLOADED = "downloaded"
    VERIFIED = "verified"
    EXTRACTED = "extracted"
    PREPARED = "prepared"
    FAILED = "failed"


_TRANSITIONS: dict[DownloadState, set[DownloadState]] = {
    DownloadState.PLANNED: {
        DownloadState.WAITING_AUTH,
        DownloadState.DOWNLOADING,
        DownloadState.FAILED,
    },
    DownloadState.WAITING_AUTH: {
        DownloadState.PLANNED,
        DownloadState.DOWNLOADING,
        DownloadState.FAILED,
    },
    DownloadState.DOWNLOADING: {
        DownloadState.PARTIAL,
        DownloadState.DOWNLOADED,
        DownloadState.FAILED,
    },
    DownloadState.PARTIAL: {
        DownloadState.DOWNLOADING,
        DownloadState.DOWNLOADED,
        DownloadState.FAILED,
    },
    DownloadState.DOWNLOADED: {
        DownloadState.DOWNLOADING,
        DownloadState.VERIFIED,
        DownloadState.FAILED,
    },
    DownloadState.VERIFIED: {
        DownloadState.EXTRACTED,
        DownloadState.PREPARED,
        DownloadState.FAILED,
    },
    DownloadState.EXTRACTED: {DownloadState.PREPARED, DownloadState.FAILED},
    DownloadState.PREPARED: {DownloadState.FAILED},
    DownloadState.FAILED: {
        DownloadState.PLANNED,
        DownloadState.WAITING_AUTH,
        DownloadState.DOWNLOADING,
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _non_empty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{where} must be a non-empty string")
    return value


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    return dict(value)


def _updated_at(value: Any) -> str:
    if not isinstance(value, str) or _RFC3339.fullmatch(value) is None:
        raise ContractError("dataset state updated_at must be an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(
            "dataset state updated_at must be an RFC 3339 timestamp"
        ) from exc
    if parsed.utcoffset() is None:
        raise ContractError("dataset state updated_at must include a timezone")
    return value


@dataclass(frozen=True)
class DatasetState:
    dataset_id: str
    version: str
    state: DownloadState
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    updated_at: str = field(default_factory=_now)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATE_SCHEMA_VERSION:
            raise ContractError(f"unsupported state schema: {self.schema_version!r}")
        _non_empty_string(self.dataset_id, "dataset state dataset_id")
        _non_empty_string(self.version, "dataset state version")
        try:
            state = DownloadState(self.state)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid dataset state: {self.state!r}") from exc
        object.__setattr__(self, "state", state)
        artifacts = _object(self.artifacts, "dataset state artifacts")
        for name, artifact in artifacts.items():
            _non_empty_string(name, "dataset state artifact name")
            artifacts[name] = _object(
                artifact, f"dataset state artifact {name!r}"
            )
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "updated_at", _updated_at(self.updated_at))
        if self.error is not None and not isinstance(self.error, str):
            raise ContractError("dataset state error must be a string or null")
        object.__setattr__(
            self, "metadata", _object(self.metadata, "dataset state metadata")
        )

    def transition(
        self,
        target: DownloadState | str,
        *,
        error: str | None = None,
    ) -> DatasetState:
        try:
            target_state = DownloadState(target)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"invalid dataset state: {target!r}") from exc
        if target_state == self.state:
            return replace(self, updated_at=_now(), error=error)
        if target_state not in _TRANSITIONS[self.state]:
            raise StateTransitionError(
                f"illegal dataset state transition: {self.state.value} -> {target_state.value}"
            )
        return replace(self, state=target_state, updated_at=_now(), error=error)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "state": self.state.value,
            "artifacts": self.artifacts,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
        if self.error is not None:
            data["error"] = self.error
        return data

    @classmethod
    def from_dict(cls, value: Any) -> DatasetState:
        data = _object(value, "dataset state")
        required = {"schema_version", "dataset_id", "version", "state"}
        allowed = {
            "schema_version",
            "dataset_id",
            "version",
            "state",
            "artifacts",
            "updated_at",
            "error",
            "metadata",
        }
        missing = required - set(data)
        if missing:
            raise ContractError(
                f"dataset state missing required fields: {sorted(missing)}"
            )
        unknown = set(data) - allowed
        if unknown:
            raise ContractError(f"dataset state has unknown fields: {sorted(unknown)}")
        return cls(
            schema_version=data["schema_version"],
            dataset_id=data["dataset_id"],
            version=data["version"],
            state=data["state"],
            artifacts=data.get("artifacts", {}),
            updated_at=data.get("updated_at", _now()),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


def inspect_download(path: str | Path, expected_bytes: int | None = None) -> DownloadState:
    artifact = Path(path)
    control = Path(str(artifact) + ".aria2")
    if control.exists():
        return DownloadState.DOWNLOADING
    if not artifact.exists():
        return DownloadState.PLANNED
    if expected_bytes is not None and artifact.stat().st_size != expected_bytes:
        return DownloadState.PARTIAL
    return DownloadState.DOWNLOADED


def load_state(path: str | Path) -> DatasetState:
    return DatasetState.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_state_atomic(state: DatasetState, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(state.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
