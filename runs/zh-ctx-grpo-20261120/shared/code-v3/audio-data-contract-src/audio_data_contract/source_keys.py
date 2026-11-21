"""Source keys: the original recordings a record was built from, and the ones evaluation holds out.

Training must not see recordings an evaluation set was cut from. Record IDs do
not say that: a derived dataset (windows, mixtures, re-simulations) gives its
records new IDs, and the original recording only survives in a field such as
``metadata.recording_id``. Two catalog declarations make it checkable:

``provenance.source_keys`` (dataset level; a split's own ``provenance.source_keys``
replaces it) says how to read the source keys of a record::

    source_keys:
    - namespace: emilia2          # the collection the keys name
      fields: [metadata.recording_id]
      pattern: '(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])'

Each field is a dotted record path (``id``, ``metadata.recording_id``,
``audio_slots``). Without ``pattern`` a string value (or each string of a list)
is the key; with it, every match in the field's text is a key, so one entry can
read the source of every slot of a mixture. Every record must yield at least one
key per entry: a declaration that matches nothing is an error, not a pass.

``provenance.holdout`` (dataset level) names the keys evaluation holds out::

    holdout:
    - namespace: emilia2
      artifact: emilia2-holdout-recordings   # kind source-keys: one ID per line
      pattern: '^[0-9a-f]{16}'               # optional, applied to each line

Keys are compared as ``<namespace>:<key>``. Both declarations live in
``provenance`` because older readers reject unknown top-level catalog fields
but load any provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ContractError
from .types import DatasetSpec

SOURCE_KEYS_KIND = "source-keys"


@dataclass(frozen=True)
class SourceKeySpec:
    namespace: str
    fields: tuple[str, ...]
    pattern: re.Pattern[str] | None = None

    @classmethod
    def from_dict(cls, value: Any, where: str) -> SourceKeySpec:
        data = _entry(value, where, {"namespace", "fields"}, {"pattern"})
        fields = data["fields"]
        if (
            not isinstance(fields, list)
            or not fields
            or not all(isinstance(item, str) and item.strip() for item in fields)
        ):
            raise ContractError(f"{where}.fields must be a non-empty list of field paths")
        return cls(
            namespace=_namespace(data["namespace"], where),
            fields=tuple(fields),
            pattern=_pattern(data.get("pattern"), where),
        )


@dataclass(frozen=True)
class HoldoutSpec:
    namespace: str
    artifact: str
    pattern: re.Pattern[str] | None = None

    @classmethod
    def from_dict(cls, value: Any, where: str) -> HoldoutSpec:
        data = _entry(value, where, {"namespace", "artifact"}, {"pattern"})
        artifact = data["artifact"]
        if not isinstance(artifact, str) or not artifact.strip():
            raise ContractError(f"{where}.artifact must name an artifact")
        return cls(
            namespace=_namespace(data["namespace"], where),
            artifact=artifact,
            pattern=_pattern(data.get("pattern"), where),
        )


@dataclass(frozen=True)
class Holdouts:
    """Held-out keys (``namespace:key``) and the declarations they came from."""

    keys: frozenset[str]
    sources: tuple[str, ...]

    @property
    def digest(self) -> str:
        """Content hash, for caches whose contents depend on the held-out keys."""
        return hashlib.sha256("\n".join(sorted(self.keys)).encode()).hexdigest()


def _entry(value: Any, where: str, required: set[str], optional: set[str]) -> Mapping:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    missing, unknown = required - set(value), set(value) - required - optional
    if missing or unknown:
        raise ContractError(
            f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}"
        )
    return value


def _namespace(value: Any, where: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", value):
        raise ContractError(f"{where}.namespace must be a lower-case identifier")
    return value


def _pattern(value: Any, where: str) -> re.Pattern[str] | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ContractError(f"{where}.pattern must be a non-empty regular expression")
    try:
        return re.compile(value)
    except re.error as exc:
        raise ContractError(f"{where}.pattern: {exc}") from None


def _list(value: Any, where: str) -> list:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{where} must be a non-empty list")
    return value


def source_key_specs(spec: DatasetSpec, split: str) -> tuple[SourceKeySpec, ...]:
    """The split's declaration, else the dataset's; () when neither declares one."""
    declared = spec.split(split).provenance.get("source_keys")
    where = f"{spec.key}:{split}.provenance.source_keys"
    if declared is None:
        declared = spec.provenance.get("source_keys")
        where = f"{spec.key}.provenance.source_keys"
    if declared is None:
        return ()
    return tuple(
        SourceKeySpec.from_dict(item, f"{where}[{i}]")
        for i, item in enumerate(_list(declared, where))
    )


def holdout_specs(spec: DatasetSpec) -> tuple[HoldoutSpec, ...]:
    declared = spec.provenance.get("holdout")
    if declared is None:
        return ()
    where = f"{spec.key}.provenance.holdout"
    specs = tuple(
        HoldoutSpec.from_dict(item, f"{where}[{i}]")
        for i, item in enumerate(_list(declared, where))
    )
    for item in specs:
        if spec.artifact(item.artifact).kind != SOURCE_KEYS_KIND:
            raise ContractError(
                f"{where}: artifact {item.artifact!r} must be of kind {SOURCE_KEYS_KIND}"
            )
    return specs


def validate_declarations(spec: DatasetSpec) -> None:
    """Fail on a malformed source-key or holdout declaration anywhere in ``spec``."""
    for split in spec.splits:
        source_key_specs(spec, split)
    holdout_specs(spec)


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _lookup(record: Any, path: str) -> Any:
    value = record
    for part in path.split("."):
        if isinstance(value, Mapping):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
        if value is None:
            return None
    return _plain(value)


def _texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return [json.dumps(value, ensure_ascii=False, sort_keys=True)]


def record_source_keys(record: Any, specs: Iterable[SourceKeySpec]) -> frozenset[str]:
    """``namespace:key`` source keys of an AudioRecord or record mapping."""
    keys: set[str] = set()
    for spec in specs:
        found: set[str] = set()
        for field in spec.fields:
            for text in _texts(_lookup(record, field)):
                if spec.pattern is None:
                    if text:
                        found.add(text)
                else:
                    found.update(m.group(0) for m in spec.pattern.finditer(text))
        if not found:
            record_id = _lookup(record, "id")
            raise ContractError(
                f"record {record_id!r} has no {spec.namespace} source key in {list(spec.fields)}"
            )
        keys.update(f"{spec.namespace}:{key}" for key in found)
    return frozenset(keys)


def _holdout_lines(path: Path, spec: HoldoutSpec, where: str) -> set[str]:
    keys = set()
    for line in path.read_text(encoding="utf-8").split():
        if spec.pattern is None:
            keys.add(line)
            continue
        match = spec.pattern.search(line)
        if match is None:
            raise ContractError(f"{where}: line {line!r} does not match the pattern")
        keys.add(match.group(0))
    if not keys:
        raise ContractError(f"{where}: holds no keys")
    return keys


def load_holdouts(catalog, roots: Mapping[str, str | Path]) -> Holdouts:
    """Every key the catalog's ``provenance.holdout`` declarations hold out.

    Each key file is verified against its registered size and sha256, so a
    replaced file cannot silently change what training excludes.
    """
    from .catalog import resolve_artifact, verify_artifact_file

    keys: set[str] = set()
    sources = []
    for spec in sorted(catalog, key=lambda item: item.key):
        for item in holdout_specs(spec):
            where = f"{spec.key}:{item.artifact}"
            artifact = spec.artifact(item.artifact)
            path = resolve_artifact(
                catalog, spec.dataset_id, spec.version, item.artifact, roots
            )
            verify_artifact_file(artifact, path)
            keys.update(
                f"{item.namespace}:{key}" for key in _holdout_lines(path, item, where)
            )
            sources.append(where)
    return Holdouts(keys=frozenset(keys), sources=tuple(sources))


def holdout_overlap(records: Iterable[Any], specs, holdouts: Holdouts) -> dict:
    """How many records draw on held-out keys; examples are record IDs."""
    total = overlapping = 0
    hit: set[str] = set()
    examples = []
    for record in records:
        total += 1
        common = record_source_keys(record, specs) & holdouts.keys
        if common:
            overlapping += 1
            hit.update(common)
            if len(examples) < 10:
                examples.append(_lookup(record, "id"))
    return {
        "records": total,
        "overlapping_records": overlapping,
        "held_out_keys_hit": len(hit),
        "examples": examples,
    }
