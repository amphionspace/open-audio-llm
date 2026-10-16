"""Dataset catalog loading, validation, and artifact resolution."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from .declarations import declaration_files, read_declarations
from .errors import ContractError, IntegrityError
from .roots import resolve_root_path
from .types import ArtifactRef, DatasetSpec


class DatasetCatalog:
    def __init__(self, specs: Iterable[DatasetSpec]):
        self._by_key: dict[str, DatasetSpec] = {}
        self._versions: dict[str, list[DatasetSpec]] = {}
        self._aliases: dict[str, str] = {}
        for spec in specs:
            if spec.key in self._by_key:
                raise ContractError(f"duplicate dataset key: {spec.key}")
            self._by_key[spec.key] = spec
            self._versions.setdefault(spec.dataset_id, []).append(spec)
            for alias in spec.aliases:
                existing = self._aliases.get(alias)
                if existing is not None and existing != spec.dataset_id:
                    raise ContractError(
                        f"alias {alias!r} maps to both {existing!r} and {spec.dataset_id!r}"
                    )
                self._aliases[alias] = spec.dataset_id
        for versions in self._versions.values():
            versions.sort(key=lambda item: item.version)

    def __len__(self) -> int:
        return len(self._by_key)

    def __iter__(self):
        return iter(self._by_key.values())

    def get(self, dataset_id: str, version: str | None = None) -> DatasetSpec:
        # A stable dataset ID takes precedence over an alias owned by another
        # versioned dataset. This permits migration entries such as the legacy
        # ``common_voice_en`` ID to coexist with a newer canonical dataset that
        # advertises that spelling as a compatibility alias.
        canonical = (
            dataset_id
            if dataset_id in self._versions
            else self._aliases.get(dataset_id, dataset_id)
        )
        versions = self._versions.get(canonical)
        if not versions:
            raise ContractError(f"unknown dataset or alias: {dataset_id!r}")
        if version is None:
            if len(versions) != 1:
                available = [item.version for item in versions]
                raise ContractError(
                    f"dataset {canonical!r} requires a version; available: {available}"
                )
            return versions[0]
        key = f"{canonical}@{version}"
        try:
            return self._by_key[key]
        except KeyError as exc:
            raise ContractError(f"unknown dataset version: {key}") from exc


def validate_catalog(specs: Iterable[DatasetSpec]) -> DatasetCatalog:
    return DatasetCatalog(specs)


def load_catalog(path: str | Path) -> DatasetCatalog:
    """Load YAML declarations from a file or directory, including legacy JSONL."""

    specs: list[DatasetSpec] = []
    for source in declaration_files(path):
        for line_number, row in read_declarations(source):
            try:
                specs.append(DatasetSpec.from_dict(row))
            except ContractError as exc:
                raise ContractError(f"{source}:{line_number}: {exc}") from exc
    return validate_catalog(specs)


def resolve_artifact(
    catalog: DatasetCatalog,
    dataset_id: str,
    version: str,
    artifact_name: str,
    roots: Mapping[str, str | Path],
) -> Path:
    artifact: ArtifactRef = catalog.get(dataset_id, version).artifact(artifact_name)
    return resolve_root_path(
        artifact.root_alias,
        artifact.relative_path,
        roots,
        f"{dataset_id}@{version}:{artifact_name}",
    )


def resolve_split(
    catalog: DatasetCatalog,
    dataset_id: str,
    version: str,
    split: str,
    role: str,
    roots: Mapping[str, str | Path],
) -> list[Path]:
    """Paths of one split role in declared order; [] when the split lacks the role."""
    spec = catalog.get(dataset_id, version)
    return [
        resolve_artifact(catalog, spec.dataset_id, spec.version, name, roots)
        for name in spec.split(split).artifacts.get(role, ())
    ]


def verify_artifact_file(
    artifact: ArtifactRef, path: str | Path
) -> dict[str, int | str]:
    """Verify a resolved file or directory against its published integrity facts."""

    selected = Path(path)
    if selected.is_dir():
        if artifact.expected_bytes is not None or artifact.sha256 is not None:
            raise IntegrityError(
                "directory artifact integrity facts must use "
                "metadata.expected_bytes and metadata.tree_sha256: "
                f"{artifact.name}"
            )
        return _verify_artifact_directory(artifact, selected)
    if not selected.is_file():
        raise IntegrityError(f"artifact is not a file or directory: {selected}")

    try:
        actual_bytes = selected.stat().st_size
    except OSError as exc:
        raise IntegrityError(f"artifact cannot be read: {selected}: {exc}") from exc
    if artifact.expected_bytes is not None and actual_bytes != artifact.expected_bytes:
        raise IntegrityError(
            f"artifact byte size mismatch for {selected}: "
            f"expected {artifact.expected_bytes}, got {actual_bytes}"
        )

    digest = hashlib.sha256()
    try:
        with selected.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise IntegrityError(f"artifact cannot be read: {selected}: {exc}") from exc
    actual_sha256 = digest.hexdigest()
    if artifact.sha256 is not None and actual_sha256 != artifact.sha256:
        raise IntegrityError(
            f"artifact sha256 mismatch for {selected}: "
            f"expected {artifact.sha256}, got {actual_sha256}"
        )

    result: dict[str, int | str] = {
        "bytes": actual_bytes,
        "sha256": actual_sha256,
    }
    expected_records = artifact.metadata.get("record_count")
    if expected_records is not None and (
        not isinstance(expected_records, int) or expected_records < 0
    ):
        raise IntegrityError(
            f"artifact metadata.record_count must be a non-negative integer: "
            f"{artifact.name}"
        )
    if expected_records is not None or selected.name.endswith(".jsonl.gz"):
        try:
            if selected.suffix == ".json":
                with selected.open(encoding="utf-8") as stream:
                    records = json.load(stream)
                if not isinstance(records, list):
                    raise IntegrityError(
                        f"artifact with record_count must contain a JSON array: {selected}"
                    )
                actual_records = len(records)
            else:
                opener = gzip.open if selected.suffix == ".gz" else Path.open
                with opener(selected, "rt", encoding="utf-8") as stream:
                    actual_records = sum(1 for line in stream if line.strip())
        except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
            raise IntegrityError(
                f"artifact cannot be read completely: {selected}: {exc}"
            ) from exc
        if expected_records is not None and actual_records != expected_records:
            raise IntegrityError(
                f"artifact record count mismatch for {selected}: "
                f"expected {expected_records}, got {actual_records}"
            )
        result["records"] = actual_records
    return result


def _verify_artifact_directory(
    artifact: ArtifactRef, selected: Path
) -> dict[str, int | str]:
    files = sorted(
        (path for path in selected.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(selected).as_posix(),
    )
    symlinks = [path for path in selected.rglob("*") if path.is_symlink()]
    if symlinks:
        raise IntegrityError(f"artifact directory contains symlinks: {symlinks[0]}")

    tree_digest = hashlib.sha256()
    actual_bytes = 0
    for file_path in files:
        relative_path = file_path.relative_to(selected).as_posix()
        size = file_path.stat().st_size
        digest = hashlib.sha256()
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        tree_digest.update(
            f"{relative_path}\0{size}\0{digest.hexdigest()}\n".encode()
        )
        actual_bytes += size

    expected_files = artifact.metadata.get("file_count")
    expected_bytes = artifact.metadata.get("expected_bytes")
    expected_tree_sha256 = artifact.metadata.get("tree_sha256")
    for name, value in (("file_count", expected_files), ("expected_bytes", expected_bytes)):
        if value is not None and (not isinstance(value, int) or value < 0):
            raise IntegrityError(
                f"artifact metadata.{name} must be a non-negative integer: "
                f"{artifact.name}"
            )
    if expected_files is not None and len(files) != expected_files:
        raise IntegrityError(
            f"artifact file count mismatch for {selected}: "
            f"expected {expected_files}, got {len(files)}"
        )
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise IntegrityError(
            f"artifact byte size mismatch for {selected}: "
            f"expected {expected_bytes}, got {actual_bytes}"
        )

    actual_tree_sha256 = tree_digest.hexdigest()
    if expected_tree_sha256 is not None:
        if (
            not isinstance(expected_tree_sha256, str)
            or len(expected_tree_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_tree_sha256.lower())
        ):
            raise IntegrityError(
                f"artifact metadata.tree_sha256 must be a 64-character hex digest: "
                f"{artifact.name}"
            )
        if actual_tree_sha256 != expected_tree_sha256.lower():
            raise IntegrityError(
                f"artifact tree sha256 mismatch for {selected}: "
                f"expected {expected_tree_sha256}, got {actual_tree_sha256}"
            )

    return {
        "bytes": actual_bytes,
        "files": len(files),
        "tree_sha256": actual_tree_sha256,
    }
