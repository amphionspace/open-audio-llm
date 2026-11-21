"""Durable cleaning patches, independently replayable on AudioRecord JSONL."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from itertools import islice
from pathlib import Path

from .errors import ContractError
from .types import AudioRecord, TransformStep

CHUNK_SIZE = 65536


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_sha256(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def chunks(items, size=None):
    size = size or CHUNK_SIZE
    items = iter(items)
    while chunk := list(islice(items, size)):
        yield chunk


class _ListHash:
    """Streaming equivalent of json_sha256(rows) for a JSON array of rows."""

    def __init__(self):
        self._digest = hashlib.sha256(b"[")
        self._first = True

    def update(self, line):
        self._digest.update((line if self._first else "," + line).encode())
        self._first = False

    def hexdigest(self):
        digest = self._digest.copy()
        digest.update(b"]")
        return digest.hexdigest()


def _overlaps(left, right):
    return left == right or left.startswith(right + ".") or right.startswith(left + ".")


def _covers(paths, field):
    return any(field == p or field.startswith(p + ".") for p in paths)


def validate_step(step, prior_writes=()):
    allowed = {"target", "language", "hotwords", "metadata", "labels", "$membership"}
    for field in step.writes:
        if field.split(".")[0] not in allowed or any(not p for p in field.split(".")):
            raise ContractError(f"unsupported Layer write field: {field}")
        if "." in field and field.split(".")[0] not in {"metadata", "labels"}:
            raise ContractError(f"unsupported Layer write field: {field}")
        if any(_overlaps(field, old) for old in prior_writes) and not _covers(
            step.overrides, field
        ):
            raise ContractError(f"field conflict for {field}; declare an override")


def apply_patch(record, patch, step):
    """Apply declared dotted field assignments; membership is separate from data."""
    if patch["id"] != record.id:
        raise ContractError("Layer patch ID does not match record")
    if patch.get("status") not in {"keep", "delete"}:
        raise ContractError("Layer status must be keep or delete")
    if patch["status"] == "delete" and "$membership" not in step.writes:
        raise ContractError("Layer deletion requires writes: $membership")
    changes = patch.get("changes")
    if not isinstance(changes, dict):
        raise ContractError("Layer changes must be an object")
    fields = list(changes)
    for i, field in enumerate(fields):
        if field == "$membership" or not _covers(step.writes, field):
            raise ContractError(f"undeclared Layer field: {field}")
        if any(_overlaps(field, other) for other in fields[i + 1 :]):
            raise ContractError(f"overlapping patch fields: {field}")
    data = json.loads(canonical_json(record.to_dict()))
    for field, value in changes.items():
        parts = field.split(".")
        if any(not part for part in parts):
            raise ContractError(f"invalid Layer field: {field}")
        node = data
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            if not isinstance(node[part], dict):
                raise ContractError(f"Layer field parent is not an object: {field}")
            node = node[part]
        leaf = parts[-1]
        if leaf in node and node[leaf] != value and not _covers(step.overrides, field):
            raise ContractError(f"existing field {field}; declare an override")
        node[leaf] = value
    return AudioRecord.from_dict(data), patch["status"] == "keep"


def write_layer(
    destination, patches, *, parent, step, tool, model=None, chunk_size=None
):
    """Persist a batch before materializing it; shared processing facts live in manifest.

    Patches are sorted and deduplicated on disk and validated against the parent in
    chunks, so memory does not grow with the batch size.
    """
    from .lance import RecordQuery, read_lance

    validate_step(step, parent.rebuild_metadata.get("writes", []))
    if not tool:
        raise ContractError("Layer requires a processing tool")
    destination = Path(destination).resolve()
    if destination.exists():
        raise ContractError(f"Layer destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".layer-", dir=destination.parent))
    try:
        with sqlite3.connect(work / "patches.sqlite") as db:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("CREATE TABLE patches (id TEXT PRIMARY KEY, row TEXT)")
            for patch in patches:
                if set(patch) != {"id", "changes", "status"}:
                    raise ContractError("patch requires exactly id, changes, status")
                if not isinstance(patch["id"], str) or not patch["id"]:
                    raise ContractError("invalid patch ID")
                row = {**patch, "parent_version": parent.snapshot_version}
                try:
                    db.execute(
                        "INSERT INTO patches VALUES (?, ?)",
                        (patch["id"], canonical_json({**row, "patch_hash": json_sha256(row)})),
                    )
                except sqlite3.IntegrityError:
                    raise ContractError(f"duplicate patch ID: {patch['id']}") from None
            ordered = (
                json.loads(row)
                for (row,) in db.execute("SELECT row FROM patches ORDER BY id")
            )
            digest, inputs, deletes = _ListHash(), 0, 0
            with (work / "patch.jsonl").open("w") as stream:
                for chunk in chunks(ordered, chunk_size):
                    ids = tuple(row["id"] for row in chunk)
                    found = {r.id: r for r in read_lance(parent, RecordQuery(ids=ids))}
                    missing = [i for i in ids if i not in found]
                    if missing:
                        raise ContractError(
                            f"unknown or deleted patch IDs: {missing[:10]}"
                        )
                    for row in chunk:
                        apply_patch(found[row["id"]], row, step)
                        line = canonical_json(row)
                        stream.write(line + "\n")
                        digest.update(line)
                        inputs += 1
                        deletes += row["status"] == "delete"
        (work / "patches.sqlite").unlink()
        if not inputs:
            raise ContractError("Layer patch batch may not be empty")
        manifest = {
            "schema_version": "audio-record-layer/1.0",
            "parent": parent.to_dict(),
            "transform": step.to_dict(),
            "tool": tool,
            "model": model,
            "parameters": step.parameters,
            "input_count": inputs,
            "output_count": inputs - deletes,
            "parent_record_count": parent.record_count,
            "result_record_count": parent.record_count - deletes,
            "patch_hash": digest.hexdigest(),
        }
        (work / "manifest.json").write_text(canonical_json(manifest) + "\n")
        work.rename(destination)
    finally:
        if work.exists():
            shutil.rmtree(work)
    return destination


def read_layer_manifest(path):
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    if manifest["schema_version"] != "audio-record-layer/1.0":
        raise ContractError("unsupported Layer schema")
    step = TransformStep.from_dict(manifest["transform"])
    validate_step(step, manifest["parent"]["rebuild_metadata"].get("writes", []))
    return manifest, step


def iter_layer_rows(path, manifest):
    """Stream patch rows in ID order; batch hash and counts are checked at the end,
    so callers must consume every row before trusting the batch."""
    digest, inputs, keeps, previous = _ListHash(), 0, 0, None
    with (Path(path) / "patch.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            digest.update(canonical_json(row))
            if previous is not None and row["id"] <= previous:
                raise ContractError(f"duplicate or unsorted patch ID: {row['id']}")
            previous = row["id"]
            payload = {key: value for key, value in row.items() if key != "patch_hash"}
            if json_sha256(payload) != row["patch_hash"]:
                raise ContractError("individual Layer patch hash mismatch")
            if row["parent_version"] != manifest["parent"]["snapshot_version"]:
                raise ContractError("Layer parent version mismatch")
            inputs += 1
            keeps += row["status"] == "keep"
            yield row
    if digest.hexdigest() != manifest["patch_hash"]:
        raise ContractError("Layer patch hash mismatch")
    if (
        inputs != manifest["input_count"]
        or keeps != manifest["output_count"]
        or manifest["result_record_count"]
        != manifest["parent_record_count"] - (inputs - keeps)
    ):
        raise ContractError("Layer count mismatch")


def read_layer(path):
    manifest, step = read_layer_manifest(path)
    return manifest, list(iter_layer_rows(path, manifest)), step


def replay_layer(records, layer):
    """Replay onto the declared parent records without importing Lance/PyArrow."""
    manifest, rows, step = read_layer(layer)
    patches = {r["id"]: r for r in rows}
    count = 0
    for record in records:
        count += 1
        patch = patches.pop(record.id, None)
        if patch is None:
            yield record
        else:
            updated, retained = apply_patch(record, patch, step)
            if retained:
                yield updated
    if patches or count != manifest["parent_record_count"]:
        raise ContractError("Layer replay parent IDs/count mismatch")
