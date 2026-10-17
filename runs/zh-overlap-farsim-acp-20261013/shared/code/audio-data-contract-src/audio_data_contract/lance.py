"""Optional, local Lance materializations; JSONL and Layers remain authoritative."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import cache
from importlib.resources import files
from pathlib import Path

from .errors import ContractError
from .layers import (
    apply_patch,
    canonical_json,
    chunks,
    iter_layer_rows,
    json_sha256,
    read_layer,
    read_layer_manifest,
    replay_layer,
)
from .records import load_records, open_text, write_records
from .roots import is_url, load_roots, portable_path, resolve_root_path
from .types import RECORD_SCHEMA_VERSION, AudioRecord, relative_posix_path

STORAGE_VERSION = "2.0"
# Fragments are the training shard unit: ~350 at 92M rows keeps 64 shards balanced.
FRAGMENT_ROWS = 2**18
MAPPING_VERSION = "audio-record-lance/1.0"


def _dependencies():
    try:
        import lance
        import pyarrow as pa
    except ImportError as exc:
        raise ContractError(
            "Lance requires: pip install 'audio-data-contract[lance]'"
        ) from exc
    return lance, pa


def _schema():
    _, pa = _dependencies()
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("language", pa.string(), nullable=False),
            pa.field("splits", pa.list_(pa.string()), nullable=False),
            pa.field("clean_pass", pa.bool_()),
            pa.field("retained", pa.bool_(), nullable=False),
            pa.field("record_json", pa.large_string(), nullable=False),
        ],
        metadata={b"mapping": MAPPING_VERSION.encode()},
    )


@cache
def schema_hash():
    contract = json.loads(
        files("audio_data_contract")
        .joinpath("schemas/audio-record-1.0.json")
        .read_text()
    )
    return json_sha256(
        {"contract": contract, "mapping": MAPPING_VERSION, "arrow": str(_schema())}
    )


def _is_ref(value):
    return isinstance(value, dict)


def _check_location(value, where):
    """A location is a legacy absolute path or a {root_alias, relative_path} ref."""
    if isinstance(value, str) and value:
        return
    if (
        _is_ref(value)
        and set(value) == {"root_alias", "relative_path"}
        and isinstance(value["root_alias"], str)
        and value["root_alias"]
    ):
        relative_posix_path(value["relative_path"], where)
        return
    raise ContractError(f"{where} must be a path or a root_alias reference")


def _location(path, roots):
    return str(Path(path).resolve()) if roots is None else portable_path(path, roots)


def _local(value, roots):
    """Resolve a stored location to a local Path, or a URI on an object-store root."""
    if not _is_ref(value):
        return Path(value)
    roots = load_roots(allow_urls=True) if roots is None else roots
    root = roots.get(value["root_alias"])
    if root is not None and is_url(root):
        return f"{str(root).rstrip('/')}/{value['relative_path']}"
    return resolve_root_path(
        value["root_alias"], value["relative_path"], roots, "Lance artifact"
    )


@dataclass(frozen=True)
class LanceArtifact:
    table_path: str | dict
    snapshot_version: int
    storage_version: str
    audio_record_schema: str
    record_count: int
    schema_hash: str
    source_view: str
    rebuild_metadata: dict
    kind: str = "lance"
    # Machine-local root aliases; never serialized. None means AUDIO_DATA_ROOTS_FILE.
    roots: dict | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        _check_location(self.table_path, "table_path")
        if self.kind != "lance" or self.audio_record_schema != RECORD_SCHEMA_VERSION:
            raise ContractError("unsupported Lance artifact or AudioRecord schema")
        if self.storage_version != STORAGE_VERSION:
            raise ContractError(
                f"unsupported Lance storage version: {self.storage_version}"
            )
        if (
            type(self.snapshot_version) is not int
            or self.snapshot_version < 1
            or type(self.record_count) is not int
            or self.record_count < 0
        ):
            raise ContractError("invalid Lance snapshot version or record count")
        if not self.source_view or "@" not in self.source_view:
            raise ContractError(
                "source_view must include a fixed version: view@version"
            )

    @property
    def portable(self):
        return _is_ref(self.table_path)

    @property
    def table_uri(self):
        return str(_local(self.table_path, self.roots))

    def local(self, value):
        """Resolve a stored location (table, source JSONL or Layer) on this machine."""
        return _local(value, self.roots)

    def location_roots(self):
        """Roots used to store new locations: None keeps legacy absolute paths."""
        if not self.portable:
            return None
        return load_roots(allow_urls=True) if self.roots is None else self.roots

    def replace(self, **changes):
        return type(self)(**{**self.to_dict(), "roots": self.roots, **changes})

    def to_dict(self):
        data = asdict(self)
        del data["roots"]
        return data

    @classmethod
    def read(cls, path, roots=None):
        try:
            return cls(**json.loads(Path(path).read_text()), roots=roots)
        except (TypeError, ValueError, OSError) as exc:
            raise ContractError(
                f"invalid Lance artifact manifest {path}: {exc}"
            ) from exc


def catalog_ref(artifact, sidecar, *, name, roots):
    """Catalog ArtifactRef that pins a published sidecar and its snapshot."""
    from .types import ArtifactRef

    if not artifact.portable:
        raise ContractError("catalog registration requires a portable Lance artifact")
    return ArtifactRef(
        name=name,
        kind="lance-table",
        **portable_path(sidecar, roots),
        metadata={
            "snapshot_version": artifact.snapshot_version,
            "record_count": artifact.record_count,
            "schema_hash": artifact.schema_hash,
            "source_view": artifact.source_view,
        },
    )


def open_catalog_artifact(catalog, dataset_id, version, artifact_name, roots):
    """Open a registered lance-table at exactly the snapshot the catalog pins."""
    from .catalog import resolve_artifact

    ref = catalog.get(dataset_id, version).artifact(artifact_name)
    if ref.kind != "lance-table":
        raise ContractError(f"artifact {artifact_name!r} is not a lance-table")
    sidecar = resolve_artifact(catalog, dataset_id, version, artifact_name, roots)
    artifact = LanceArtifact.read(sidecar, roots)
    for key in ("snapshot_version", "record_count", "schema_hash"):
        if getattr(artifact, key) != ref.metadata[key]:
            raise ContractError(f"Lance sidecar {key} differs from catalog: {sidecar}")
    return artifact


@dataclass(frozen=True)
class RecordQuery:
    ids: tuple[str, ...] | None = None
    task: str | None = None
    language: str | None = None
    split: str | None = None
    clean_pass: bool | None = None

    def __post_init__(self):
        if self.clean_pass is not None and type(self.clean_pass) is not bool:
            raise ContractError("clean_pass must be boolean")

    def matches(self, record):
        clean = record.metadata.get("clean")
        passed = clean.get("pass") if isinstance(clean, dict) else None
        return (
            (self.ids is None or record.id in self.ids)
            and (self.task is None or record.task == self.task)
            and (self.language is None or record.language == self.language)
            and (
                self.split is None
                or any(s.ref.split == self.split for s in record.audio_slots)
            )
            and (self.clean_pass is None or passed is self.clean_pass)
        )

    def expression(self):
        def quote(value):
            return "'" + value.replace("'", "''") + "'"

        parts = ["retained = true"]
        if self.ids is not None:
            parts.append(
                "id IN (" + ",".join(map(quote, self.ids)) + ")"
                if self.ids
                else "false"
            )
        for name in ("task", "language"):
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name} = {quote(value)}")
        if self.split is not None:
            parts.append(f"array_has(splits, {quote(self.split)})")
        if self.clean_pass is not None:
            parts.append("clean_pass = " + str(self.clean_pass).lower())
        return " AND ".join(parts)


def _row(record, retained=True, *, validate=True):
    if validate:
        # Revalidate objects constructed directly by callers.
        record = AudioRecord.from_dict(record.to_dict())
    clean = record.metadata.get("clean")
    passed = clean.get("pass") if isinstance(clean, dict) else None
    return {
        "id": record.id,
        "task": record.task,
        "language": record.language,
        "splits": sorted({s.ref.split for s in record.audio_slots}),
        "clean_pass": passed if type(passed) is bool else None,
        "retained": retained,
        "record_json": canonical_json(record.to_dict()),
    }


# Scalar indexes on every query projection; each build commits a table version.
INDICES = (
    ("id", "BTREE"),
    ("task", "BITMAP"),
    ("language", "BITMAP"),
    ("splits", "LABEL_LIST"),
    ("clean_pass", "BITMAP"),
    ("retained", "BITMAP"),
)


def _create_indices(dataset):
    for column, kind in INDICES:
        dataset.create_scalar_index(column, kind)
    return dataset


def _encode_lines(chunk):
    """Validate one chunk of raw JSONL lines into an Arrow batch, in source order.

    Runs in import worker processes; returns the batch, its IDs and the canonical
    record stream bytes that feed the source digest.
    """
    _, pa = _dependencies()
    source, first_line, lines = chunk
    rows = []
    for number, line in enumerate(lines, first_line):
        line = line.strip()
        if not line:
            continue
        try:
            record = AudioRecord.from_dict(json.loads(line))
        except ValueError as exc:
            raise ContractError(f"{source}:{number}: {exc}") from exc
        rows.append(_row(record, validate=False))
    stream = "".join(row["record_json"] + "\n" for row in rows).encode()
    batch = pa.RecordBatch.from_pylist(rows, schema=_schema())
    return batch, [row["id"] for row in rows], stream


def _line_chunks(source, size):

    with open_text(source, "r") as stream:
        lines, first = [], 1
        for number, line in enumerate(stream, 1):
            lines.append(line)
            if len(lines) == size:
                yield str(source), first, lines
                lines, first = [], number + 1
        if lines:
            yield str(source), first, lines


def _ordered(executor, function, items, window):
    """Lazy, order-preserving map with at most `window` chunks in flight."""
    from collections import deque

    pending = deque()
    for item in items:
        pending.append(executor.submit(function, item))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def _claim_ids(db, ids):
    """Reserve a batch of IDs in the disk-backed index; reject any duplicate."""
    try:
        db.executemany("INSERT INTO ids VALUES (?)", ((i,) for i in ids))
        db.commit()
    except sqlite3.IntegrityError:
        db.rollback()
        seen = set()
        for i in ids:
            if i in seen or db.execute("SELECT 1 FROM ids WHERE id=?", (i,)).fetchone():
                raise ContractError(f"duplicate record ID: {i}") from None
            seen.add(i)
        raise


def open_artifact(artifact, *, verify_count=False):
    lance, _ = _dependencies()
    if artifact.schema_hash != schema_hash():
        raise ContractError(
            "Lance schema hash mismatch; rebuild with this mapping version"
        )
    try:
        dataset = lance.dataset(artifact.table_uri, version=artifact.snapshot_version)
        if not dataset.schema.equals(_schema(), check_metadata=True):
            raise ContractError("Lance physical schema mismatch")
        if dataset.data_storage_version != artifact.storage_version:
            raise ContractError("Lance storage version mismatch")
        if (
            verify_count
            and dataset.count_rows(filter="retained = true") != artifact.record_count
        ):
            raise ContractError("Lance snapshot record count mismatch")
        return dataset
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(
            f"cannot read Lance snapshot {artifact.snapshot_version} at "
            f"{artifact.table_path}; check version/schema compatibility: {exc}"
        ) from exc


def read_lance(artifact, query=None, *, batch_size=4096):
    dataset = open_artifact(artifact)
    try:
        for batch in dataset.to_batches(
            columns=["record_json"],
            filter=(query or RecordQuery()).expression(),
            batch_size=batch_size,
        ):
            for value in batch.column(0).to_pylist():
                yield AudioRecord.from_dict(json.loads(value))
    except Exception as exc:
        raise ContractError(f"Lance snapshot read/schema error: {exc}") from exc


def import_jsonl(
    source,
    destination,
    *,
    source_view,
    batch_size=4096,
    rebuild_metadata=None,
    roots=None,
    workers=1,
):
    """Publish a new local directory only after all batches validate successfully.

    With roots, stored locations become portable root_alias references. Workers > 1
    validate/encode batches in spawned processes; output order stays source order.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    lance, _ = _dependencies()
    if batch_size < 1 or workers < 1:
        raise ContractError("batch_size and workers must be positive")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ContractError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".lance-import-", dir=destination.parent))
    count = 0
    digest = hashlib.sha256()
    pool = None
    try:
        chunks = _line_chunks(source, batch_size)
        if workers == 1:
            encoded = map(_encode_lines, chunks)
        else:
            # A dead worker raises BrokenProcessPool instead of hanging the import.
            pool = ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")
            )
            encoded = _ordered(pool, _encode_lines, chunks, 2 * workers)
        with sqlite3.connect(work / "ids.sqlite", check_same_thread=False) as ids:
            ids.execute("PRAGMA journal_mode=MEMORY")
            ids.execute("PRAGMA synchronous=OFF")
            ids.execute("CREATE TABLE ids (id TEXT PRIMARY KEY)")

            def batches():
                nonlocal count
                for batch, batch_ids, stream in encoded:
                    if not batch_ids:
                        continue
                    _claim_ids(ids, batch_ids)
                    digest.update(stream)
                    count += len(batch_ids)
                    yield batch

            dataset = lance.write_dataset(
                batches(),
                work / "table.lance",
                schema=_schema(),
                data_storage_version=STORAGE_VERSION,
                max_rows_per_file=FRAGMENT_ROWS,
            )
        (work / "ids.sqlite").unlink()
        if count:
            dataset = _create_indices(dataset)
        artifact = LanceArtifact(
            _location(destination / "table.lance", roots),
            dataset.version,
            STORAGE_VERSION,
            RECORD_SCHEMA_VERSION,
            count,
            schema_hash(),
            source_view,
            rebuild_metadata
            or {
                "source_jsonl": _location(source, roots),
                "source_records_sha256": digest.hexdigest(),
                "mapping": MAPPING_VERSION,
                "layers": [],
                "writes": [],
                "pylance": lance.__version__,
            },
            roots=roots,
        )
        if dataset.count_rows() != count:
            raise ContractError("Lance import record count mismatch")
        dataset.tags.create("import", dataset.version)
        (work / "artifact.json").write_text(canonical_json(artifact.to_dict()) + "\n")
        work.rename(destination)
        return artifact
    except Exception as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"Lance import failed; nothing published: {exc}") from exc
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
        if work.exists():
            shutil.rmtree(work)


MEMORY_POOL_ENV = "LANCE_MEM_POOL_SIZE"


def _merge_memory_pool(staged_bytes):
    """merge_insert's hash join cannot spill, and Lance's default 150 MiB pool fails
    near 200k patches; size it to the staged Layer unless the caller set it."""
    if MEMORY_POOL_ENV in os.environ:
        return None
    os.environ[MEMORY_POOL_ENV] = str(max(150 * 2**20, 4 * staged_bytes))
    return MEMORY_POOL_ENV


@contextmanager
def _writer_lock(artifact):
    """Serialize every writer of one table (local POSIX lock beside the table)."""
    import fcntl

    if is_url(artifact.table_uri):
        raise ContractError(
            "object-store Lance tables are read-only mirrors; write the local table"
        )
    with (Path(artifact.table_uri).parent / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _published(dataset):
    """Tagged versions are the published snapshots; cleanup keeps every one."""
    return {name: tag["version"] for name, tag in dataset.tags.list().items()}


def _publish(dataset, artifact, tag, output):
    """Tag the committed version first, then publish its sidecar atomically."""
    if tag not in _published(dataset):
        dataset.tags.create(tag, artifact.snapshot_version)
    open_artifact(artifact, verify_count=True)
    temporary = output.with_name("." + output.name + ".tmp")
    temporary.write_text(canonical_json(artifact.to_dict()) + "\n")
    temporary.rename(output)
    return artifact


def materialize_layer(layer, *, source_view, roots=None):
    """One local writer, one merge commit, then publish an immutable artifact manifest.

    If this fails, patch.jsonl and manifest.json remain available for JSONL replay.
    Rerunning it resumes: a tagged commit only needs its sidecar, and untagged
    commits left after a tagged parent are rolled back to the parent content.
    All writers to a managed table must use this adapter (local POSIX lock).
    """

    lance, pa = _dependencies()
    layer = Path(layer).resolve()
    manifest, step = read_layer_manifest(layer)
    # Verify the whole canonical batch (hashes, order, counts) before any write.
    for _ in iter_layer_rows(layer, manifest):
        pass
    parent = LanceArtifact(**manifest["parent"], roots=roots)
    output = layer / "artifact.json"
    if output.exists():
        raise ContractError(f"Layer already published: {output}")
    # Identity of the whole Layer (parent, transform, tool, model and patches).
    tag = "layer-" + json_sha256(manifest)[:32]
    metadata = {
        **parent.rebuild_metadata,
        "layers": [
            *parent.rebuild_metadata["layers"],
            _location(layer, parent.location_roots()),
        ],
        "writes": sorted(set(parent.rebuild_metadata["writes"]) | set(step.writes)),
    }
    artifact = parent.replace(
        snapshot_version=parent.snapshot_version + 1,
        record_count=manifest["result_record_count"],
        source_view=source_view,
        rebuild_metadata=metadata,
    )
    with _writer_lock(parent):
        dataset = open_artifact(parent)
        latest = lance.dataset(parent.table_uri)
        published = _published(latest)
        if tag in published:
            # Committed and tagged before an interrupted sidecar publication.
            artifact = artifact.replace(snapshot_version=published[tag])
            return _publish(latest, artifact, tag, output)
        restore = latest.version != parent.snapshot_version
        if restore and (
            parent.snapshot_version not in published.values()
            or max(published.values()) > parent.snapshot_version
        ):
            raise ContractError(
                "stale Layer parent snapshot; replay or rebuild required"
            )
        # Stage patched rows chunk by chunk; memory stays bounded by the chunk size.
        with tempfile.NamedTemporaryFile(
            prefix=".layer-rows-", suffix=".arrows", dir=layer.parent
        ) as staged:
            with pa.ipc.new_stream(staged, _schema()) as writer:
                for chunk in chunks(iter_layer_rows(layer, manifest)):
                    ids = tuple(patch["id"] for patch in chunk)
                    found = {r.id: r for r in read_lance(parent, RecordQuery(ids=ids))}
                    rows = []
                    for patch in chunk:
                        if patch["id"] not in found:
                            raise ContractError(
                                f"unknown or deleted patch ID: {patch['id']}"
                            )
                        record, retained = apply_patch(found[patch["id"]], patch, step)
                        rows.append(_row(record, retained))
                    writer.write_batch(
                        pa.RecordBatch.from_pylist(rows, schema=_schema())
                    )
            staged.flush()
            pool = _merge_memory_pool(os.path.getsize(staged.name))
            try:
                if restore:
                    # Only unpublished commits follow the parent: roll them back.
                    dataset.restore()
                    dataset = lance.dataset(parent.table_uri)
                with pa.memory_map(staged.name) as source:
                    dataset.merge_insert("id").when_matched_update_all().execute(
                        pa.ipc.open_stream(source)
                    )
                if pool:
                    del os.environ[pool]
                # Local lock guarantees this adapter is the only writer to this table.
                committed = lance.dataset(parent.table_uri)
                if committed.describe_indices():
                    # Index the rewritten rows; this commits one more table version.
                    committed.optimize.optimize_indices()
                    committed = lance.dataset(parent.table_uri)
                artifact = artifact.replace(snapshot_version=committed.version)
                return _publish(committed, artifact, tag, output)
            except Exception as exc:
                raise ContractError(
                    f"Lance Layer materialization failed; canonical patch retained at {layer}. "
                    f"Rerun to resume, or replay/rebuild from its parent: {exc}"
                ) from exc
            finally:
                if pool:
                    os.environ.pop(pool, None)


def _content_digest(dataset):
    """Order-insensitive digest of every visible record."""
    total = 0
    for batch in dataset.to_batches(
        columns=["record_json"], filter="retained = true", use_scalar_index=False
    ):
        for value in batch.column(0).to_pylist():
            total += int.from_bytes(hashlib.sha256(value.encode()).digest(), "big")
    return total % 2**256


def compact_artifact(artifact, destination):
    """Compact the newest published snapshot into an equivalent, tagged sidecar."""
    lance, _ = _dependencies()
    destination = Path(destination).resolve()
    if destination.exists():
        raise ContractError(f"destination already exists: {destination}")
    with _writer_lock(artifact):
        dataset = open_artifact(artifact, verify_count=True)
        latest = lance.dataset(artifact.table_uri)
        if latest.version != artifact.snapshot_version:
            raise ContractError("compaction requires the newest published snapshot")
        before = _content_digest(dataset)
        latest.optimize.compact_files(
            target_rows_per_fragment=FRAGMENT_ROWS, materialize_deletions=True
        )
        compacted = lance.dataset(artifact.table_uri)
        if compacted.describe_indices():
            compacted.optimize.optimize_indices()
            compacted = lance.dataset(artifact.table_uri)
        if _content_digest(compacted) != before:
            raise ContractError("compaction changed visible records; not published")
        result = artifact.replace(snapshot_version=compacted.version)
        destination.parent.mkdir(parents=True, exist_ok=True)
        return _publish(compacted, result, f"compact-{compacted.version}", destination)


def protect_artifact(artifact):
    """Tag a sidecar's snapshot (e.g. one published before tags) against cleanup."""
    with _writer_lock(artifact):
        dataset = open_artifact(artifact)
        if artifact.snapshot_version not in _published(dataset).values():
            dataset.tags.create(
                f"protect-{artifact.snapshot_version}", artifact.snapshot_version
            )
    return sorted(set(_published(dataset).values()))


def cleanup_artifact(artifact, *, older_than_days):
    """Delete untagged versions older than the cutoff; tagged ones are kept."""
    from datetime import timedelta

    lance, _ = _dependencies()
    if older_than_days < 0:
        raise ContractError("older_than_days must be non-negative")
    with _writer_lock(artifact):
        dataset = lance.dataset(artifact.table_uri)
        if artifact.snapshot_version not in _published(dataset).values():
            raise ContractError(
                "cleanup needs a tagged snapshot; protect every published sidecar first"
            )
        stats = dataset.cleanup_old_versions(
            older_than=timedelta(days=older_than_days),
            error_if_tagged_old_versions=False,
        )
    return {"old_versions": stats.old_versions, "bytes_removed": stats.bytes_removed}


def mirror_artifact(artifact, target_root):
    """Copy a published local table to another root, e.g. COS/OBS as s3://bucket/x.

    Files keep the sidecar's relative path, so the same sidecar reads the mirror
    once a roots file maps its alias to target_root. Immutable files go first,
    version manifests next and tags last; existing immutable files are skipped.
    Object-store credentials/endpoints come from the standard AWS_* environment.
    """
    from pyarrow import fs

    if not artifact.portable:
        raise ContractError("mirroring requires a portable Lance artifact")
    alias = artifact.table_path["root_alias"]
    target = f"{str(target_root).rstrip('/')}/{artifact.table_path['relative_path']}"
    if is_url(target):
        remote, base = fs.FileSystem.from_uri(target)
    else:
        remote, base = fs.LocalFileSystem(), str(Path(target).resolve())
    local = fs.LocalFileSystem()

    def order(path):
        top = path.parts[0]
        return (top == "_versions") + 2 * (top == "_refs"), str(path)

    copied = skipped = size = 0
    with _writer_lock(artifact):
        source = Path(artifact.table_uri)
        files = sorted(
            (p.relative_to(source) for p in source.rglob("*") if p.is_file()),
            key=order,
        )
        for relative in files:
            destination = f"{base}/{relative.as_posix()}"
            length = (source / relative).stat().st_size
            info = remote.get_file_info(destination)
            if (
                relative.parts[0] != "_refs"
                and info.type == fs.FileType.File
                and info.size == length
            ):
                skipped += 1
                continue
            remote.create_dir(destination.rsplit("/", 1)[0], recursive=True)
            fs.copy_files(
                str(source / relative),
                destination,
                source_filesystem=local,
                destination_filesystem=remote,
            )
            copied, size = copied + 1, size + length
    mirrored = artifact.replace(roots={**artifact.location_roots(), alias: target_root})
    open_artifact(mirrored, verify_count=True)
    return {"target": target, "copied": copied, "skipped": skipped, "bytes": size}


def inspect_table(artifact):
    """Report published (tagged) and unpublished versions of a table."""
    lance, _ = _dependencies()
    dataset = lance.dataset(artifact.table_uri)
    published = sorted(set(_published(dataset).values()))
    newest = max(published, default=0)
    return {
        "latest": dataset.version,
        "published": published,
        "unpublished": [
            v["version"] for v in dataset.versions() if v["version"] > newest
        ],
        "tags": _published(dataset),
    }


def _digest(record):
    return hashlib.sha256(canonical_json(record.to_dict()).encode()).digest()


def verify_equivalence(artifact, source):
    """Streaming exact ID/fact comparison with a disk-backed uniqueness index."""
    dataset = open_artifact(artifact, verify_count=True)
    del dataset
    with (
        tempfile.TemporaryDirectory(prefix="lance-verify-") as directory,
        sqlite3.connect(Path(directory) / "facts.sqlite") as db,
    ):
        # Digests instead of payloads keep the full-scale index small.
        db.execute("CREATE TABLE facts (id TEXT PRIMARY KEY, digest BLOB)")
        count = 0
        for record in load_records(source):
            try:
                db.execute(
                    "INSERT INTO facts VALUES (?, ?)",
                    (record.id, _digest(record)),
                )
            except sqlite3.IntegrityError as exc:
                raise ContractError(f"duplicate JSONL ID: {record.id}") from exc
            count += 1
        for record in read_lance(artifact):
            found = db.execute(
                "SELECT digest FROM facts WHERE id=?", (record.id,)
            ).fetchone()
            if found is None or found[0] != _digest(record):
                raise ContractError(f"Lance/JSONL record mismatch: {record.id}")
            db.execute("DELETE FROM facts WHERE id=?", (record.id,))
        if db.execute("SELECT count(*) FROM facts").fetchone()[0]:
            raise ContractError("Lance/JSONL ID sets differ")
    return {"record_count": count, "ids_equal": True, "records_equal": True}


def rebuild_artifact(artifact, destination):
    """Rebuild entirely from the original JSONL and canonical Layers, without old table."""

    metadata = artifact.rebuild_metadata

    def source_records():
        digest = hashlib.sha256()
        for record in load_records(artifact.local(metadata["source_jsonl"])):
            digest.update((canonical_json(record.to_dict()) + "\n").encode())
            yield record
        if digest.hexdigest() != metadata["source_records_sha256"]:
            raise ContractError("rebuild source JSONL hash mismatch")

    records = source_records()
    previous_layers = []
    for layer in metadata["layers"]:
        manifest, _, _ = read_layer(artifact.local(layer))
        parent_metadata = manifest["parent"]["rebuild_metadata"]
        if (
            parent_metadata["source_records_sha256"]
            != metadata["source_records_sha256"]
            or parent_metadata["layers"] != previous_layers
        ):
            raise ContractError("rebuild Layer parent lineage mismatch")
        records = replay_layer(records, artifact.local(layer))
        previous_layers.append(layer)
    with tempfile.TemporaryDirectory(prefix="lance-rebuild-") as directory:
        source = Path(directory) / "records.jsonl"
        write_records(records, source)
        # Check count before publication, including a rebuild from an empty result.
        if sum(1 for _ in load_records(source)) != artifact.record_count:
            raise ContractError("rebuild record count mismatch")
        rebuilt = import_jsonl(
            source,
            destination,
            source_view=artifact.source_view,
            rebuild_metadata=metadata,
            roots=artifact.location_roots(),
        )
    return rebuilt


def rebuild_layer(layer, destination, *, source_view, roots=None):
    """Recover a failed query-layer write using only durable processing facts."""

    manifest, _, step = read_layer(layer)
    parent = LanceArtifact(**manifest["parent"], roots=roots)
    metadata = {
        **parent.rebuild_metadata,
        "layers": [
            *parent.rebuild_metadata["layers"],
            _location(layer, parent.location_roots()),
        ],
        "writes": sorted(set(parent.rebuild_metadata["writes"]) | set(step.writes)),
    }
    intended = parent.replace(
        source_view=source_view,
        record_count=manifest["result_record_count"],
        rebuild_metadata=metadata,
    )
    return rebuild_artifact(intended, destination)
