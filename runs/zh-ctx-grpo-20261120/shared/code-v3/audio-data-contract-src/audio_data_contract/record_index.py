"""Lance record index: per-record metadata and audio locations, without audio bytes.

Built once from a registered version's `records` + `audio_index`, or Lhotse
`recordings` + `supervisions`, so training loads metadata from one pinned snapshot
instead of parsing JSONL and converting the audio index at start-up. The JSONL
artifacts stay authoritative; the index is a read-only derivative of them.

    index = open_index(catalog, "emilia2_short", "short-20261014", "record_index", roots)
    table = to_table(index, split="train_zh", filter="duration <= 30")
    audio = resolve_root_path(row["root_alias"], row["relative_path"], roots, "audio")
    # read num_frames frames from start_frame at sample_rate
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from .catalog import resolve_artifact, verify_artifact_file
from .errors import ContractError
from .lance import STORAGE_VERSION, _dependencies
from .records import open_text
from .roots import portable_path, resolve_root_path
from .tts_lance import _sha256
from .types import AudioRecord

ARTIFACT_KIND = "lance-record-index"
MAPPING_VERSION = "record-index-lance/1.2"
TABLE_PATH = "table.lance"
MANIFEST_NAME = "index.json"
# Split role that names the index; check_index only checks splits that carry it.
ROLE = "records_lance"
# Fragments are the shard unit; 64Ki rows gives ~60 fragments per 4M records.
FRAGMENT_ROWS = 2**16
# Optional per-record quality facts copied from AudioRecord.metadata (null if absent):
# RMS loudness in dBFS, fraction of samples with |x| >= 0.999, and fraction of 25 ms
# frames above -50 dBFS (an energy gate, not voice activity detection).
QUALITY = ("loudness_dbfs", "clip_ratio", "voiced_ratio")
# Source spellings mapped to one gender vocabulary; unknown spellings fail the build.
GENDERS = {
    "male": "male",
    "male_masculine": "male",
    "m": "male",
    "female": "female",
    "female_feminine": "female",
    "f": "female",
    "non-binary": "other",
    "transgender": "other",
    "intersex": "other",
    "other": "other",
    "": None,
    "do_not_wish_to_say": None,
    "unknown": None,
}
# Columns read from TS labels (n_spk, target_present, sample_type, target_spk).
LABELS = ("n_spk", "target_present", "sample_type", "target_speaker")
# record_json carries every AudioRecord field (e.g. extra TS slots); read on demand.
DEFAULT_COLUMNS = (
    "id",
    "splits",
    "task",
    "language",
    "duration",
    "text",
    "speaker",
    "cut_id",
    "root_alias",
    "relative_path",
    "start_frame",
    "num_frames",
    "sample_rate",
    "channels",
    "channel",
    "slots",
    "source_group",
    *QUALITY,
    "recording_id",
    "gender",
    *LABELS,
)
COMPARED = (
    "splits",
    "duration",
    "cut_id",
    "root_alias",
    "relative_path",
    "start_frame",
    "num_frames",
    "slots",
    "speaker",
    "source_group",
    *QUALITY,
    "recording_id",
    "gender",
    *LABELS,
)
INDICES = (
    ("id", "BTREE"),
    ("splits", "LABEL_LIST"),
    ("task", "BITMAP"),
    ("language", "BITMAP"),
    ("duration", "BTREE"),
    ("speaker", "BTREE"),
    ("source_group", "BTREE"),
    ("recording_id", "BTREE"),
    ("target_speaker", "BTREE"),
    ("gender", "BITMAP"),
    ("n_spk", "BITMAP"),
    ("sample_type", "BITMAP"),
)


_SLOT_FIELDS = (
    "name",
    "purpose",
    "cut_id",
    "duration",
    "root_alias",
    "relative_path",
    "start_frame",
    "num_frames",
    "sample_rate",
    "channels",
    "channel",
)


def _slot_type():
    """Every audio slot of a record (e.g. TS mixture and enrollment) with its location."""
    _, pa = _dependencies()
    return pa.struct(
        [
            pa.field("name", pa.string(), nullable=False),
            pa.field("purpose", pa.string()),
            pa.field("cut_id", pa.string(), nullable=False),
            pa.field("duration", pa.float64(), nullable=False),
            pa.field("root_alias", pa.string(), nullable=False),
            pa.field("relative_path", pa.string(), nullable=False),
            pa.field("start_frame", pa.int64(), nullable=False),
            pa.field("num_frames", pa.int64(), nullable=False),
            pa.field("sample_rate", pa.int32(), nullable=False),
            pa.field("channels", pa.int16()),
            pa.field("channel", pa.int16()),
        ]
    )


def _schema():
    _, pa = _dependencies()
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("splits", pa.list_(pa.string()), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("language", pa.string(), nullable=False),
            pa.field("duration", pa.float64(), nullable=False),
            pa.field("text", pa.large_string(), nullable=False),
            pa.field("speaker", pa.string()),
            pa.field("cut_id", pa.string(), nullable=False),
            pa.field("root_alias", pa.string(), nullable=False),
            pa.field("relative_path", pa.string(), nullable=False),
            pa.field("start_frame", pa.int64(), nullable=False),
            pa.field("num_frames", pa.int64(), nullable=False),
            pa.field("sample_rate", pa.int32(), nullable=False),
            pa.field("channels", pa.int16()),
            pa.field("channel", pa.int16()),
            pa.field("slots", pa.list_(_slot_type()), nullable=False),
            pa.field("source_group", pa.string()),
            *(pa.field(name, pa.float64()) for name in QUALITY),
            pa.field("recording_id", pa.string()),
            pa.field("gender", pa.string()),
            pa.field("n_spk", pa.int16()),
            pa.field("target_present", pa.bool_()),
            pa.field("sample_type", pa.string()),
            pa.field("target_speaker", pa.string()),
            pa.field("record_json", pa.large_string()),
        ],
        metadata={b"mapping": MAPPING_VERSION.encode()},
    )


@cache
def schema_hash():
    text = json.dumps({"mapping": MAPPING_VERSION, "arrow": str(_schema())})
    return hashlib.sha256(text.encode()).hexdigest()


def _lines(path: Path):
    with open_text(path, "r") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ContractError(f"{path}:{number}: {exc}") from exc


# Source location: (root_alias, relative_path, sample_rate, channels, num_frames, duration)
def _audio_index_locations(path, wanted=None):
    locations = {}
    for row in _lines(path):
        if wanted is not None and row["cut_id"] not in wanted:
            continue
        rate = row["sample_rate"]
        frames = row.get("num_frames")
        if frames is None:
            frames = round(row["duration"] * rate)
        locations[row["cut_id"]] = (
            row["root_alias"],
            row["relative_path"],
            rate,
            row.get("channels"),
            frames,
            row.get("duration", frames / rate),
        )
    return locations


def _portable(path, roots, prefer):
    """Location under the preferred alias when possible; equal-depth roots otherwise
    tie by name (e.g. aidc_data vs shared_storage for the same directory)."""
    if prefer in roots:
        root = Path(roots[prefer]).resolve()
        resolved = Path(path).resolve()
        if root in resolved.parents:
            return {"root_alias": prefer, "relative_path": resolved.relative_to(root).as_posix()}
    return portable_path(path, roots)


def _recording_locations(path, roots, prefer, wanted=None):
    locations = {}
    for row in _lines(path):
        if wanted is not None and row["id"] not in wanted:
            continue
        sources = row["sources"]
        if len(sources) != 1 or sources[0]["type"] != "file":
            raise ContractError(f"recording {row['id']}: only one file source is supported")
        location = _portable(sources[0]["source"], roots, prefer)
        locations[row["id"]] = (
            location["root_alias"],
            location["relative_path"],
            row["sampling_rate"],
            len(row.get("channel_ids") or sources[0]["channels"]),
            row["num_samples"],
            row["duration"],
        )
    return locations


def _channel(value, where):
    if value is None or type(value) is int:
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    raise ContractError(f"{where}: multi-channel selections are not supported")


def _locate(location, start, duration):
    alias, relative, rate, channels, frames, total = location
    if start is None:
        return {
            "duration": total if duration is None else duration,
            "root_alias": alias,
            "relative_path": relative,
            "start_frame": 0,
            "num_frames": frames,
            "sample_rate": rate,
            "channels": channels,
        }
    if duration is None:
        duration = total - start
    return {
        "duration": duration,
        "root_alias": alias,
        "relative_path": relative,
        "start_frame": round(start * rate),
        "num_frames": round(duration * rate),
        "sample_rate": rate,
        "channels": channels,
    }


def _primary(record: AudioRecord):
    """The slot the target describes: the only slot, else `primary`, else a TS `mixture`."""
    if len(record.audio_slots) == 1:
        return record.audio_slots[0]
    for main in ("primary", "mixture"):
        found = [s for s in record.audio_slots if main in (s.purpose, s.name)]
        if len(found) == 1:
            return found[0]
    raise ContractError(f"record {record.id}: cannot choose one main audio slot")


def _slot(record, slot, locations):
    ref = slot.ref
    location = locations.get(ref.cut_id)
    if location is None:
        raise ContractError(f"record {record.id}: cut {ref.cut_id} not in audio_index")
    return {
        "name": slot.name,
        "purpose": slot.purpose,
        "cut_id": ref.cut_id,
        "channel": _channel(ref.channel, record.id),
        **_locate(location, ref.start, ref.duration),
    }


def _quality(record):
    values = {}
    for name in QUALITY:
        value = record.metadata.get(name)
        if value is not None and (type(value) is bool or not isinstance(value, (int, float))):
            raise ContractError(f"record {record.id}: metadata.{name} must be a number")
        values[name] = None if value is None else float(value)
    return values


def _gender(value, where):
    if value is None:
        return None
    key = str(value).strip().lower()
    if key not in GENDERS:
        raise ContractError(f"{where}: unknown gender {value!r}")
    return GENDERS[key]


def _string(value):
    return value if isinstance(value, str) and value else None


def _labels(record):
    """TS facts from labels only; metadata.sample_type (hw_sft) means something else."""
    labels = record.labels
    n_spk = labels.get("n_spk")
    present = labels.get("target_present")
    if n_spk is not None and type(n_spk) is not int:
        raise ContractError(f"record {record.id}: labels.n_spk must be an integer")
    if present is not None and type(present) is not bool:
        raise ContractError(f"record {record.id}: labels.target_present must be boolean")
    return {
        "n_spk": n_spk,
        "target_present": present,
        "sample_type": _string(labels.get("sample_type")),
        "target_speaker": _string(labels.get("target_spk")),
    }


def _record_row(splits, item, locations):
    record = AudioRecord.from_dict(item)
    slots = [_slot(record, slot, locations) for slot in record.audio_slots]
    main = slots[record.audio_slots.index(_primary(record))]
    speaker = record.metadata.get("speaker")
    group = record.metadata.get("source_group")
    return {
        "id": record.id,
        "splits": splits,
        "task": record.task,
        "language": record.language,
        "text": record.target,
        "speaker": speaker if isinstance(speaker, str) else None,
        "source_group": group if isinstance(group, str) else None,
        **_quality(record),
        "recording_id": _string(record.metadata.get("recording_id")),
        "gender": _gender(record.metadata.get("gender"), f"record {record.id}"),
        **_labels(record),
        "slots": slots,
        "record_json": json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True),
        **{key: main[key] for key in _SLOT_FIELDS if key not in ("name", "purpose")},
    }


def _supervision_row(splits, task, item, locations):
    location = locations.get(item["recording_id"])
    if location is None:
        raise ContractError(f"supervision {item['id']}: unknown recording")
    main = {
        "cut_id": item["id"],
        "channel": _channel(item.get("channel"), item["id"]),
        **_locate(location, item["start"], item["duration"]),
    }
    return {
        "id": item["id"],
        "splits": splits,
        "task": task,
        "language": item.get("language") or "N/A",
        "text": item.get("text") or "",
        "speaker": item.get("speaker"),
        "source_group": None,
        **dict.fromkeys(QUALITY),
        "recording_id": item["recording_id"],
        "gender": _gender(
            item.get("gender", (item.get("custom") or {}).get("gender")),
            f"supervision {item['id']}",
        ),
        **dict.fromkeys(LABELS),
        "slots": [{"name": "primary", "purpose": "primary", **main}],
        "record_json": None,
        **main,
    }


def _plan(spec):
    """Per split: (row kind, source artifact names, location artifact names)."""
    plan = _buildable(spec)
    if not plan:
        raise ContractError(f"{spec.key}: no split has records+audio_index or Lhotse roles")
    return plan


def _buildable(spec):
    plan = {}
    for name, split in spec.splits.items():
        roles = split.artifacts
        if "records" in roles and "audio_index" in roles:
            plan[name] = ("records", roles["records"], roles["audio_index"])
        elif "supervisions" in roles and "recordings" in roles:
            plan[name] = ("supervisions", roles["supervisions"], roles["recordings"])
    return plan


def _sources(spec):
    """Per source artifact, in declaration order: (kind, location names, splits).

    A record is stored once even when several splits (e.g. a union `train`)
    list its source artifact; `splits` names all of them.
    """
    sources = {}
    for split, (kind, names, location_names) in _plan(spec).items():
        for name in names:
            known = sources.setdefault(name, (kind, location_names, []))
            if known[:2] != (kind, location_names):
                raise ContractError(f"{spec.key}:{name}: splits pair it with other sources")
            known[2].append(split)
    return sources


def _task(spec, splits):
    tasks = {spec.splits[split].task for split in splits}
    if tasks == {None} and len(spec.tasks) == 1:
        tasks = set(spec.tasks)
    if len(tasks) != 1 or None in tasks:
        raise ContractError(f"{spec.key}:{splits}: split task is ambiguous")
    return tasks.pop()


def _source_path(spec, name, roots):
    ref = spec.artifact(name)
    return ref, resolve_root_path(ref.root_alias, ref.relative_path, roots, f"{spec.key}:{name}")


def _locations(spec, kind, names, roots, wanted=None):
    locations = {}
    for name in names:
        _, path = _source_path(spec, name, roots)
        if kind == "records":
            locations.update(_audio_index_locations(path, wanted))
        else:
            locations.update(_recording_locations(path, roots, spec.artifact(name).root_alias, wanted))
    return locations


def _rows(spec, splits, kind, item, locations):
    if kind == "records":
        return _record_row(splits, item, locations)
    return _supervision_row(splits, _task(spec, splits), item, locations)


def build_index(catalog, dataset_id, version, destination, roots, *, batch_size=65536):
    """Write a new index directory for every buildable split; publish it atomically.

    Every source artifact is checked against its catalog digest first, and its
    digest is recorded in index.json. Nothing is published if any row fails.
    """
    lance, pa = _dependencies()
    spec = catalog.get(dataset_id, version)
    plan = _sources(spec)
    destination = Path(destination).resolve()
    if destination.exists():
        raise ContractError(f"destination already exists: {destination}")
    sources = []
    for name, (_, location_names, _) in plan.items():
        for used in (name, *location_names):
            if used in {s["name"] for s in sources}:
                continue
            ref, path = _source_path(spec, used, roots)
            if ref.sha256 is not None:
                verify_artifact_file(ref, path)
            sources.append({"name": used, "kind": ref.kind, "sha256": ref.sha256})
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".record-index-", dir=destination.parent))
    counts = dict.fromkeys(_plan(spec), 0)
    try:
        cache_key, locations = None, {}
        # IDs are unique within a split; two source files (e.g. train and its
        # train_hotwords subset) may repeat an ID as separate rows.
        seen = {split: set() for split in counts}
        total = 0

        def batches():
            nonlocal cache_key, locations, total
            rows = []
            for name, (kind, location_names, splits) in plan.items():
                if cache_key != (kind, location_names):
                    locations = _locations(spec, kind, location_names, roots)
                    cache_key = (kind, location_names)
                _, path = _source_path(spec, name, roots)
                for item in _lines(path):
                    row = _rows(spec, splits, kind, item, locations)
                    for split in splits:
                        if row["id"] in seen[split]:
                            raise ContractError(f"duplicate record ID in {split}: {row['id']}")
                        seen[split].add(row["id"])
                        counts[split] += 1
                    rows.append(row)
                    total += 1
                    if len(rows) == batch_size:
                        yield pa.RecordBatch.from_pylist(rows, schema=_schema())
                        rows = []
            if rows:
                yield pa.RecordBatch.from_pylist(rows, schema=_schema())

        dataset = lance.write_dataset(
            batches(),
            work / TABLE_PATH,
            schema=_schema(),
            data_storage_version=STORAGE_VERSION,
            max_rows_per_file=FRAGMENT_ROWS,
        )
        for column, index_type in INDICES:
            dataset.create_scalar_index(column, index_type)
        if dataset.count_rows() != total:
            raise ContractError("record index row count mismatch")
        manifest = {
            "kind": ARTIFACT_KIND,
            "mapping": MAPPING_VERSION,
            "schema_hash": schema_hash(),
            "storage_version": STORAGE_VERSION,
            "dataset_id": spec.dataset_id,
            "version": spec.version,
            "table_path": TABLE_PATH,
            "snapshot_version": dataset.version,
            "record_count": total,
            "split_counts": counts,
            "sources": sources,
            "pylance": lance.__version__,
        }
        (work / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # mkdtemp is 0700; publish with the caller's umask so trainers can read it.
        umask = os.umask(0)
        os.umask(umask)
        work.chmod(0o777 & ~umask)
        work.rename(destination)
        return manifest
    except Exception as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"record index build failed; nothing published: {exc}") from exc
    finally:
        if work.exists():
            shutil.rmtree(work)


def catalog_ref(manifest_path, *, name, roots, root_alias=None):
    """Catalog ArtifactRef dict that pins index.json and the snapshot it names."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    return {
        "name": name,
        "kind": ARTIFACT_KIND,
        **_portable(path, roots, root_alias),
        "expected_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        # "rows", not "record_count": verify_artifact_file counts records in a JSON file.
        "metadata": {
            "snapshot_version": manifest["snapshot_version"],
            "rows": manifest["record_count"],
            "schema_hash": manifest["schema_hash"],
        },
    }


def register_index(catalog_path, dataset_id, version, ref, splits):
    """Add the index artifact and each indexed split's records_lance role to the
    YAML file that declares dataset_id@version; nothing else in it changes.
    A rebuilt index at the same location replaces the old artifact's pins in place.
    The whole catalog must still load afterwards, or the file is restored."""
    from .catalog import load_catalog
    from .declarations import declaration_files, edit_declaration

    def edit(row):
        old = [a for a in row["artifacts"] if a["name"] == ref["name"]]
        location = (ref["kind"], ref["root_alias"], ref["relative_path"])
        if old and (old[0]["kind"], old[0]["root_alias"], old[0]["relative_path"]) != location:
            raise ContractError(f"{dataset_id}@{version} already has {ref['name']!r}")
        for split in splits:
            roles = row["splits"][split]["artifacts"]
            if roles.get(ROLE, [ref["name"]]) != [ref["name"]]:
                raise ContractError(f"{dataset_id}@{version}:{split} already has {ROLE}")
        if old:
            old[0].update(ref)
        else:
            row["artifacts"].append(dict(ref))
        for split in splits:
            row["splits"][split]["artifacts"].setdefault(ROLE, [ref["name"]])

    for path in declaration_files(catalog_path):
        original = path.read_text(encoding="utf-8")
        if edit_declaration(path, dataset_id, version, edit):
            try:
                load_catalog(catalog_path)
            except ContractError:
                path.write_text(original, encoding="utf-8")
                raise
            return path
    raise ContractError(f"{dataset_id}@{version} is not declared in {catalog_path}")


def index_gaps(spec):
    """Splits that could be indexed but have no records_lance role."""
    missing = [
        split
        for split in _buildable(spec)
        if not spec.splits[split].artifacts.get(ROLE)
    ]
    if not missing:
        return []
    return [f"{spec.key}: no {ARTIFACT_KIND} for splits {', '.join(missing)}"]


@dataclass(frozen=True)
class RecordIndex:
    dataset_id: str
    version: str
    manifest_path: Path
    manifest: dict[str, Any]
    dataset: Any


def _open(spec, ref, path):
    lance, _ = _dependencies()
    if ref.kind != ARTIFACT_KIND:
        raise ContractError(f"artifact {ref.name!r} is not a {ARTIFACT_KIND}")
    if _sha256(path) != ref.sha256:
        raise ContractError(f"record index manifest sha256 differs from catalog: {path}")
    manifest = json.loads(path.read_text())
    pinned = {"snapshot_version": "snapshot_version", "record_count": "rows"}
    for key, ref_key in {**pinned, "schema_hash": "schema_hash"}.items():
        if manifest.get(key) != ref.metadata[ref_key]:
            raise ContractError(f"record index {key} differs from catalog: {path}")
    if (manifest["dataset_id"], manifest["version"]) != (spec.dataset_id, spec.version):
        raise ContractError(f"record index belongs to another dataset version: {path}")
    if manifest["schema_hash"] != schema_hash():
        raise ContractError(f"record index mapping differs from this reader: {path}")
    try:
        dataset = lance.dataset(
            str(path.parent / manifest["table_path"]), version=manifest["snapshot_version"]
        )
        rows = dataset.count_rows()
        same_schema = dataset.schema.equals(_schema(), check_metadata=True)
    except Exception as exc:
        raise ContractError(f"cannot open pinned record index for {path}: {exc}") from exc
    if not same_schema:
        raise ContractError(f"record index physical schema mismatch: {path}")
    if rows != manifest["record_count"]:
        raise ContractError(f"record index row count differs from manifest: {path}")
    return RecordIndex(spec.dataset_id, spec.version, path, manifest, dataset)


def open_index(catalog, dataset_id, version, artifact_name, roots) -> RecordIndex:
    """Open a registered index at exactly the manifest and snapshot it pins."""
    spec = catalog.get(dataset_id, version)
    path = resolve_artifact(catalog, dataset_id, version, artifact_name, roots)
    return _open(spec, spec.artifact(artifact_name), path)


def _expression(split, filter):
    parts = []
    if split is not None:
        parts.append("array_has(splits, '" + split.replace("'", "''") + "')")
    if filter:
        parts.append(f"({filter})")
    return " AND ".join(parts) or None


def to_table(index: RecordIndex, *, split=None, filter=None, columns=DEFAULT_COLUMNS):
    """All matching rows as one Arrow table; `filter` is a Lance SQL predicate."""
    return index.dataset.to_table(
        columns=list(columns), filter=_expression(split, filter)
    )


def iter_batches(
    index: RecordIndex,
    *,
    split=None,
    filter=None,
    columns=DEFAULT_COLUMNS,
    shard=0,
    num_shards=1,
    seed=0,
    epoch=0,
    batch_size=4096,
):
    """Arrow batches for this shard: whole fragments, reshuffled per (seed, epoch)."""
    if not 0 <= shard < num_shards:
        raise ContractError("shard must be in [0, num_shards)")
    fragments = sorted(index.dataset.get_fragments(), key=lambda f: f.fragment_id)
    random.Random(f"{seed}:{epoch}").shuffle(fragments)
    expression = _expression(split, filter)
    for fragment in fragments[shard::num_shards]:
        yield from fragment.to_batches(
            columns=list(columns), filter=expression, batch_size=batch_size
        )


def check_index(spec, ref, path, roots, *, sample=20, seed=0) -> list[str]:
    """Registration check: per-split counts equal the JSONL sources, and a seeded
    sample of source rows matches the index on duration and audio location."""
    label = f"{spec.key}:{ref.name}"
    try:
        index = _open(spec, ref, path)
    except ContractError as exc:
        return [f"{label}: {exc}"]
    problems = []
    rng = random.Random(seed)
    sources = _sources(spec)
    for split, (kind, names, location_names) in _plan(spec).items():
        if ref.name not in spec.splits[split].artifacts.get(ROLE, ()):
            continue
        chosen, seen = [], 0
        for name in names:
            _, source = _source_path(spec, name, roots)
            for item in _lines(source):
                seen += 1
                if len(chosen) < sample:
                    chosen.append((name, item))
                elif (slot := rng.randrange(seen)) < sample:
                    chosen[slot] = (name, item)
        stored = index.dataset.count_rows(filter=_expression(split, None))
        if stored != seen:
            problems.append(f"{label}:{split}: index has {stored} rows, sources have {seen}")
        if kind == "records":
            wanted = {
                slot.ref.cut_id
                for _, i in chosen
                for slot in AudioRecord.from_dict(i).audio_slots
            }
        else:
            wanted = {i["recording_id"] for _, i in chosen}
        locations = _locations(spec, kind, location_names, roots, wanted)
        expected = {}
        for name, item in chosen:
            try:
                row = _rows(spec, sources[name][2], kind, item, locations)
            except ContractError as exc:
                problems.append(f"{label}:{split}: {exc}")
                continue
            expected[row["id"]] = row
        if not expected:
            continue
        quoted = ", ".join("'" + i.replace("'", "''") + "'" for i in expected)
        found = {
            row["id"]: row
            for row in index.dataset.to_table(
                columns=["id", *COMPARED],
                filter=_expression(split, f"id IN ({quoted})"),
            ).to_pylist()
        }
        for record_id, row in expected.items():
            got = found.get(record_id)
            if got is None:
                problems.append(f"{label}:{split}: {record_id} missing from index")
                continue
            for key in COMPARED:
                if got[key] != row[key]:
                    problems.append(
                        f"{label}:{split}: {record_id} {key} {got[key]!r} != {row[key]!r}"
                    )
            try:
                audio = resolve_root_path(row["root_alias"], row["relative_path"], roots, label)
                with audio.open("rb") as stream:
                    if not stream.read(64):
                        problems.append(f"{label}:{split}: empty audio file: {audio}")
            except (ContractError, OSError) as exc:
                problems.append(f"{label}:{split}: {record_id}: {exc}")
    return problems
