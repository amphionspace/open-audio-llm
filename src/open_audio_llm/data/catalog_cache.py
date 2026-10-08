"""Read-only, indexed Catalog metadata shared by training ranks and workers."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import logging
import os
import socket
import struct
import sqlite3
import time
from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from audio_data_contract import AudioRecord, resolve_artifact
from lhotse.serialization import deserialize_item

from .hotwords import is_valid_hotword
from .records import ResolvedAudioRecord
from .sot import TIMESTAMP_FORMAT

_META = np.dtype([("offset", "<u8"), ("duration", "<f8"), ("slots", "<u2")])
_PACK = struct.Struct("<QdH")

# Polling interval while another rank builds a shared cache entry.
LOCK_RETRY_SECONDS = 0.5


def lock_exclusive(lock):
    """Wait for an exclusive flock.

    quarkfs (FUSE) answers a contended blocking flock with EAGAIN instead of
    waiting, so poll a non-blocking lock. The kernel releases it if the holder dies.
    """
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            time.sleep(LOCK_RETRY_SECONDS)


@contextmanager
def exclusive_lock(path):
    """Open (creating if needed) and exclusively lock a shared lock file.

    quarkfs can answer simultaneous creation of one file from two hosts with
    EEXIST even without O_EXCL; the file then exists, so opening again succeeds.
    """
    while True:
        try:
            # Opened outside `with` so FileExistsError from the locked body is not retried.
            lock = open(path, "a")  # noqa: SIM115
            break
        except FileExistsError:
            time.sleep(LOCK_RETRY_SECONDS)
    with lock:
        lock_exclusive(lock)
        yield



class AudioIndex:
    """Keep large shared audio indexes outside Python's cyclic object graph."""

    def __init__(self, source, root):
        key = hashlib.sha256(
            json.dumps(
                [str(source), source.stat().st_size, source.stat().st_mtime_ns]
            ).encode()
        ).hexdigest()
        self.path = Path(root).expanduser().resolve() / f"audio-{key}.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(self.path.with_suffix(".lock")):
            if not self.path.exists():
                # flock does not exclude other hosts on quarkfs; a per-process
                # temporary keeps concurrent builders from sharing one file.
                temporary = self.path.with_name(
                    f"{self.path.stem}.{socket.gethostname()}.{os.getpid()}.tmp"
                )
                temporary.unlink(missing_ok=True)
                db = sqlite3.connect(temporary)
                try:
                    db.execute(
                        "CREATE TABLE audio (id TEXT PRIMARY KEY, duration REAL, payload TEXT)"
                    )
                    opener = gzip.open if source.suffix == ".gz" else open
                    with opener(source, "rt", encoding="utf-8") as stream:
                        batch = []
                        for line in stream:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            batch.append((row["cut_id"], row["duration"], line))
                            if len(batch) == 10000:
                                db.executemany(
                                    "INSERT OR REPLACE INTO audio VALUES (?, ?, ?)",
                                    batch,
                                )
                                batch.clear()
                        db.executemany(
                            "INSERT OR REPLACE INTO audio VALUES (?, ?, ?)", batch
                        )
                    db.commit()
                finally:
                    db.close()
                os.replace(temporary, self.path)
        self._connection, self._pid = None, None

    def _db(self):
        if self._pid != os.getpid():
            if self._connection is not None:
                self._connection.close()
            self._connection = sqlite3.connect(
                self.path.as_uri() + "?mode=ro&immutable=1", uri=True
            )
            self._pid = os.getpid()
        return self._connection

    def __getitem__(self, key):
        row = (
            self._db()
            .execute("SELECT payload FROM audio WHERE id = ?", (key,))
            .fetchone()
        )
        if row is None:
            raise KeyError(key)
        return json.loads(row[0])

    def duration(self, key):
        row = (
            self._db()
            .execute("SELECT duration FROM audio WHERE id = ?", (key,))
            .fetchone()
        )
        if row is None:
            raise KeyError(key)
        return row[0]

    def __getstate__(self):
        return {"path": self.path, "_connection": None, "_pid": None}


READ_ROLES = ("cuts", "recordings", "supervisions", "records", "audio_index")


def _legacy_source_identity(dataset, source):
    spec = dataset.resolver.catalog.get(source["dataset_id"], source["version"])
    split = spec.splits[source["split"]]
    paths = []
    if source.get("sot_timestamps"):
        paths.append(Path(source["sot_alignment_index"]))
    if source.get("exclude_records"):
        # The ID list changes the indexed records; its size/mtime keys the cache.
        paths.append(Path(source["exclude_records"]))
    # Only roles the loader reads key the index; declared alternatives such as
    # punctuated or clean supervisions may be absent locally and never change it.
    for role in READ_ROLES:
        for artifact in split.artifacts.get(role, ()):
            paths.append(resolve_artifact(
                dataset.resolver.catalog,
                spec.dataset_id,
                spec.version,
                artifact,
                dataset.resolver.roots,
            ))
    return {
        "format": 1,
        **({"sot_timestamp_format": TIMESTAMP_FORMAT}
           if source.get("sot_timestamps") else {}),
        # Earlier portable indexes ignored require_clean_pass. Never reuse them
        # for clean-only training, even if their old source flag was true.
        **({"clean_record_filter": 1} if source.get("require_clean_pass") and
           "records" in split.artifacts else {}),
        "spec": spec.to_dict(),
        # Replay quotas do not change the indexed records or their audio cost.
        "source": {k: v for k, v in source.items() if k != "weight"},
        "roots": {k: str(v) for k, v in dataset.resolver.roots.items()},
        "files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths],
        "sampling_rate": dataset.sampling_rate,
        "message_format": dataset.message_format,
        "speed_prob": dataset.augmentation.speed_prob,
        "speed_factors": dataset.augmentation.speed_factors,
    }


def source_identity(dataset, source):
    identity = _legacy_source_identity(dataset, source)
    split = identity["spec"]["splits"][source["split"]]
    artifacts = {name for names in split["artifacts"].values() for name in names}
    aliases = {artifact["root_alias"] for artifact in identity["spec"]["artifacts"]
               if artifact["name"] in artifacts}
    identity["roots"] = {alias: identity["roots"][alias] for alias in sorted(aliases)}
    # Mapping traversal order is immaterial. Ordered shard lists remain in spec:
    # changing them still changes record order and therefore the cache identity.
    identity["files"] = sorted(identity["files"])
    return identity


def _identity_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _legacy_paths(dataset, source, root):
    identity = _legacy_source_identity(dataset, source)
    return [Path(root) / _identity_key(value) for value in
            (identity, {**identity, "source": source})]


def _reuse_completed(destination, candidates):
    if destination.exists():
        return
    for candidate in candidates:
        if (candidate / "complete.json").is_file():
            try:
                destination.symlink_to(candidate.resolve(), target_is_directory=True)
            except FileExistsError:
                pass
            return


def reuse_indexes(config, previous_config, root, message_format="qwen3_asr"):
    """Register old completed indexes only after comparing their actual inputs."""
    from types import SimpleNamespace
    from .augment import AugmentConfig
    from .catalog_resolver import LhotseCatalogAudioResolver

    def context(recipe):
        return SimpleNamespace(
            resolver=LhotseCatalogAudioResolver(recipe["catalog"], recipe["roots"]),
            sampling_rate=int(recipe.get("sampling_rate", 16000)),
            message_format=message_format,
            augmentation=AugmentConfig(**recipe.get("augmentation", {})),
        )

    previous, current = context(previous_config), context(config)
    candidates = {}
    for source in previous_config["train"] + previous_config.get("validation", []):
        key = _identity_key(source_identity(previous, source))
        candidates.setdefault(key, []).extend(
            [Path(root) / key, *_legacy_paths(previous, source, root)])
    reused = []
    for source in config["train"] + config.get("validation", []):
        key = _identity_key(source_identity(current, source))
        destination = Path(root) / key
        _reuse_completed(destination, candidates.get(key, []))
        if (destination / "complete.json").is_file():
            reused.append({"dataset_id": source["dataset_id"], "split": source["split"],
                           "key": key, "path": str(destination.resolve())})
    return reused


def sampling_cost(row, dataset):
    speed = dataset.augmentation
    slowest = min(1.0, *speed.speed_factors) if speed.speed_prob else 1.0
    if row.record.metadata.get("sot_output_format") == TIMESTAMP_FORMAT:
        slowest = 1.0
    native_ts = (
        getattr(dataset, "message_format", None) == "qwen3_asr"
        and row.record.task == "ts_asr"
    )
    total = 0.0
    for slot in row.record.audio_slots:
        if native_ts and slot.name == "enrollment":
            continue
        duration = slot.ref.duration
        if duration is None:
            duration = dataset.resolver.get_duration(slot.ref)
        total += duration / slowest + 1 / dataset.sampling_rate
    if native_ts:
        from .qwen3_asr import ENROLL_SECONDS

        total += ENROLL_SECONDS
    if row.record.metadata.get("enrollment_config") is not None:
        from .target_sot import EnrollmentConfig

        config = EnrollmentConfig(**row.record.metadata["enrollment_config"])
        # Add SEP and independent segment rounding slack to the upper bound.
        total += config.max_targets * (config.max_seconds + .16)
        return total, 1
    return total, 1 if native_ts else len(row.record.audio_slots)


class SourceRecordIndex:
    def __init__(self, dataset, source, root):
        identity = source_identity(dataset, source)
        self.key = _identity_key(identity)
        self.path = Path(root) / self.key
        Path(root).mkdir(parents=True, exist_ok=True)
        _reuse_completed(self.path, _legacy_paths(dataset, source, root))
        self.path.mkdir(parents=True, exist_ok=True)
        # The completion marker is written last. A failed build is never opened.
        with exclusive_lock(self.path / "build.lock"):
            if not (self.path / "complete.json").exists():
                self._build(dataset, source)
        info = json.loads((self.path / "complete.json").read_text())
        self.count = info["count"]
        self.hotwords = info["hotwords"]
        self.metadata = np.memmap(self.path / "metadata.bin", dtype=_META, mode="r")
        self._fd = None

    def _build(self, dataset, source):
        count, hotwords = 0, set()
        logging.getLogger(__name__).warning("Building Catalog index: %s", source)
        with (
            (self.path / "records.jsonl").open("wb") as records,
            (self.path / "metadata.bin").open("wb") as metadata,
            (self.path / "ids.jsonl").open("w") as ids,
        ):
            for row in dataset._source_records(source):
                cuts = {name: cut.to_dict() for name, cut in (row.cuts or {}).items()}
                for slot in row.record.audio_slots:
                    ref = slot.ref
                    key = (ref.dataset_id, ref.version, ref.split, ref.cut_id)
                    cut = dataset._audio_cuts.pop(key, None)
                    if cut is not None:
                        cuts[slot.name] = cut.to_dict()
                duration, slots = sampling_cost(row, dataset)
                if not np.isfinite(duration) or duration <= 0:
                    raise ValueError(f"Invalid sampling duration: {row.record.id}")
                metadata.write(_PACK.pack(records.tell(), duration, slots))
                payload = [row.record.to_dict(), cuts]
                records.write(
                    json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                    + b"\n"
                )
                ids.write(json.dumps(row.record.id) + "\n")
                hotwords.update(h for h in row.record.hotwords if is_valid_hotword(h))
                count += 1
                if count % 100000 == 0:
                    logging.getLogger(__name__).warning(
                        "Catalog index %s:%s: %d",
                        source["dataset_id"],
                        source["split"],
                        count,
                    )
        if not count:
            raise ValueError(f"Empty source: {source}")
        (self.path / "complete.json").write_text(
            json.dumps({"count": count, "hotwords": sorted(hotwords)})
        )

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        if self._fd is None:
            self._fd = os.open(self.path / "records.jsonl", os.O_RDONLY)
        start = int(self.metadata[index]["offset"])
        end = (
            int(self.metadata[index + 1]["offset"])
            if index + 1 < self.count
            else os.fstat(self._fd).st_size
        )
        record, cuts = json.loads(os.pread(self._fd, end - start, start))
        return ResolvedAudioRecord(
            AudioRecord.from_dict(record),
            {name: deserialize_item(cut) for name, cut in cuts.items()} or None,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fd"] = None
        state["metadata"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.metadata = np.memmap(self.path / "metadata.bin", dtype=_META, mode="r")

    def __del__(self, close=os.close):
        fd = getattr(self, "_fd", None)
        if fd is not None:
            close(fd)


class CatalogRecordIndex:
    def __init__(self, dataset, root):
        self.root = Path(root).expanduser().resolve()
        self.sources = [
            SourceRecordIndex(dataset, source, self.root) for source in dataset.sources
        ]
        self.ends = np.cumsum([len(s) for s in self.sources]).tolist()
        self.hotword_pool = sorted({h for s in self.sources for h in s.hotwords})
        self.durations = np.concatenate([s.metadata["duration"] for s in self.sources])
        self.slots = np.concatenate([s.metadata["slots"] for s in self.sources])

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        source = bisect_right(self.ends, index)
        start = self.ends[source - 1] if source else 0
        return self.sources[source][index - start]

    def signature(self, identity):
        """Stream the legacy JSON identity so existing checkpoint cursors still match."""
        key = hashlib.sha256(
            json.dumps(
                [identity, [s.key for s in self.sources]], sort_keys=True
            ).encode()
        ).hexdigest()
        path = self.root / f"signature-{key}.txt"
        with exclusive_lock(path.with_suffix(".lock")):
            if path.exists():
                return path.read_text().strip()
            digest = hashlib.sha256()
            digest.update(b"{")
            for number, name in enumerate(sorted([*identity, "records"])):
                if number:
                    digest.update(b", ")
                digest.update(json.dumps(name).encode() + b": ")
                if name != "records":
                    digest.update(json.dumps(identity[name], sort_keys=True).encode())
                    continue
                digest.update(b"[")
                first = True
                for source in self.sources:
                    with (source.path / "ids.jsonl").open() as ids:
                        for line, meta in zip(ids, source.metadata):
                            if not first:
                                digest.update(b", ")
                            first = False
                            digest.update(b"[" + line.strip().encode() + b", ")
                            digest.update(
                                json.dumps(float(meta["duration"])).encode() + b", "
                            )
                            digest.update(str(int(meta["slots"])).encode() + b"]")
                digest.update(b"]")
            digest.update(b"}")
            path.write_text(digest.hexdigest() + "\n")
            return digest.hexdigest()


def _prepare_source(config, source, message_format, training=True):
    from .catalog_dataset import CatalogSwiftDataset

    selected = {**config, "train" if training else "validation": [source]}
    dataset = CatalogSwiftDataset(selected, training=training, message_format=message_format)
    return source["dataset_id"], source["split"], len(dataset)


def main():
    import argparse
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from .catalog_dataset import read_data_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--message_format", default="qwen3_asr")
    parser.add_argument("--reuse-config", help="Previous recipe whose completed indexes may be reused")
    parser.add_argument("--preflight-report", type=Path,
                        help="Also check the real sampler and decode duration extremes before reporting success")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()
    config = read_data_config(args.data_config)
    if not config.get("metadata_cache"):
        parser.error("Set metadata_cache in data config")
    if args.workers <= 0:
        parser.error("workers must be positive")
    if args.batch_size <= 0 or args.world_size <= 0:
        parser.error("batch-size and world-size must be positive")
    reused = []
    if args.reuse_config:
        previous = read_data_config(args.reuse_config)
        reused = reuse_indexes(config, previous, config["metadata_cache"],
                               args.message_format)
        print("Verified reusable Catalog indexes:", len(reused), flush=True)
    sources = [(source, True) for source in config["train"]]
    sources += [(source, False) for source in config.get("validation", [])]
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(_prepare_source, config, source, args.message_format, training)
            for source, training in sources
        ]
        for future in as_completed(futures):
            print("Catalog index ready:", *future.result(), flush=True)
    if args.preflight_report:
        from .catalog_preflight import check_data

        report = check_data(config, message_format=args.message_format,
                            batch_size=args.batch_size, world_size=args.world_size)
        report["verified_reused_indexes"] = reused
        args.preflight_report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.preflight_report.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.preflight_report)
        print("Catalog data preflight passed:", args.preflight_report, flush=True)


if __name__ == "__main__":
    main()
