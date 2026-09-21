"""Read-only, indexed Catalog metadata shared by training ranks and workers."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import logging
import os
import struct
import sqlite3
from bisect import bisect_right
from pathlib import Path

import numpy as np
from audio_data_contract import AudioRecord, resolve_artifact
from lhotse.serialization import deserialize_item

from .hotwords import is_valid_hotword
from .records import ResolvedAudioRecord
from .sot import TIMESTAMP_FORMAT

_META = np.dtype([("offset", "<u8"), ("duration", "<f8"), ("slots", "<u2")])
_PACK = struct.Struct("<QdH")


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
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.path.exists():
                temporary = self.path.with_suffix(".tmp")
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


def source_identity(dataset, source):
    spec = dataset.resolver.catalog.get(source["dataset_id"], source["version"])
    split = spec.splits[source["split"]]
    paths = []
    if source.get("sot_timestamps"):
        paths.append(Path(source["sot_alignment_index"]))
    for name, value in split.items():
        names = (
            [value]
            if name.endswith("_artifact")
            else value
            if name.endswith("_artifacts")
            else []
        )
        for artifact in names:
            path = resolve_artifact(
                dataset.resolver.catalog,
                spec.dataset_id,
                spec.version,
                artifact,
                dataset.resolver.roots,
            )
            if name == "manifest_dir_artifact":
                prefix = split["manifest_prefix"]
                part = split.get("source_split", source["split"])
                paths.extend(
                    path / f"{prefix}_{kind}_{part}.jsonl.gz"
                    for kind in ("recordings", "supervisions")
                )
            else:
                paths.append(path)
    return {
        "format": 1,
        **({"sot_timestamp_format": TIMESTAMP_FORMAT}
           if source.get("sot_timestamps") else {}),
        # Earlier portable indexes ignored require_clean_pass. Never reuse them
        # for clean-only training, even if their old source flag was true.
        **({"clean_record_filter": 1} if source.get("require_clean_pass") and
           ("records_artifact" in split or "records_artifacts" in split) else {}),
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
        from .qwen3_asr import ENROLL_SECONDS, SILENCE_SECONDS

        total += ENROLL_SECONDS + SILENCE_SECONDS
    return total, 1 if native_ts else len(row.record.audio_slots)


class SourceRecordIndex:
    def __init__(self, dataset, source, root):
        identity = source_identity(dataset, source)
        self.key = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        self.path = Path(root) / self.key
        legacy_identity = {**identity, "source": source}
        legacy_key = hashlib.sha256(json.dumps(legacy_identity, sort_keys=True).encode()).hexdigest()
        legacy_path = Path(root) / legacy_key
        if legacy_path != self.path and (legacy_path / "complete.json").is_file():
            try:
                self.path.symlink_to(legacy_path.name, target_is_directory=True)
            except FileExistsError:
                pass
        self.path.mkdir(parents=True, exist_ok=True)
        # The completion marker is written last. A failed build is never opened.
        with (self.path / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
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
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
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
    args = parser.parse_args()
    config = read_data_config(args.data_config)
    if not (
        os.environ.get("AUDIO_DATA_METADATA_CACHE") or config.get("metadata_cache")
    ):
        parser.error("Set AUDIO_DATA_METADATA_CACHE or metadata_cache")
    if args.workers <= 0:
        parser.error("workers must be positive")
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


if __name__ == "__main__":
    main()
