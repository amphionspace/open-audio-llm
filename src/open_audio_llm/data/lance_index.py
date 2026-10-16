"""Record sources served from a registered Lance record index (``records_lance``).

Replaces the per-source JSONL cache: the sampler arrays (duration, slot count) come from
one scan of the index and a record is read by row position when it is needed. The scan
runs in a spawn process because Lance starts runtime threads that must not exist before
DataLoader workers fork; each worker opens its own handle on first use.

A source is served from the index only when every key of its config is understood here.
Anything else keeps the JSONL cache, so enabling ``record_index: lance`` never changes the
meaning of a source.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

ROLE = "records_lance"
# Source keys this reader implements; any other key keeps the JSONL path.
SUPPORTED_KEYS = {
    "dataset_id", "version", "split", "weight", "task",
    "exclude_records", "min_duration", "max_duration", "online_mix", "nonspeech",
}
ROW_COLUMNS = [
    "id", "task", "language", "text", "cut_id", "root_alias", "relative_path",
    "start_frame", "num_frames", "sample_rate", "channels", "channel", "slots", "record_json",
]


def _expression(split, min_duration, max_duration):
    parts = ["array_has(splits, '" + split.replace("'", "''") + "')"]
    if min_duration:
        parts.append(f"duration >= {float(min_duration)!r}")
    if max_duration is not None and max_duration != float("inf"):
        parts.append(f"duration <= {float(max_duration)!r}")
    return " AND ".join(parts)


def _scan_many(catalog_path, roots_path, requests):
    """Scan every requested split in one spawn process: imports and the catalog load once."""
    from audio_data_contract import load_catalog
    from audio_data_contract.roots import load_roots

    catalog, roots = load_catalog(catalog_path), load_roots(roots_path)
    return [_scan(catalog, roots, request) for request in requests]


def _scan(catalog, roots, args):
    """Read the arrays the sampler needs for one split."""
    import pyarrow.compute as pc
    from audio_data_contract.record_index import open_index

    (dataset_id, version, artifact, split, min_duration, max_duration, allowed) = args
    index = open_index(catalog, dataset_id, version, artifact, roots)
    dataset = index.dataset
    expression = _expression(split, min_duration, max_duration)
    table = dataset.to_table(columns=["id", "task", "slots"], filter=expression, with_row_address=True)
    # Row addresses are (fragment id << 32) | row-in-fragment; Dataset.take wants positions.
    starts, total = {}, 0
    for fragment in sorted(dataset.get_fragments(), key=lambda f: f.fragment_id):
        starts[fragment.fragment_id] = total
        total += fragment.count_rows()
    addresses = table["_rowaddr"].to_numpy()
    base = np.vectorize(starts.__getitem__, otypes=[np.int64])(addresses >> 32)
    positions = base + (addresses & 0xFFFFFFFF)
    slots = table["slots"].combine_chunks()
    counts = pc.list_value_length(slots).to_numpy().astype(np.int64)
    durations = pc.list_flatten(slots).field("duration").to_numpy(zero_copy_only=False)
    begin = np.cumsum(counts) - counts
    sums = np.add.reduceat(durations, begin) if len(counts) else durations
    is_ts = pc.equal(table["task"], "ts_asr").to_numpy(zero_copy_only=False)
    keep = np.ones(len(positions), dtype=bool)
    if allowed is not None:
        ids = table["id"].to_pylist()
        keep = np.fromiter((allowed(i) for i in ids), dtype=bool, count=len(ids))
    null_rows = dataset.count_rows(filter=expression + " AND record_json IS NULL")
    kind = "lhotse" if null_rows == len(positions) else "records" if null_rows == 0 else "mixed"
    hotwords = set()
    if kind == "records":
        from .hotwords import is_valid_hotword

        nonempty = dataset.to_table(
            columns=["id", "record_json"],
            filter=expression + " AND NOT (record_json LIKE '%\"hotwords\": []%')")
        for ident, text in zip(nonempty["id"].to_pylist(), nonempty["record_json"].to_pylist()):
            if allowed is None or allowed(ident):
                hotwords.update(h for h in json.loads(text).get("hotwords", ()) if is_valid_hotword(h))
    return {
        "positions": positions[keep], "duration_sum": sums[keep], "slot_count": counts[keep],
        "is_ts": is_ts[keep], "kind": kind, "hotwords": sorted(hotwords),
        "uri": str(index.manifest_path.parent / index.manifest["table_path"]),
        "snapshot": int(index.manifest["snapshot_version"]),
        "sha256": spec_sha(catalog, dataset_id, version, artifact),
    }


def spec_sha(catalog, dataset_id, version, artifact):
    return catalog.get(dataset_id, version).artifact(artifact).sha256


class LanceSourceIndex:
    """Same surface as SourceRecordIndex: len, row access, metadata arrays, hotwords."""

    @classmethod
    def open_many(cls, dataset, sources):
        """One index per source (None where the JSONL cache must be used), scanned together."""
        pending = []
        for number, source in enumerate(sources):
            spec = dataset.resolver.catalog.get(source["dataset_id"], source["version"])
            names = spec.splits[source["split"]].artifacts.get(ROLE)
            if names and not set(source) - SUPPORTED_KEYS and not dataset.augmentation.speed_prob:
                excluded_path = source.get("exclude_records")
                allowed = None
                if excluded_path:
                    # Lhotse rows are listed with their prefixed ID, like the JSONL path names them.
                    allowed = _Allowed(set(Path(excluded_path).read_text().split()), f"{spec.key}:{source['split']}:")
                request = (spec.dataset_id, spec.version, names[0], source["split"],
                           source.get("min_duration", 0), source.get("max_duration"), allowed)
                pending.append((number, source, spec, request))
        found = [None] * len(sources)
        if pending:
            context = multiprocessing.get_context("spawn")
            with context.Pool(1) as pool:
                scans = pool.apply(_scan_many, (
                    str(dataset.resolver.catalog_path), str(dataset.resolver.roots_path),
                    [request for *_, request in pending]))
            for (number, source, spec, _), scan in zip(pending, scans):
                found[number] = cls(dataset, source, spec, scan)
        return found

    @classmethod
    def open(cls, dataset, source):
        return cls.open_many(dataset, [source])[0]

    def __init__(self, dataset, source, spec, scan):
        self.source, self.spec, self.split = source, spec, source["split"]
        self.roots = {k: Path(v) for k, v in dataset.resolver.roots.items()}
        self.task = source.get("task", "asr")
        excluded_path = source.get("exclude_records")
        if not len(scan["positions"]):
            raise ValueError(f"Empty source: {source}")
        self.kind, self.hotwords = scan["kind"], scan["hotwords"]
        if self.kind == "mixed":
            raise ValueError(f"{spec.key}:{self.split} mixes AudioRecord and Lhotse rows")
        self.positions, self.uri, self.snapshot = scan["positions"], scan["uri"], scan["snapshot"]
        self.message_format = dataset.message_format
        sampling_rate = dataset.sampling_rate
        # sampling_cost without speed perturbation: sum of slot durations plus one sample per slot.
        duration = scan["duration_sum"] + scan["slot_count"] / sampling_rate
        native_ts = (dataset.message_format == "qwen3_asr") & scan["is_ts"]
        slots = np.where(native_ts, 1, scan["slot_count"]).astype("<u2")
        self.metadata = {"duration": duration.astype("<f8"), "slots": slots}
        self.count = len(self.positions)
        identity = {
            "sha256": scan["sha256"], "snapshot": self.snapshot, "split": self.split,
            "source": {k: v for k, v in source.items() if k != "weight"},
            "exclude": [str(excluded_path), Path(excluded_path).stat().st_size] if excluded_path else None,
            "sampling_rate": sampling_rate, "message_format": dataset.message_format}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
        digest.update(self.metadata["duration"].tobytes())
        digest.update(self.metadata["slots"].tobytes())
        self.key = digest.hexdigest()
        self._dataset, self._pid = None, None

    def __len__(self):
        return self.count

    def _open(self):
        if self._pid != os.getpid():
            import lance

            self._dataset, self._pid = lance.dataset(self.uri, version=self.snapshot), os.getpid()
        return self._dataset

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        row = self._open().take([int(self.positions[index])], columns=ROW_COLUMNS).to_pylist()[0]
        return self._from_record(row) if row["record_json"] is not None else self._from_supervision(row)

    def _with_options(self, record):
        """Source options that travel with the record, as the JSONL path attaches them."""
        extra = {key: self.source[key] for key in ("online_mix", "nonspeech") if self.source.get(key) is not None}
        return replace(record, metadata={**record.metadata, **extra}) if extra else record

    def _from_record(self, row):
        from audio_data_contract import AudioRecord

        from .records import ResolvedAudioRecord

        record = AudioRecord.from_dict(json.loads(row["record_json"]))
        # Portable records may omit durations: the index holds the located ones, in slot order.
        slots = tuple(
            replace(slot, ref=replace(slot.ref, duration=located["duration"])) if slot.ref.duration is None else slot
            for slot, located in zip(record.audio_slots, row["slots"]))
        record = replace(record, audio_slots=slots)
        record = self._with_options(record)
        return ResolvedAudioRecord(record)

    def _from_supervision(self, row):
        from audio_data_contract import AudioRecord, AudioRef, AudioSlot

        from .records import ResolvedAudioRecord

        rate = row["sample_rate"]
        cut = supervision_cut(row, self.roots[row["root_alias"]] / row["relative_path"])
        ref = AudioRef(self.spec.dataset_id, self.spec.version, self.split, row["cut_id"],
                       duration=cut.duration, purpose="primary")
        language = row["language"] if row["language"] != "N/A" else self.spec.languages[0]
        record = AudioRecord(
            id=f"{self.spec.key}:{self.split}:{row['id']}", task=self.task,
            audio_slots=(AudioSlot("primary", ref),), target=row["text"], language=language)
        record = self._with_options(record)
        return ResolvedAudioRecord(record, cuts={"primary": cut})

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_dataset"], state["_pid"] = None, None
        return state


def supervision_cut(row, path):
    """The span of a long recording that a Lhotse supervision row names.

    The recording is declared one second longer than the span ends. A recording that ends
    exactly where the span does makes Lhotse read a span that starts at 0 as the whole file,
    and then reject the length.
    """
    from lhotse import AudioSource, Recording
    from lhotse.cut import MonoCut

    rate, start, frames = row["sample_rate"], row["start_frame"], row["num_frames"]
    recording = Recording(
        id=row["cut_id"], sources=[AudioSource("file", list(range(row["channels"] or 1)), str(path))],
        sampling_rate=rate, num_samples=start + frames + rate, duration=(start + frames) / rate + 1.0)
    return MonoCut(id=row["cut_id"], start=start / rate, duration=frames / rate,
                   channel=row["channel"] or 0, recording=recording)


class _Allowed:
    """True for IDs not in the exclusion list; Lhotse rows are listed with their prefixed ID."""

    def __init__(self, blocked, prefix):
        self.blocked, self.prefix = blocked, prefix

    def __call__(self, ident):
        return ident not in self.blocked and self.prefix + ident not in self.blocked
