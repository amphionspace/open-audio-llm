"""Resolve portable ``AudioRef`` objects to exact Lhotse cut audio."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from audio_data_contract import AudioRef, load_catalog, load_roots, resolve_artifact
from lhotse import (
    AudioSource,
    CutSet,
    Recording,
    RecordingSet,
    SupervisionSet,
    load_manifest_lazy,
)


class LhotseCatalogAudioResolver:
    """Resolve exact Lhotse cuts; optionally share indexed metadata on disk."""

    def __init__(
        self,
        catalog_path: str | Path,
        roots_path: str | Path,
        index_cache: str | Path | None = None,
    ) -> None:
        self.catalog = load_catalog(catalog_path)
        self.roots = load_roots(roots_path)
        self._cuts: dict[tuple[str, str, str], dict[str, object]] = {}
        self._audio_indexes = {}
        self.index_cache = index_cache
        from .pyav_backend import install_pyav_backend

        install_pyav_backend()

    def _load_split(self, ref: AudioRef) -> dict[str, object]:
        key = (ref.dataset_id, ref.version, ref.split)
        if key in self._cuts:
            return self._cuts[key]
        cuts = self.iter_cuts(*key)

        indexed: dict[str, object] = {cut.id: cut for cut in cuts}
        # AudioRecord cut IDs commonly originate from supervision IDs.
        for cut in cuts.trim_to_supervisions(keep_overlapping=False):
            indexed[cut.id] = cut
            for supervision in cut.supervisions:
                indexed[supervision.id] = cut
        self._cuts[key] = indexed
        return indexed

    def iter_cuts(self, dataset_id: str, version: str, split_name: str):
        """Read a Catalog split without decoding audio or writing WAV files."""
        spec = self.catalog.get(dataset_id, version)
        split = spec.splits[split_name]

        def paths(kind):
            names = split.get(f"{kind}_artifacts")
            if names is None:
                name = split.get(f"{kind}_artifact")
                names = [name] if name else []
            return [
                resolve_artifact(
                    self.catalog, spec.dataset_id, spec.version, name, self.roots
                )
                for name in names
            ]

        cut_paths = paths("cuts")
        if cut_paths:
            return CutSet.from_cuts(
                cut for path in cut_paths for cut in load_manifest_lazy(path)
            )
        recording_paths = paths("recordings")
        supervision_paths = paths("supervisions")
        if "manifest_dir_artifact" in split:
            directory = resolve_artifact(
                self.catalog,
                spec.dataset_id,
                spec.version,
                split["manifest_dir_artifact"],
                self.roots,
            )
            prefix = split["manifest_prefix"]
            source_split = split.get("source_split", split_name)
            recording_paths = [
                directory / f"{prefix}_recordings_{source_split}.jsonl.gz"
            ]
            supervision_paths = [
                directory / f"{prefix}_supervisions_{source_split}.jsonl.gz"
            ]
        if recording_paths:
            recordings = RecordingSet.from_recordings(
                rec for path in recording_paths for rec in load_manifest_lazy(path)
            )
            supervisions = (
                SupervisionSet.from_segments(
                    sup
                    for path in supervision_paths
                    for sup in load_manifest_lazy(path)
                )
                if supervision_paths
                else None
            )
            return CutSet.from_manifests(
                recordings=recordings, supervisions=supervisions
            )
        raise ValueError(
            f"{spec.key}:{split_name} has no supported Lhotse artifact mapping"
        )

    def get_cut(self, ref: AudioRef):
        spec = self.catalog.get(ref.dataset_id, ref.version)
        index_name = spec.splits[ref.split].get("audio_index_artifact")
        if index_name:
            row = self._index_row(ref, index_name)
            root = self.roots[row["root_alias"]]
            path = (root / row["relative_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"audio index path escapes root: {ref.cut_id}")
            cut = Recording(
                id=ref.cut_id,
                sources=[AudioSource("file", list(range(row["channels"])), str(path))],
                sampling_rate=row["sample_rate"],
                num_samples=row["num_frames"],
                duration=row["duration"],
            ).to_cut()
        else:
            indexed = self._load_split(ref)
            try:
                cut = indexed[ref.cut_id]
            except KeyError as exc:
                raise KeyError(
                    f"cut {ref.cut_id!r} not found in "
                    f"{ref.dataset_id}@{ref.version}:{ref.split}"
                ) from exc
        if ref.start is not None or ref.duration is not None:
            offset = float(ref.start or 0.0)
            duration = ref.duration or max(0.0, cut.duration - offset)
            cut = cut.truncate(offset=offset, duration=duration)
        return cut

    def get_duration(self, ref: AudioRef):
        """Read indexed duration without resolving an audio path or building a cut."""
        spec = self.catalog.get(ref.dataset_id, ref.version)
        index_name = spec.splits[ref.split].get("audio_index_artifact")
        if not index_name:
            return self.get_cut(ref).duration
        index = self._get_index(ref, index_name)
        duration = index.duration(ref.cut_id) if self.index_cache else index[ref.cut_id]["duration"]
        available = max(0.0, duration - (ref.start or 0.0))
        return min(ref.duration, available) if ref.duration is not None else available

    def _index_row(self, ref, index_name):
        return self._get_index(ref, index_name)[ref.cut_id]

    def _get_index(self, ref, index_name):
        key = (ref.dataset_id, ref.version, index_name)
        if key not in self._audio_indexes:
            path = resolve_artifact(self.catalog, *key, self.roots)
            if self.index_cache:
                from .catalog_cache import AudioIndex

                self._audio_indexes[key] = AudioIndex(path, self.index_cache)
            else:
                opener = gzip.open if path.suffix == ".gz" else open
                with opener(path, "rt", encoding="utf-8") as stream:
                    rows = (json.loads(line) for line in stream if line.strip())
                    self._audio_indexes[key] = {row["cut_id"]: row for row in rows}
        return self._audio_indexes[key]
