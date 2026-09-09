"""Resolve portable ``AudioRef`` objects to exact Lhotse cut audio."""

from __future__ import annotations

from pathlib import Path

from audio_data_contract import AudioRef, load_catalog, load_roots, resolve_artifact
from lhotse import CutSet, RecordingSet, SupervisionSet, load_manifest_lazy


class LhotseCatalogAudioResolver:
    """Resolve Catalog audio references to exact Lhotse cuts without writing files."""

    def __init__(
        self,
        catalog_path: str | Path,
        roots_path: str | Path,
    ) -> None:
        self.catalog = load_catalog(catalog_path)
        self.roots = load_roots(roots_path)
        self._cuts: dict[tuple[str, str, str], dict[str, object]] = {}

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
