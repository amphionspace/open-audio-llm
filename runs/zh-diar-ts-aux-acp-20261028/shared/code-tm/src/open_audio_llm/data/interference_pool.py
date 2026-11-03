"""Interferers named by record: a source file plus an optional start and duration.

A pool registered as ``records`` + ``audio_index`` points each clip at a span of its
source file, so long recordings are read by offset and no copy of the audio is needed.
A record whose ref names a Lance release (``tts-lance-release``) instead of the pool
itself is read from that release by sample ID.
"""

from __future__ import annotations

import gzip
import json

from audio_data_contract import AudioRef, resolve_split

from .lance_source import SAMPLES_ROLE, open_lance_release, read_lance_audio


class PoolClip:
    """One interferer. Audio is resolved on demand, like any other slot."""

    __slots__ = ("resolver", "base", "cut_id", "start", "duration", "sampling_rate")

    def __init__(self, resolver, base, cut_id, start, duration):
        self.resolver, self.base, self.cut_id = resolver, base, cut_id
        self.start, self.duration, self.sampling_rate = start, duration, None

    @property
    def id(self):
        return self.cut_id

    def load_audio(self):
        dataset_id, version, split = self.base
        cut = self.resolver.get_cut(AudioRef(
            dataset_id, version, split, self.cut_id, start=self.start, duration=self.duration))
        self.sampling_rate = cut.sampling_rate
        return cut.load_audio()


class LanceClip:
    """One interferer stored whole in a Lance release, read by sample ID."""

    __slots__ = ("resolver", "release", "cut_id", "sampling_rate")

    def __init__(self, resolver, release, cut_id):
        self.resolver, self.release, self.cut_id, self.sampling_rate = resolver, release, cut_id, None

    @property
    def id(self):
        return self.cut_id

    def load_audio(self):
        release = open_lance_release(self.resolver.catalog, self.resolver.roots, *self.release)
        audio, self.sampling_rate = read_lance_audio(release, self.cut_id)
        return audio


def _lance_release_key(resolver, dataset_id, version, split_name):
    spec = resolver.catalog.get(dataset_id, version)
    names = spec.splits[split_name].artifacts.get(SAMPLES_ROLE)
    if not names:
        raise ValueError(f"pool clip names {dataset_id}@{version}:{split_name}, which is neither the pool nor a Lance release")
    return spec.dataset_id, spec.version, names[0]


def pool_clips(resolver, spec, split_name):
    """Yield a PoolClip or LanceClip per record of the split's ``records`` artifacts."""
    base = (spec.dataset_id, spec.version, split_name)
    releases = {}
    for path in resolve_split(resolver.catalog, spec.dataset_id, spec.version, split_name,
                              "records", resolver.roots):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    ref = json.loads(line)["audio_slots"][0]["ref"]
                    owner = (ref["dataset_id"], ref["version"], ref["split"])
                    if owner == base:
                        yield PoolClip(resolver, base, ref["cut_id"], ref.get("start"), ref.get("duration"))
                        continue
                    if ref.get("start") is not None or ref.get("duration") is not None:
                        raise ValueError("a Lance pool clip is a whole sample; it takes no start or duration")
                    if owner not in releases:
                        releases[owner] = _lance_release_key(resolver, *owner)
                    yield LanceClip(resolver, releases[owner], ref["cut_id"])
