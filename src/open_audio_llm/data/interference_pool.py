"""Interferers named by record: a source file plus an optional start and duration.

A pool registered as ``records`` + ``audio_index`` points each clip at a span of its
source file, so long recordings are read by offset and no copy of the audio is needed.
"""

from __future__ import annotations

import gzip
import json

from audio_data_contract import AudioRef, resolve_split


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


def pool_clips(resolver, spec, split_name):
    """Yield a PoolClip per record of the split's ``records`` artifacts."""
    base = (spec.dataset_id, spec.version, split_name)
    for path in resolve_split(resolver.catalog, spec.dataset_id, spec.version, split_name,
                              "records", resolver.roots):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    ref = json.loads(line)["audio_slots"][0]["ref"]
                    yield PoolClip(resolver, base, ref["cut_id"], ref.get("start"), ref.get("duration"))
