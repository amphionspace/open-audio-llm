"""DATA-TTS-UNIFIED Lance releases: ASR records from scalar columns, audio on demand."""

from __future__ import annotations

import os
from dataclasses import replace
from io import BytesIO

from audio_data_contract import AudioSlot

from .records import ResolvedAudioRecord

# Split role registered by audio-data-contract for `tts-lance-release` artifacts.
SAMPLES_ROLE = "samples"
# Manifest fields read after open_release; the rest (emilia2@tts-unified-v0.1 carries a
# 135 MB speaker table) would stay resident in every DataLoader worker.
_MANIFEST_KEYS = ("table_path", "lance_version", "rows")
_RELEASES = {}


def open_lance_release(catalog, roots, dataset_id, version, artifact):
    """The pinned release, opened once per process.

    Keyed by PID: a Lance handle opened before a DataLoader fork must not be reused.
    """
    key = (os.getpid(), dataset_id, version, artifact)
    if key not in _RELEASES:
        from audio_data_contract.tts_lance import open_release

        release = open_release(catalog, dataset_id, version, artifact, roots)
        _RELEASES[key] = replace(release, manifest={k: release.manifest[k] for k in _MANIFEST_KEYS})
    return _RELEASES[key]


def lance_records(release, spec, split_name, predicate=None):
    """Yield ASR records kept by the contract's read rules; no audio is read.

    The template slot is named "primary" like Lhotse-backed ASR sources, and the
    ID carries the dataset key because sample IDs are only unique per release.
    `predicate` is a Lance SQL filter evaluated by the table scan, before rows
    reach Python, e.g. "language = 'zh'".
    """
    from audio_data_contract.tts_lance import iter_asr_samples

    for record, _ in iter_asr_samples(release, with_audio=False, filter=predicate):
        duration = record.metadata.get("duration_seconds")
        if duration is None:
            raise ValueError(f"{spec.key}: {record.id} has no duration_seconds")
        slot = record.audio_slots[0]
        yield ResolvedAudioRecord(replace(
            record,
            id=f"{spec.key}:{split_name}:{record.id}",
            audio_slots=(AudioSlot(
                "primary", replace(slot.ref, duration=float(duration)), "primary"),),
        ))


def read_lance_audio(release, sample_id):
    """Decode one embedded sample to channels-first float32 and its sample rate."""
    import soundfile as sf
    from audio_data_contract.tts_lance import read_audio

    audio, rate = sf.read(
        BytesIO(read_audio(release, [sample_id])[sample_id]),
        dtype="float32", always_2d=True,
    )
    return audio.T, rate
