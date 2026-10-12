"""DATA-TTS-UNIFIED Lance releases: ASR records from scalar columns, audio on demand."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO

from audio_data_contract import AudioSlot

from .records import ResolvedAudioRecord

# Split role registered by audio-data-contract for `tts-lance-release` artifacts.
SAMPLES_ROLE = "samples"


def lance_records(release, spec, split_name):
    """Yield ASR records kept by the contract's read rules; no audio is read.

    The template slot is named "primary" like Lhotse-backed ASR sources, and the
    ID carries the dataset key because sample IDs are only unique per release.
    """
    from audio_data_contract.tts_lance import iter_asr_samples

    for record, _ in iter_asr_samples(release, with_audio=False):
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
