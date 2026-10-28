"""Compatibility exports; audio remains outside Lance and resolves through AudioRef."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .errors import ContractError
from .records import read_artifact, write_jsonl, write_records


def lhotse_cuts(records, *, slot, resolve_audio):
    """Export one named slot per record; custom.audio_record keeps every slot/fact.

    resolve_audio(record) returns the existing index resolver's per-slot mapping:
    path, sample_rate, channels (count), duration (whole recording, in seconds).
    No audio is copied. Enrollment/other slots remain explicit in custom metadata.
    """
    for record in records:
        selected = record.slot(slot)
        ref = selected.ref
        info = resolve_audio(record)[slot]
        sample_rate, channels = info["sample_rate"], info["channels"]
        start = ref.start or 0.0
        duration = (
            ref.duration if ref.duration is not None else info["duration"] - start
        )
        if start < 0 or duration <= 0 or start + duration > info["duration"] + 1e-6:
            raise ContractError(f"invalid Lhotse segment bounds: {record.id}")
        channel = ref.channel
        if channel is None:
            channel = 0 if channels == 1 else tuple(range(channels))
        selected_channels = list(channel) if isinstance(channel, tuple) else [channel]
        if not selected_channels or any(
            c < 0 or c >= channels for c in selected_channels
        ):
            raise ContractError(f"invalid Lhotse channel: {record.id}")
        channel = selected_channels if isinstance(channel, tuple) else channel
        recording_id = f"{ref.dataset_id}@{ref.version}/{ref.cut_id}"
        yield {
            "id": record.id,
            "start": start,
            "duration": duration,
            "channel": channel,
            "type": "MultiCut" if isinstance(channel, list) else "MonoCut",
            "recording": {
                "id": recording_id,
                "sources": [
                    {
                        "type": "file",
                        "channels": list(range(channels)),
                        "source": str(info["path"]),
                    }
                ],
                "sampling_rate": sample_rate,
                "num_samples": round(info["duration"] * sample_rate),
                "duration": info["duration"],
                "channel_ids": list(range(channels)),
            },
            "supervisions": [
                {
                    "id": record.id,
                    "recording_id": recording_id,
                    "start": 0.0,
                    "duration": duration,
                    "channel": channel,
                    "text": record.target,
                    "language": record.language,
                }
            ],
            "custom": {
                "audio_record": record.to_dict(),
                "audio_slot": slot,
                "hotwords": list(record.hotwords),
                "labels": record.labels,
                **(
                    {"clean": record.metadata["clean"]}
                    if "clean" in record.metadata
                    else {}
                ),
            },
        }


def export_artifact(
    artifact, destination, *, query=None, slot=None, resolve_audio=None
):
    """Atomically export JSONL, or Lhotse cuts when a named slot/resolver is supplied."""
    destination = Path(destination)
    if destination.exists():
        raise ContractError(f"export destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=".export-", suffix=destination.suffix, dir=destination.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        records = read_artifact(artifact, query)
        if slot is None:
            write_records(records, temporary)
        else:
            if resolve_audio is None:
                raise ContractError("Lhotse export requires an audio-index resolver")
            write_jsonl(
                lhotse_cuts(records, slot=slot, resolve_audio=resolve_audio), temporary
            )
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)
