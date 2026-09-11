"""Runtime record metadata and ordered ms-swift message conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from audio_data_contract import AudioContent, AudioExample, AudioRecord, TextContent


@dataclass(frozen=True)
class ResolvedAudioRecord:
    """Runtime view of a portable AudioRecord; audio is resolved on demand."""

    record: AudioRecord
    cuts: dict[str, Any] | None = None

    @property
    def dataset_id(self) -> str:
        return self.record.audio_slots[0].ref.dataset_id

    @property
    def duration(self) -> float | None:
        for preferred in ("primary", "mixture"):
            for slot in self.record.audio_slots:
                if slot.name == preferred:
                    return slot.ref.duration
        return self.record.audio_slots[-1].ref.duration

    @property
    def audio_slot_count(self) -> int:
        return len(self.record.audio_slots)


def to_swift_record(
    example: AudioExample,
    audio_paths: dict[str, str | bytes],
    *,
    solution: str | None = None,
    candidate_hotwords: str | None = None,
) -> dict[str, Any]:
    """Derive ordered ms-swift audio inputs (paths or in-memory WAV bytes)."""

    messages: list[dict[str, str]] = []
    audios: list[str | bytes] = []
    for message in example.messages:
        rendered: list[str] = []
        for block in message.content:
            if isinstance(block, TextContent):
                rendered.append(block.text)
            elif isinstance(block, AudioContent):
                try:
                    audios.append(audio_paths[block.slot])
                except KeyError as exc:
                    raise ValueError(
                        f"no resolved path for audio slot {block.slot!r} in {example.id}"
                    ) from exc
                rendered.append("<audio>")
        messages.append({"role": message.role, "content": "".join(rendered)})
    result: dict[str, Any] = {
        "messages": messages,
        "audios": audios,
        "task": example.task,
        "solution": solution,
        "candidate_hotwords": candidate_hotwords,
        "audio_slot_count": len(audios),
    }
    return result
