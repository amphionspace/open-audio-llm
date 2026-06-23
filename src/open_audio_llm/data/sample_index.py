"""Offline sample-index schema.

The index stores stable sample facts. Training-time randomization belongs in
`LhotseSwiftDataset` and collators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class SampleIndexRecord:
    id: str
    dataset_id: str
    task: str
    audio_path: str
    start: float = 0.0
    duration: float | None = None
    text: str = ""
    language: str = "N/A"
    real_hotwords: list[str] = field(default_factory=list)
    enrollment_audio: str | None = None
    enrollment_start: float = 0.0
    enrollment_duration: float | None = None
    esc_background: dict[str, Any] | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleIndexRecord":
        return cls(**data)


def write_jsonl(records: Iterable[SampleIndexRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_json())
            f.write("\n")


def read_jsonl(path: str | Path) -> Iterator[SampleIndexRecord]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield SampleIndexRecord.from_dict(json.loads(line))


def to_sharegpt_record(record: SampleIndexRecord, prompt: str, answer: str) -> dict[str, Any]:
    audios = [record.audio_path]
    if record.enrollment_audio:
        audios = [record.enrollment_audio, record.audio_path]
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "audios": audios,
        "task": record.task,
        "solution": answer,
        "candidate_hotwords": ",".join(record.real_hotwords) if record.real_hotwords else "N/A",
    }
