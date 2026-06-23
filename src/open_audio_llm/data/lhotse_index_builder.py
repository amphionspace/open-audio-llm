"""Build sample indexes from Lhotse-style JSONL manifests."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator

from .sample_index import SampleIndexRecord, write_jsonl


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_from_recordings_supervisions(
    recordings_path: str | Path,
    supervisions_path: str | Path,
    dataset_id: str,
    task: str = "asr",
    language: str = "N/A",
) -> list[SampleIndexRecord]:
    recordings = {rec["id"]: rec for rec in read_jsonl(recordings_path)}
    records: list[SampleIndexRecord] = []
    for sup in read_jsonl(supervisions_path):
        rec_id = sup.get("recording_id", sup.get("id"))
        rec = recordings.get(rec_id)
        if not rec:
            continue
        source = (rec.get("sources") or [{}])[0].get("source")
        if not source:
            continue
        custom = sup.get("custom") or {}
        records.append(
            SampleIndexRecord(
                id=sup.get("id", rec_id),
                dataset_id=dataset_id,
                task=custom.get("task", task),
                audio_path=source,
                start=float(sup.get("start", 0.0)),
                duration=sup.get("duration"),
                text=(sup.get("text") or "").strip(),
                language=custom.get("language", language),
                real_hotwords=[
                    h for h in custom.get("hotwords", []) if isinstance(h, str)
                ],
                enrollment_audio=custom.get("enrollment_audio"),
                esc_background=custom.get("esc_background"),
                labels={k: v for k, v in custom.items() if k.endswith("_label")},
                metadata={"recording_id": rec_id},
            )
        )
    return records


def build_index_file(
    recordings_path: str | Path,
    supervisions_path: str | Path,
    output_path: str | Path,
    dataset_id: str,
    task: str = "asr",
    language: str = "N/A",
) -> None:
    write_jsonl(
        build_from_recordings_supervisions(
            recordings_path,
            supervisions_path,
            dataset_id=dataset_id,
            task=task,
            language=language,
        ),
        output_path,
    )
