"""Online prompt rendering and hotword sampling from Catalog records."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

import torch
from audio_data_contract import (
    AudioRecord,
    PromptAudio,
    PromptTemplate,
    PromptText,
    render_example,
)
from torch.utils.data import Dataset

from .hotwords import build_hotword_pool, sample_hotwords
from .records import ResolvedAudioRecord, to_swift_record

TASK_TEMPLATES = {
    "speaker_attributed_asr": PromptTemplate(
        "open-audio-llm/speaker-attributed-asr", "1",
        ((PromptText("Transcribe every speaker, grouped by speaker:"), PromptAudio("mixture")),),
    ),
    "asr": PromptTemplate(
        "open-audio-llm/asr",
        "1",
        ((PromptText("Transcribe the following audio:"), PromptAudio("primary")),),
    ),
    "asr_hotwords": PromptTemplate(
        "open-audio-llm/asr-hotwords",
        "1",
        (
            (
                PromptText("Transcribe the following audio.\nHotwords: {hotwords}"),
                PromptAudio("primary"),
            ),
        ),
    ),
    "ts_asr": PromptTemplate(
        "open-audio-llm/ts-asr",
        "1",
        (
            (
                PromptText("Given the speaker's voice:"),
                PromptAudio("enrollment"),
                PromptText(
                    "\nTranscribe what this speaker says in the following audio:"
                ),
                PromptAudio("mixture"),
            ),
        ),
    ),
    "ser": PromptTemplate(
        "open-audio-llm/ser",
        "1",
        (
            (
                PromptText("Recognize the speaker emotion in the following audio:"),
                PromptAudio("primary"),
            ),
        ),
    ),
    "sec": PromptTemplate(
        "open-audio-llm/sec",
        "1",
        (
            (
                PromptText(
                    "Classify the speaker emotion category of the following audio:"
                ),
                PromptAudio("primary"),
            ),
        ),
    ),
    "sepc": PromptTemplate(
        "open-audio-llm/sepc",
        "1",
        (
            (
                PromptText("Describe the speaker emotion and speaking style:"),
                PromptAudio("primary"),
            ),
        ),
    ),
    "sei": PromptTemplate(
        "open-audio-llm/sei",
        "1",
        (
            (
                PromptText("Classify the emotion intensity of the following audio:"),
                PromptAudio("primary"),
            ),
        ),
    ),
    "esc": PromptTemplate(
        "open-audio-llm/esc",
        "1",
        (
            (
                PromptText(
                    "Describe the background acoustic scene only and ignore spoken content:"
                ),
                PromptAudio("primary"),
            ),
        ),
    ),
}


def build_answer(record: AudioRecord, hotwords: str) -> str:
    if record.task in {"ser", "sec", "sei"} and record.labels:
        return str(next(iter(record.labels.values())))
    if record.task == "esc" and record.labels:
        return str(record.labels.get("caption") or next(iter(record.labels.values())))
    detected = []
    if hotwords != "N/A":
        candidates = {h.strip() for h in hotwords.split(",") if h.strip()}
        detected = [h for h in record.hotwords if h in candidates]
    return (
        f"Language: {record.language}\n"
        f"Hotwords: {','.join(detected) if detected else 'N/A'}\n"
        f"Transcription: {record.target}"
    )


def _render(
    resolved: ResolvedAudioRecord,
    *,
    hotwords: str,
    prompt_seed: int,
):
    record = resolved.record
    answer = build_answer(record, hotwords)
    template = TASK_TEMPLATES.get(record.task, TASK_TEMPLATES["asr"])
    example = render_example(
        replace(record, target=answer),
        template,
        seed=prompt_seed,
        context={"hotwords": hotwords},
    )
    return example, answer


class OnlineAudioDataset(Dataset):
    """Materialize swift-compatible records dynamically from stable facts."""

    def __init__(
        self,
        records: list[ResolvedAudioRecord],
        seed: int = 42,
        max_hotwords: int = 30,
        hotword_prompt_prob: float = 0.8,
        hotword_miss_prob: float = 0.0,
        training: bool = True,
        message_format: str = "generic",
    ):
        if message_format not in {"generic", "qwen3_asr"}:
            raise ValueError(f"Unsupported message format: {message_format}")
        self.message_format = message_format
        self.records = records
        self.training = training
        self.seed = seed
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.max_hotwords = max_hotwords
        self.hotword_prompt_prob = hotword_prompt_prob
        self.hotword_miss_prob = hotword_miss_prob
        self.hotword_pool = (
            records.hotword_pool if hasattr(records, "hotword_pool")
            else build_hotword_pool(self.records)
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(int(epoch) if self.training else 0)

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def _resolve_audio(self, resolved, rng):
        raise NotImplementedError("The Catalog dataset resolves audio slots on demand")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        epoch, record_idx, occurrence = (
            idx if isinstance(idx, tuple) else (self.epoch, idx, None)
        )
        resolved = self.records[record_idx]
        record = resolved.record
        seed = f"{self.seed}:{epoch}:{record_idx}"
        if occurrence is not None:
            seed += f":{occurrence}"
        rng = random.Random(seed)
        hotwords = (
            sample_hotwords(
                list(record.hotwords),
                self.hotword_pool,
                max_hotwords=self.max_hotwords,
                prompt_prob=self.hotword_prompt_prob,
                miss_prob=self.hotword_miss_prob,
                rng=rng,
            )
            if self.training
            else (",".join(record.hotwords) or "N/A")
        )
        example, answer = _render(
            resolved,
            hotwords=hotwords,
            prompt_seed=epoch,
        )
        sample = to_swift_record(
            example,
            self._resolve_audio(resolved, rng),
            solution=answer,
            candidate_hotwords=hotwords,
        )
        sample["duration"] = resolved.duration
        sample["dataset_id"] = resolved.dataset_id
        if self.message_format == "qwen3_asr":
            from .qwen3_asr import concat_ts_audio, native_messages

            sample["messages"] = native_messages(record, hotwords)
            sample["solution"] = sample["messages"][-1]["content"]
            if record.task == "ts_asr":
                sample["audios"] = [concat_ts_audio(sample["audios"], self.sampling_rate)]
                sample["audio_slot_count"] = 1
        return sample
