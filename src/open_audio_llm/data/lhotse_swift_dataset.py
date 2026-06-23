"""Dynamic ms-swift dataset backed by `sample_index.jsonl`."""

from __future__ import annotations

import random
from typing import Any

from torch.utils.data import Dataset

from .augment import apply_augmentations
from .hotwords import build_hotword_pool, sample_hotwords
from .sample_index import SampleIndexRecord, read_jsonl, to_sharegpt_record


TASK_PROMPTS = {
    "asr": "Transcribe the following audio:{speech}",
    "asr_hotwords": "Transcribe the following audio.\nHotwords: {hotwords}{speech}",
    "ts_asr": (
        "Given the speaker's voice:{speaker_speech}\n"
        "Transcribe what this speaker says in the following audio:{speech}"
    ),
    "ser": "Recognize the speaker emotion in the following audio:{speech}",
    "sec": "Classify the speaker emotion category of the following audio:{speech}",
    "sepc": "Describe the speaker emotion and speaking style:{speech}",
    "sei": "Classify the emotion intensity of the following audio:{speech}",
    "esc": "Describe the background acoustic scene only and ignore spoken content:{speech}",
}


def build_answer(record: SampleIndexRecord, hotwords: str) -> str:
    if record.task in {"ser", "sec", "sei"} and record.labels:
        return str(next(iter(record.labels.values())))
    if record.task == "esc" and record.labels:
        return str(record.labels.get("caption") or next(iter(record.labels.values())))
    detected = []
    if hotwords != "N/A":
        candidates = {h.strip() for h in hotwords.split(",") if h.strip()}
        detected = [h for h in record.real_hotwords if h in candidates]
    return (
        f"Language: {record.language}\n"
        f"Hotwords: {','.join(detected) if detected else 'N/A'}\n"
        f"Transcription: {record.text}"
    )


class LhotseSwiftDataset(Dataset):
    """Materialize swift-compatible records on the fly from sample facts."""

    def __init__(
        self,
        index_path: str,
        seed: int = 42,
        max_hotwords: int = 30,
        hotword_prompt_prob: float = 0.8,
        hotword_miss_prob: float = 0.0,
        augment_hooks: list | None = None,
    ):
        self.records = list(read_jsonl(index_path))
        self.seed = seed
        self.max_hotwords = max_hotwords
        self.hotword_prompt_prob = hotword_prompt_prob
        self.hotword_miss_prob = hotword_miss_prob
        self.augment_hooks = augment_hooks or []
        self.hotword_pool = build_hotword_pool(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        rng = random.Random((self.seed, idx, random.getrandbits(32)).__repr__())
        hotwords = sample_hotwords(
            record.real_hotwords,
            self.hotword_pool,
            max_hotwords=self.max_hotwords,
            prompt_prob=self.hotword_prompt_prob,
            miss_prob=self.hotword_miss_prob,
            rng=rng,
        )
        prompt = TASK_PROMPTS.get(record.task, TASK_PROMPTS["asr"])
        prompt = prompt.format(
            speech="<audio>",
            speaker_speech="<audio>",
            hotwords=hotwords,
        )
        answer = build_answer(record, hotwords)
        sample = to_sharegpt_record(record, prompt, answer)
        sample["duration"] = record.duration
        sample["dataset_id"] = record.dataset_id
        sample["task"] = record.task
        return apply_augmentations(sample, rng, self.augment_hooks)
