"""Catalog samples -> LongEnrollTSASR examples.

The Catalog dataset keeps doing what it does for every other model: record
selection, decoding, waveform augmentation, hotword sampling and the native
Qwen3-ASR answer text. This module only turns its output into the v5 inputs
(built by features.py, which inference shares):

  ts_asr            mixture mel + whole-enrollment mel + ECAPA fbank of the enrollment
                    head, v5 TS system prompt, mode BOTH
  asr / hotwords    mixture mel, the record's own system turn ("" / "Hotwords: ..."),
                    mode NONE; the enrollment path runs on the first seconds of the
                    mixture and is multiplied by zero (every parameter gets a gradient)
"""

from __future__ import annotations

import time
import zlib
from typing import Sequence

import numpy as np
import torch

from ..data.catalog_dataset import CatalogSwiftDataset
from .features import (  # noqa: F401 -- re-exported for the training loop and tests
    DUMMY_ENROLL_SECONDS,
    ECAPA_MIN_SECONDS,
    ECAPA_SECONDS,
    ECAPA_WHOLE_SECONDS,
    TASK_NAMES,
    TS_SYSTEM,
    LongEnrollFeatures,
    task_name,
)


class LongEnrollCatalogDataset(CatalogSwiftDataset):
    """CatalogSwiftDataset whose items are LongEnrollTSASR examples."""

    def __init__(self, config, processor, *, training=True, collect_metrics=False, ts_system=TS_SYSTEM):
        super().__init__(config, training=training, message_format="qwen3_asr",
                         collect_metrics=collect_metrics, waveforms=True)
        if training and self.augmentation.spec_aug_prob:
            raise ValueError("SpecAugment runs in the ms-swift template; LongEnrollTSASR does not apply it")
        self.features = LongEnrollFeatures(processor, ts_system)

    def __getitem__(self, idx):
        started = time.perf_counter()
        sample = super().__getitem__(idx)
        key = idx if isinstance(idx, tuple) else (self.epoch, idx, None)
        rng = (np.random.default_rng(zlib.crc32(f"{self.seed}:{key}:ecapa".encode()))
               if self.training else None)
        example = self.features(sample, rng)
        if example["_performance"] is not None:
            example["_performance"]["example_s"] = time.perf_counter() - started
        return example


def collate(examples: Sequence[dict], pad_id: int) -> dict:
    bsz = len(examples)
    max_len = max(len(e["input_ids"]) for e in examples)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attention = torch.zeros((bsz, max_len), dtype=torch.long)
    for row, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[row, :n] = torch.tensor(e["input_ids"])
        labels[row, :n] = torch.tensor(e["labels"])
        attention[row, :n] = 1
    performance = [e["_performance"] for e in examples if e.get("_performance")]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention,
        "mix_mels": [e["mix_mel"] for e in examples],
        "enroll_mels": [e["enroll_mel"] for e in examples],
        "ecapa_fbank": [e["ecapa_fbank"] for e in examples],
        "mode": torch.tensor([e["mode"] for e in examples], dtype=torch.long),
        "task": [e["task"] for e in examples],
        "answer": [e["answer"] for e in examples],
        "system": [e["system"] for e in examples],
        "n_audio": [e["n_audio"] for e in examples],
        "language": [e["language"] for e in examples],
        "mix_sec": torch.tensor([e["mix_sec"] for e in examples], dtype=torch.float32),
        "enroll_sec": torch.tensor([e["enroll_sec"] for e in examples], dtype=torch.float32),
        "example_s": float(sum(p.get("example_s", 0.0) for p in performance)),
        "decode_s": float(sum(p.get("decode_s", 0.0) for p in performance)),
        "augment_s": float(sum(p.get("wave_augment_s", 0.0) for p in performance)),
    }
