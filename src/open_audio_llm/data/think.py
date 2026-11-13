"""Think text for Qwen3-ASR SFT: how the speaker says a character is written (context_correction_clean_zh).

A think file (``think.jsonl.gz`` of ``context_correction_clean_zh@think-v1-20261112``) lists training records of
the cleaned sources as ``{dataset, record_id, think_on, think}``. ``think_on: false`` marks records that may only
train with think off: their explanation is uncertain, or the scan flagged them and they may still explain a
spelling. A record of such a source that the file does not list is ordinary; when it trains with think on, its
think is ``NO_EXPLANATION``. Sources without a think file always train with think off.

Think off keeps the existing target, ``language X<asr_text>text``. Think on prefixes it with
``<think>\\n...\\n</think>\\n\\n``; the template does not supervise the opening ``<think>\\n``, so the model opens a
think block only when the prompt does.
"""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path

from audio_data_contract import AudioRecord

THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n\n"
NO_EXPLANATION = "这段没有说明字怎么写。"
# The cleaning covered plain and target-speaker ASR records with a Chinese target.
THINK_TASKS = frozenset({"asr", "ts_asr"})
CHINESE = frozenset({"zh", "zh-cn", "chinese", "mandarin"})


@dataclass(frozen=True)
class ThinkConfig:
    files: tuple[str, ...]
    # Probability that a record with an explanation trains with think on.
    explained_prob: float
    # Probability that an ordinary record of a source with a think file trains with think on.
    ordinary_prob: float

    def __post_init__(self):
        if not self.files:
            raise ValueError("think.files must list at least one think file")
        for name in ("explained_prob", "ordinary_prob"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"think.{name} must be a probability")


class ThinkTable:
    """Think text per (dataset key, record ID), drawn per example."""

    def __init__(self, config: ThinkConfig, source_keys: set[str]):
        self.config = config
        self.rows: dict[tuple[str, str], str | None] = {}
        for path in config.files:
            with gzip.open(Path(path), "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    key = (row["dataset"], row["record_id"])
                    if key in self.rows:
                        raise ValueError(f"{path}: {key} listed twice")
                    if row["think_on"] and not row["think"]:
                        raise ValueError(f"{path}: {key} is think-on without think text")
                    self.rows[key] = row["think"] if row["think_on"] else None
        self.datasets = {dataset for dataset, _ in self.rows}
        unused = self.datasets - source_keys
        if unused:
            # A think file for a version that is not trained on points at the wrong version.
            raise ValueError(f"think files name datasets that no train source reads: {sorted(unused)}")

    def choose(self, source_key: str, record: AudioRecord, seed: str) -> str | None:
        """Think text for this example, or None for think off."""
        if (source_key not in self.datasets or record.task not in THINK_TASKS
                or record.language.lower() not in CHINESE or not record.target.strip()):
            return None
        key = (source_key, record.id)
        draw = random.Random(f"{seed}:think").random()
        if key in self.rows:
            think = self.rows[key]
            return think if think is not None and draw < self.config.explained_prob else None
        return NO_EXPLANATION if draw < self.config.ordinary_prob else None


def think_answer(think: str, answer: str) -> str:
    return f"{THINK_OPEN}{think}{THINK_CLOSE}{answer}"
