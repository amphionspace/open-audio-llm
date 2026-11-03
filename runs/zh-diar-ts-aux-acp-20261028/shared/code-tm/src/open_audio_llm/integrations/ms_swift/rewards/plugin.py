"""ms-swift reward plugin."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from .base import (
        char_error_rate,
        hotword_f1,
        hotword_match_accuracy,
        parse_hotwords_from_instruction,
        parse_structured_answer,
    )
except ImportError:  # pragma: no cover - ms-swift file plugin import mode
    ms_swift_dir = str(Path(__file__).resolve().parents[1])
    if ms_swift_dir not in sys.path:
        sys.path.insert(0, ms_swift_dir)
    from rewards.base import (
        char_error_rate,
        hotword_f1,
        hotword_match_accuracy,
        parse_hotwords_from_instruction,
        parse_structured_answer,
    )

try:
    from swift.rewards import ORM, orms
except Exception:  # pragma: no cover - optional ms-swift dependency
    ORM = object
    orms = {}


class ASRFormatReward(ORM):
    """Reward whether output follows Language/Hotwords/Transcription format."""

    _FIELD_PATTERNS = [
        re.compile(r"Language:\s*\S+"),
        re.compile(r"Hotwords:\s*\S+"),
        re.compile(r"Transcription:\s*\S+"),
    ]

    def __call__(self, completions, **kwargs):
        rewards = []
        for text in completions:
            if not text or not str(text).strip():
                rewards.append(0.0)
                continue
            hits = sum(1 for pattern in self._FIELD_PATTERNS if pattern.search(text))
            rewards.append(round(hits / len(self._FIELD_PATTERNS), 4))
        return rewards


class ASRAccuracyReward(ORM):
    def __call__(self, completions, solution, **kwargs):
        rewards = []
        for completion, sol in zip(completions, solution):
            pred = parse_structured_answer(completion)["transcription"]
            ref = parse_structured_answer(sol)["transcription"]
            rewards.append(max(0.0, 1.0 - char_error_rate(str(pred), str(ref))))
        return rewards


class HotwordReward(ORM):
    def __call__(self, completions, solution, **kwargs):
        candidate_col = kwargs.get("candidate_hotwords")
        messages_col = kwargs.get("messages")
        rewards = []
        for idx, (completion, sol) in enumerate(zip(completions, solution)):
            pred = parse_structured_answer(completion)["hotwords"]
            ref = parse_structured_answer(sol)["hotwords"]
            candidates = self._resolve_candidates(idx, candidate_col, messages_col)
            if candidates is not None:
                rewards.append(
                    hotword_match_accuracy(list(pred), list(ref), candidates)
                )
            else:
                rewards.append(hotword_f1(list(pred), list(ref)))
        return rewards

    @staticmethod
    def _resolve_candidates(idx: int, candidate_col, messages_col) -> list[str] | None:
        if candidate_col is not None:
            raw = candidate_col[idx]
            if not raw or str(raw).strip().upper() == "N/A":
                return []
            return [item.strip() for item in str(raw).split(",") if item.strip()]

        if messages_col is not None:
            messages = messages_col[idx]
            if isinstance(messages, list):
                for message in messages:
                    if message.get("role") == "user":
                        result = parse_hotwords_from_instruction(
                            message.get("content", "")
                        )
                        if result is not None:
                            return result
        return None


if hasattr(orms, "__setitem__"):
    orms["asr_format_reward"] = ASRFormatReward
    orms["asr_accuracy_reward"] = ASRAccuracyReward
    orms["hotword_reward"] = HotwordReward
