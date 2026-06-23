"""ASR metrics."""

from __future__ import annotations

from open_audio_llm.integrations.ms_swift.rewards.base import char_error_rate


def cer(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    total_edits = 0.0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        total_edits += char_error_rate(pred, ref) * max(len(ref), 1)
        total_chars += max(len(ref), 1)
    return total_edits / max(total_chars, 1)


def exact_match(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(references)
