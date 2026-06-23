"""Pure reward helpers shared by GRPO and evaluation code."""

from __future__ import annotations

import re


def parse_structured_answer(text: str | None) -> dict[str, object]:
    """Extract Language / Hotwords / Transcription from a structured answer."""

    result: dict[str, object] = {
        "language": None,
        "transcription": "",
        "hotwords": [],
    }
    if not text:
        return result

    lang_match = re.search(r"Language:\s*(.+?)(?:\n|$)", str(text))
    if lang_match:
        result["language"] = lang_match.group(1).strip()

    hotword_match = re.search(r"Hotwords:\s*(.+?)(?:\n|$)", str(text))
    if hotword_match:
        raw = hotword_match.group(1).strip()
        if raw and raw.upper() != "N/A":
            result["hotwords"] = [
                item.strip() for item in raw.split(",") if item.strip()
            ]

    transcription_match = re.search(r"Transcription:\s*(.+)", str(text))
    if transcription_match:
        result["transcription"] = transcription_match.group(1).strip()
    elif text:
        result["transcription"] = str(text).strip()
    return result


def parse_hotwords_from_instruction(instruction: str | None) -> list[str] | None:
    """Extract candidate hotwords from a user instruction when present."""

    if not instruction:
        return None
    hotword_match = re.search(r"Hotwords:\s*(.+?)(?:\n|$)", instruction)
    if not hotword_match:
        return None
    raw = hotword_match.group(1).strip()
    if not raw or raw.upper() == "N/A":
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def char_error_rate(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    prev = list(range(len(ref) + 1))
    for i, pc in enumerate(pred, start=1):
        cur = [i]
        for j, rc in enumerate(ref, start=1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (pc != rc)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def hotword_f1(pred: list[str], ref: list[str]) -> float:
    pred_set, ref_set = set(pred), set(ref)
    if not pred_set and not ref_set:
        return 1.0
    if not pred_set or not ref_set:
        return 0.0
    tp = len(pred_set & ref_set)
    precision = tp / len(pred_set)
    recall = tp / len(ref_set)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def hotword_match_accuracy(
    pred_hotwords: list[str],
    ref_hotwords: list[str],
    candidates: list[str],
) -> float:
    """Score agreement on each prompted hotword candidate."""

    if not candidates:
        return 1.0 if not pred_hotwords and not ref_hotwords else 0.0
    pred_set = set(pred_hotwords)
    ref_set = set(ref_hotwords)
    correct = sum(
        1 for word in candidates if (word in pred_set) == (word in ref_set)
    )
    return correct / len(candidates)
