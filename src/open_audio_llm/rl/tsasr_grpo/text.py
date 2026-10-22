"""Output parsing and error counting shared by the GRPO reward and its evaluation.

Normalization is the AmphionEval TS-ASR scoring (``amphion_eval.legacy.
open_audio_llm.qwen3_asr.normalize``): NFKC, lower case, punctuation and
symbols removed; Chinese is scored per character, other languages per word.
Keeping the reward on the same units as the benchmark avoids rewarding
punctuation or casing the benchmark ignores.
"""

from __future__ import annotations

import unicodedata

from rapidfuzz.distance import Levenshtein

_CHINESE = {"zh", "zh-cn", "chinese", "mandarin"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join("".join(
        " " if unicodedata.category(c)[0] in {"P", "S"} else c for c in text
    ).split())


def units(text: str, language: str) -> list[str]:
    text = normalize(text)
    return list(text.replace(" ", "")) if language.lower() in _CHINESE else text.split()


def errors(reference: str, hypothesis: str, language: str) -> tuple[int, int]:
    """(edit distance, reference length) in scoring units.

    For an absent target (empty reference) the distance is the hypothesis
    length, so any non-zero value is a false alarm.
    """
    ref, hyp = units(reference, language), units(hypothesis, language)
    return Levenshtein.distance(ref, hyp), len(ref)


def parse_output(raw: str) -> str:
    """Transcript of a raw Qwen3-ASR decode such as ``language Chinese<asr_text>...``.

    Repetitions are deliberately not collapsed: a degenerate loop should cost
    reward rather than be repaired before scoring.
    """
    for token in ("<|im_end|>", "<|endoftext|>"):
        raw = raw.replace(token, "")
    if "<asr_text>" in raw:
        return raw.split("<asr_text>", 1)[1].strip()
    return raw.strip() if not raw.strip().startswith("language ") else ""
