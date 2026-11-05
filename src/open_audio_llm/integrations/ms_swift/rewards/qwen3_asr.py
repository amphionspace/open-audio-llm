"""GRPO rewards for native Qwen3-ASR outputs ``language X<asr_text>text``.

Scores are computed on mixed error units: one unit per CJK/kana/Hangul character and one per
Latin/digit word, after NFKC, lower-casing and dropping punctuation. Repetition loops are scored
as produced (no ``detect_and_fix_repetitions``), so the policy pays for them.
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz.distance import Levenshtein

ASR_TEXT_TAG = "<asr_text>"
_FORMAT = re.compile(r"^language [A-Za-z]+<asr_text>")
_UNITS = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]|[a-z0-9]+(?:'[a-z0-9]+)*")


def transcript(output: str | None) -> str:
    """Text after the ``<asr_text>`` tag; ``language None`` and untagged output count as written."""
    text = str(output or "")
    return text.split(ASR_TEXT_TAG, 1)[1] if ASR_TEXT_TAG in text else text


def units(text: str) -> list[str]:
    return _UNITS.findall(unicodedata.normalize("NFKC", text).lower())


def well_formed(output: str | None) -> bool:
    return bool(_FORMAT.match(str(output or "").strip()))


def accuracy(output: str, solution: str) -> float:
    """1 - mixed error rate, floored at 0; an empty reference scores 1 only for empty output."""
    hyp, ref = units(transcript(output)), units(transcript(solution))
    if not ref:
        return float(not hyp)
    return max(0.0, 1.0 - Levenshtein.distance(hyp, ref) / len(ref))


def presence(output: str, solution: str) -> float:
    """+1 for a correct empty output, -1 for output on an empty reference or nothing on speech."""
    hyp, ref = units(transcript(output)), units(transcript(solution))
    if not ref:
        return 1.0 if not hyp else -1.0
    return -1.0 if not hyp else 0.0


def _contains(haystack: list[str], needle: list[str]) -> bool:
    return bool(needle) and f" {' '.join(needle)} " in f" {' '.join(haystack)} "


def prompt_hotwords(messages) -> str | None:
    """The list the model was shown: the system prompt's ``Hotwords:`` line, None without one.

    Plain ASR sends the line as the whole system prompt; TS-ASR (``ts_prompt.ts_system``) puts it on
    its own line after the TS instruction, so every line is checked.
    """
    for message in messages or []:
        if message.get("role") != "system":
            continue
        for line in str(message.get("content") or "").splitlines():
            if line.startswith("Hotwords:"):
                return line.split(":", 1)[1].strip()
    return None


def hotword(output: str, solution: str, candidates: str | None) -> float | None:
    """Recall of listed words the reference contains, minus listed words written but not spoken.

    Listed words absent from the reference are the distractors (near-homophones in the hard sets);
    the penalty is their count relative to the spoken listed words, capped at 1. Without a list the
    reward is None (not scored), so prompts without hotwords neither move nor dilute it.
    """
    words = [w.strip() for w in str(candidates or "").split(",") if w.strip()]
    if not words or words == ["N/A"]:
        return None
    hyp, ref = units(transcript(output)), units(transcript(solution))
    spoken = [w for w in words if _contains(ref, units(w))]
    hits = sum(_contains(hyp, units(w)) for w in spoken)
    false_alarms = sum(_contains(hyp, units(w)) for w in words if w not in spoken)
    recall = hits / len(spoken) if spoken else 1.0
    return recall - min(1.0, false_alarms / max(1, len(spoken)))
