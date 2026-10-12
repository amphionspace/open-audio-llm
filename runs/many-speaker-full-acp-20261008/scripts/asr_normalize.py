"""ASR regression text normalization used by score.py.

Verbatim copy of ``normalize`` from src/open_audio_llm/eval/qwen3_asr.py at 05187d7;
that module was removed when evaluation moved to AmphionEval (2375bd7). Kept here
so ASR regression scores stay comparable with the archived 2026-10-06 results.
"""
import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join("".join(
        " " if unicodedata.category(c)[0] in {"P", "S"} else c for c in text
    ).split())
