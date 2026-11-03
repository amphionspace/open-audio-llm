"""Elapsed-time markers inside the audio placeholder span.

MOSS-Transcribe-Diarize (arXiv 2601.01554; processor ``time_marker_every_seconds: 5``) writes the elapsed seconds as
plain digit tokens between the audio tokens, every few seconds, so the decoder reads absolute time from text instead
of inferring it from positions. Here the same layout is built from the audio token count alone: Qwen3-ASR's encoder
emits ``TOKENS_PER_SECOND`` tokens per full second (``_get_feat_extract_output_lengths``: 13 per 100 Mel frames), so
the training template and the vLLM processor, which both know only that count, place every marker identically.

A marker follows each complete ``every`` seconds of audio (as in MOSS: "5" after the first 5 s, "10" after the next,
also when the clip ends exactly on the boundary); the audio tokens themselves are unchanged, so the encoder output
still fills exactly the placeholder positions.

Enabled with ``OPEN_AUDIO_LLM_AUDIO_TIME_MARKER_SECONDS`` (seconds between markers; unset or 0 = off), set in the
training and serving configs alike.
"""

from __future__ import annotations

import os

ENV = "OPEN_AUDIO_LLM_AUDIO_TIME_MARKER_SECONDS"
TOKENS_PER_SECOND = 13


def time_marker_seconds() -> int:
    """Seconds between markers from the environment; 0 when markers are off."""
    value = os.environ.get(ENV, "").strip()
    if not value:
        return 0
    seconds = int(value)
    if seconds < 0:
        raise ValueError(f"{ENV} must be >= 0, got {value!r}")
    return seconds


def marker_layout(n_tokens: int, every: int) -> list[int | str]:
    """Runs of audio tokens (ints) and the elapsed seconds written after each run (strs)."""
    if every <= 0:
        return [n_tokens]
    run = TOKENS_PER_SECOND * every
    layout: list[int | str] = []
    used, seconds = 0, every
    while used + run <= n_tokens:
        layout += [run, str(seconds)]
        used += run
        seconds += every
    if n_tokens > used:
        layout.append(n_tokens - used)
    return layout


def marked_placeholders(audio_token, n_tokens: int, every: int, digit=str) -> list:
    """The placeholder span: ``audio_token`` per audio token with ``digit(d)`` for each digit of every marker."""
    out = []
    for part in marker_layout(n_tokens, every):
        if isinstance(part, int):
            out += [audio_token] * part
        else:
            out += [digit(d) for d in part]
    return out
