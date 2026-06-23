"""Hotword metrics."""

from __future__ import annotations


def hotword_recall(predicted_texts: list[str], reference_hotwords: list[list[str]]) -> float:
    total = 0
    hit = 0
    for text, hotwords in zip(predicted_texts, reference_hotwords):
        for hotword in hotwords:
            total += 1
            hit += int(hotword in text)
    return hit / total if total else 1.0


def hotword_false_alarm(predicted_texts: list[str], negative_hotwords: list[list[str]]) -> float:
    total = 0
    false_alarm = 0
    for text, hotwords in zip(predicted_texts, negative_hotwords):
        for hotword in hotwords:
            total += 1
            false_alarm += int(hotword in text)
    return false_alarm / total if total else 0.0
