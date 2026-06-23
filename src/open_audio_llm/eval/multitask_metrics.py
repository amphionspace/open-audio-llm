"""Metrics for non-ASR tasks."""

from __future__ import annotations

from collections import Counter

from .asr_metrics import cer, exact_match


def classification_accuracy(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(references)


def macro_f1(predictions: list[str], references: list[str]) -> float:
    labels = sorted(set(predictions) | set(references))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(p == label and r == label for p, r in zip(predictions, references))
        fp = sum(p == label and r != label for p, r in zip(predictions, references))
        fn = sum(p != label and r == label for p, r in zip(predictions, references))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def summarize_by_task(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("task", "asr"), []).append(row)
    summary = {}
    for task, task_rows in grouped.items():
        preds = [r.get("prediction", "") for r in task_rows]
        refs = [r.get("reference", "") for r in task_rows]
        if task in {"ser", "sec", "sei"}:
            summary[task] = {
                "accuracy": classification_accuracy(preds, refs),
                "macro_f1": macro_f1(preds, refs),
                "labels": dict(Counter(refs)),
            }
        else:
            summary[task] = {
                "cer": cer(preds, refs),
                "exact_match": exact_match(preds, refs),
            }
    return summary
