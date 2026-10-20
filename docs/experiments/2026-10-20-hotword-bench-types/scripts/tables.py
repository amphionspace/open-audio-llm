"""Print the README tables from two re-scored summaries (rescore.py output).

python tables.py <34479.json> <qwen.json>
"""

import json
import sys

ZH = [
    "aishell1-ne",
    "contextasr-speech-zh",
    "real-zh",
    "cv-zh",
    "aishell2",
    "aishell3",
    "kespeech",
]
TYPES = [
    "person",
    "place",
    "organization",
    "product",
    "work",
    "term",
    "other",
    "common",
]
EN = [
    ("gigaspeech", "l50"),
    ("librispeech-clean", "n100"),
    ("librispeech-other", "n100"),
    ("cv-en", "k50"),
]
MIN_CELL = 150  # words (recall) or utterances (false alarms) per table cell


def pct(value):
    return "–" if value is None else f"{100 * value:.1f}"


def by_type(summary, label, condition):
    key = f"{label}/{condition}"
    return summary["breakdowns"].get(key, {}).get("hotword_type", {})


def pooled(summary, labels, condition, kind):
    """Words, pooled recall and utterance-weighted false-alarm rate over ``labels``."""
    hits = words = utterances = alarmed = 0
    for label in labels:
        cell = by_type(summary, label, condition).get(kind)
        if not cell:
            continue
        words += cell["gold_total"]
        hits += (cell["recall"] or 0) * cell["gold_total"]
        if cell["false_alarm_rate"] is not None:
            utterances += cell["utterances"]
            alarmed += cell["false_alarm_rate"] * cell["utterances"]
    return (
        words,
        hits / words if words else None,
        alarmed / utterances if utterances else None,
    )


def pooled_table(models, labels):
    for kind in TYPES:
        cells = []
        for summary in models.values():
            words, before, _ = pooled(summary, labels, "none", kind)
            _, after, _ = pooled(summary, labels, "pool50", kind)
            _, _, alarm = pooled(summary, labels, "pool50_no_gold", kind)
            cells.append((words, before, after, alarm))
        (n, a0, a1, fa_a), (_, b0, b1, fa_b) = cells
        print(
            f"| {kind} | {n} | {pct(a0)} → {pct(a1)} | {pct(b0)} → {pct(b1)} "
            f"| {pct(fa_a)} | {pct(fa_b)} |"
        )


def split_table(models, condition, field, size_field, show_size=False):
    a_model, b_model = models.values()
    for label in ZH:
        row = []
        for kind in TYPES:
            a = by_type(a_model, label, condition).get(kind)
            b = by_type(b_model, label, condition).get(kind)
            if not a or a[size_field] < MIN_CELL:
                row.append("")
                continue
            size = f" ({a[size_field]})" if show_size else ""
            row.append(f"{pct(a[field])} / {pct(b[field])}{size}")
        print(f"| {label} | " + " | ".join(row) + " |")


def domain_table(models):
    a_model, b_model = models.values()

    def groups(summary, condition):
        return summary["breakdowns"][f"contextasr-speech-zh/{condition}"][
            "domain_group"
        ]

    listed = groups(a_model, "pool50")
    for group, a1 in sorted(listed.items(), key=lambda item: -item[1]["utterances"]):
        a0, an = (
            groups(a_model, "none")[group],
            groups(a_model, "pool50_no_gold")[group],
        )
        b0 = groups(b_model, "none")[group]
        b1 = groups(b_model, "pool50")[group]
        bn = groups(b_model, "pool50_no_gold")[group]
        print(
            f"| {group} | {a1['utterances']} | {pct(a0['recall'])} → {pct(a1['recall'])} "
            f"| {pct(b0['recall'])} → {pct(b1['recall'])} "
            f"| {pct(an['false_alarm_rate'])} | {pct(bn['false_alarm_rate'])} "
            f"| {pct(a1['b_error_rate'])} | {pct(b1['b_error_rate'])} |"
        )


def english_table(models):
    a_model, b_model = models.values()
    for label, condition in EN:
        for kind in TYPES:
            a1 = by_type(a_model, label, condition).get(kind)
            if not a1 or a1["gold_total"] < MIN_CELL:
                continue
            a0 = by_type(a_model, label, "none")[kind]
            b0 = by_type(b_model, label, "none")[kind]
            b1 = by_type(b_model, label, condition)[kind]
            print(
                f"| {label} {condition} | {kind} | {a1['gold_total']} "
                f"| {pct(a0['recall'])} → {pct(a1['recall'])} "
                f"| {pct(b0['recall'])} → {pct(b1['recall'])} |"
            )


def main(path_34479, path_qwen):
    models = {}
    for name, path in (("34479", path_34479), ("Qwen", path_qwen)):
        with open(path) as stream:
            models[name] = json.load(stream)
    print(
        "## Chinese, pooled per type: words | recall none → pool50 | pool50_no_gold FA"
    )
    pooled_table(models, ZH)
    print("## Chinese without ContextASR")
    pooled_table(models, [label for label in ZH if label != "contextasr-speech-zh"])
    print(f"## pool50 recall per split (34479 / Qwen), cells with >= {MIN_CELL} words")
    split_table(models, "pool50", "recall", "gold_total", show_size=True)
    print(f"## none recall per split (34479 / Qwen), cells with >= {MIN_CELL} words")
    split_table(models, "none", "recall", "gold_total")
    print(
        f"## pool50_no_gold FA per split (34479 / Qwen), cells with >= {MIN_CELL} utterances"
    )
    split_table(models, "pool50_no_gold", "false_alarm_rate", "utterances")
    print("## ContextASR domain groups: utterances | recall | FA | B-CER")
    domain_table(models)
    print("## English per type: words | recall none → list (34479 / Qwen)")
    english_table(models)


if __name__ == "__main__":
    main(*sys.argv[1:3])
