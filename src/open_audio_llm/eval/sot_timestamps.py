"""Utterance-boundary scoring, separate from timestamp-free transcription errors."""

import re
from collections import defaultdict

from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from open_audio_llm.data.sot import TIMESTAMP_FORMAT

from .qwen3_asr import normalize
from .sot import TAG, summarize_sot, transcription_units

LINE = re.compile(r"\[S([1-9]\d*)\]\[(\d+\.\d{2})-(\d+\.\d{2})\] (.+)")
TIME_FIELD = re.compile(r"(\[S\d+\])\[[^\]\r\n]*\]")
COLLAR_SECONDS = 0.5


def untimed_groups(text):
    text = TIME_FIELD.sub(r"\1", text)
    matches = list(TAG.finditer(text))
    preamble = text[:matches[0].start()] if matches else text
    groups = defaultdict(list)
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        groups[int(match.group(1))].append(text[match.end():end].strip())
    return preamble, {label: " ".join(parts) for label, parts in sorted(groups.items())}


def grouped_transcript(text):
    preamble, groups = untimed_groups(text)
    return preamble + "\n".join(f"[S{label}] {body}" for label, body in groups.items())


def parse_intervals(text, duration):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    segments, valid, last_start, ends = [], bool(lines), -1.0, {}
    for line in lines:
        match = LINE.fullmatch(line)
        if not match:
            valid = False
            continue
        label, start, end, body = match.groups()
        label, start, end = int(label), float(start), float(end)
        # Decimal rounding can move the final boundary by at most 5 ms.
        if not 0 <= start < end <= duration + .005001 or not normalize(body):
            valid = False
            continue
        if start < last_start or start < ends.get(label, 0):
            valid = False
        last_start, ends[label] = start, end
        segments.append((label, start, end, body))
    labels = {item[0] for item in segments}
    valid = valid and labels == set(range(1, len(labels) + 1))
    valid = valid and list(dict.fromkeys(item[0] for item in segments)) == list(range(1, len(labels) + 1))
    return segments, valid, len(lines)


def timing_counts(reference, prediction, duration, chinese, mixed):
    refs, valid, _ = parse_intervals(reference, duration)
    if not valid:
        raise ValueError("Invalid timestamp reference")
    hyps, format_valid, attempted = parse_intervals(prediction, duration)
    _, ref_text = untimed_groups(reference)
    _, hyp_text = untimed_groups(prediction)
    ref_labels, hyp_labels = list(ref_text), list(hyp_text)
    size = max(len(ref_labels), len(hyp_labels))

    def units(text):
        return transcription_units(text, chinese, mixed)

    cost = [[Levenshtein.distance(units(ref_text[ref_labels[i]]) if i < len(ref_labels) else [],
                                  units(hyp_text[hyp_labels[j]]) if j < len(hyp_labels) else [])
             for j in range(size)] for i in range(size)]
    rows, columns = linear_sum_assignment(cost)
    matched = within = 0
    absolute_error = 0.0
    for i, j in zip(rows, columns):
        if i >= len(ref_labels) or j >= len(hyp_labels):
            continue
        left = sorted((s for s in refs if s[0] == ref_labels[i]), key=lambda s: s[1])
        right = sorted((s for s in hyps if s[0] == hyp_labels[j]), key=lambda s: s[1])
        if not right:
            continue
        # Pair turns by transcript edit distance after speaker assignment.
        # Time never affects matching, so boundary error cannot choose its own pairing.
        turn_cost = [[Levenshtein.distance(units(a[3]), units(b[3])) for b in right] for a in left]
        turn_rows, turn_columns = linear_sum_assignment(turn_cost)
        for a, b in zip(turn_rows, turn_columns):
            start_error, end_error = abs(left[a][1] - right[b][1]), abs(left[a][2] - right[b][2])
            matched += 1
            absolute_error += start_error + end_error
            within += max(start_error, end_error) <= COLLAR_SECONDS + 1e-8
    return {"reference_segments": len(refs), "predicted_segments": attempted,
            "matched_segments": matched, "within_collar_segments": within,
            "boundary_absolute_error_seconds": absolute_error, "format_valid": format_valid}


def summarize_timing(counts):
    keys = ("reference_segments", "predicted_segments", "matched_segments",
            "within_collar_segments", "boundary_absolute_error_seconds")
    result = {key: sum(row[key] for row in counts) for key in keys}
    refs, hyps, matched, correct = (result[key] for key in keys[:4])
    result.update(method="text_assignment_utterance_boundaries_v1", collar_seconds=COLLAR_SECONDS,
                  boundary_mae_seconds=result[keys[-1]] / (2 * matched) if matched else None,
                  matched_reference_fraction=matched / refs if refs else None,
                  precision=correct / hyps if hyps else 0.0,
                  recall=correct / refs if refs else 0.0,
                  f1=2 * correct / (refs + hyps) if refs + hyps else 0.0)
    return result


def summarize_timed_sot(items, chinese):
    if not all(row.get("sot_output_format") == TIMESTAMP_FORMAT for row in items):
        raise ValueError("Cannot combine grouped and timestamped SOT protocols")
    plain = [{**{k: v for k, v in row.items() if k != "sot_output_format"},
              "reference": grouped_transcript(row["reference"]),
              "prediction": grouped_transcript(row["prediction"])} for row in items]
    result = summarize_sot(plain, chinese)
    counts = [timing_counts(row["reference"], row["prediction"], row["duration"],
                           chinese, row["language"] == "zh-en") for row in items]
    result["format_valid_rate"] = sum(row["format_valid"] for row in counts) / len(counts)
    result["timestamps"] = summarize_timing(counts)
    return result
