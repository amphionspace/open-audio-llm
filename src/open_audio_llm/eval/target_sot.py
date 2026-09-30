"""Fixed-target transcription scoring; only anonymous speakers may permute."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from open_audio_llm.data.target_sot import LINE, parse_target_segments

from .sot import transcription_units
from .sot_timestamps import COLLAR_SECONDS, summarize_timing


def transcript_body(text):
    return text.split("<asr_text>", 1)[-1].strip()


def _groups(text):
    groups = defaultdict(list)
    for line in transcript_body(text).splitlines():
        match = re.match(r"\[([ST][1-9]\d*)\](?:\[[^\]]*\])?\s*(.*)", line.strip())
        label, body = match.groups() if match else ("unattributed", line)
        groups[label].append(body)
    return {label: " ".join(parts) for label, parts in groups.items()}


def score_target_sot(reference, prediction, *, duration, count, mode, language):
    reference, prediction = transcript_body(reference), transcript_body(prediction)
    refs = parse_target_segments(reference, duration, count=count, mode=mode)
    try:
        hyps = parse_target_segments(prediction, duration, count=count, mode=mode)
        valid = True
    except ValueError:
        # Still score attributable text and valid individual boundaries. Invalid
        # output cannot gain a perfect score merely by disappearing in the parser.
        hyps, valid = [], False
        for line in prediction.splitlines():
            match = LINE.fullmatch(line.strip())
            if match:
                label, start, end, text = match.groups()
                if 0 <= float(start) < float(end) <= duration + .005001:
                    hyps.append({"speaker_id": label, "start": float(start), "end": float(end), "text": text})
    ref_text, hyp_text = _groups(reference), _groups(prediction)
    units = lambda text: transcription_units(text, language in {"zh", "Chinese"}, language == "zh-en")
    targets = [f"T{i + 1}" for i in range(count)]
    pairs = [(label, label) for label in targets]
    left = [label for label in ref_text if label.startswith("S")]
    right = [label for label in hyp_text if label.startswith("S")]
    size = max(len(left), len(right))
    left += [None] * (size - len(left))
    right += [None] * (size - len(right))
    if size:
        cost = [[Levenshtein.distance(units(ref_text.get(a, "")), units(hyp_text.get(b, "")))
                 for b in right] for a in left]
        rows, columns = linear_sum_assignment(cost)
        pairs.extend((left[i], right[j]) for i, j in zip(rows, columns))
    errors = target_errors = target_units = 0
    deletions = insertions = substitutions = 0
    matched = within = 0
    absolute_error = 0.0
    for a, b in pairs:
        r, h = units(ref_text.get(a, "")), units(hyp_text.get(b, ""))
        distance = Levenshtein.distance(r, h)
        errors += distance
        if a in targets:
            target_errors += distance
            target_units += len(r)
            for op in Levenshtein.editops(r, h):
                deletions += op.tag == "delete"
                insertions += op.tag == "insert"
                substitutions += op.tag == "replace"
        r_turns = [s for s in refs if s["speaker_id"] == a]
        h_turns = [s for s in hyps if s["speaker_id"] == b]
        if r_turns and h_turns:
            cost = [[Levenshtein.distance(units(x["text"]), units(y["text"]))
                     for y in h_turns] for x in r_turns]
            rows, columns = linear_sum_assignment(cost)
            for i, j in zip(rows, columns):
                differences = [abs(r_turns[i][key] - h_turns[j][key]) for key in ("start", "end")]
                matched += 1
                absolute_error += sum(differences)
                within += max(differences) <= COLLAR_SECONDS + 1e-8
    used = {b for _, b in pairs}
    unassigned = sum(len(units(text)) for label, text in hyp_text.items() if label not in used)
    errors += unassigned
    absent = [t for t in targets if not units(ref_text.get(t, ""))]
    return {
        "errors": errors, "reference_units": sum(len(units(s)) for s in ref_text.values()),
        "target_errors": target_errors, "target_reference_units": target_units,
        "target_deletions": deletions, "target_insertions": insertions, "target_substitutions": substitutions,
        "absent_targets": len(absent), "absent_target_false_activations": sum(bool(units(hyp_text.get(t, ""))) for t in absent),
        "non_target_output_units": (sum(len(units(v)) for k, v in hyp_text.items() if k not in targets)
                                 if mode == "targets_only" else 0),
        "unattributed_output_units": unassigned, "format_valid": valid,
        "reference_segments": len(refs), "predicted_segments": sum(bool(l.strip()) for l in prediction.splitlines()),
        "matched_segments": matched, "within_collar_segments": within,
        "boundary_absolute_error_seconds": absolute_error,
    }


def summarize_target_sot(rows):
    counts = [score_target_sot(r["reference"], r["prediction"], duration=r["duration"],
                              count=r["count"], mode=r["mode"], language=r["language"]) for r in rows]
    if not counts:
        raise ValueError("No conditional SOT predictions")
    sums = {key: sum(r[key] for r in counts) for key in counts[0]}
    sums.update(utterances=len(rows), metric="fixed_target_mixed_units",
                error_rate=sums["errors"] / sums["reference_units"] if sums["reference_units"] else None,
                target_error_rate=sums["target_errors"] / sums["target_reference_units"] if sums["target_reference_units"] else None,
                format_valid_rate=sums.pop("format_valid") / len(rows),
                timestamps=summarize_timing(counts))
    sums["timestamps"]["method"] = "fixed_targets_anonymous_text_assignment_v1"
    return sums


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    args.output.write_text(json.dumps(summarize_target_sot(rows), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
