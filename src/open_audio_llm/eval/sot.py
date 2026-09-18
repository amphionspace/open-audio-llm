"""Permutation-invariant transcription scores without discarding untagged output."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from .qwen3_asr import normalize

TAG = re.compile(r"\[S(\d+)\]")
ATTRIBUTION_METHOD = "unique_reference_units_v1"
ATTRIBUTION_COUNTS = ("correct_units", "scored_units", "unique_reference_units", "excess_reference_units")


def parse_speakers(text):
    matches = list(TAG.finditer(text))
    speakers = {}
    preamble = text[:matches[0].start()] if matches else text
    if normalize(preamble):
        speakers["unattributed"] = preamble
    labels = []
    for index, match in enumerate(matches):
        label = int(match.group(1))
        labels.append(label)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        speakers[label] = (speakers.get(label, "") + " " + text[match.end():end]).strip()
    valid = (bool(matches) and not preamble.strip()
             and labels == list(range(1, len(matches) + 1))
             and all(normalize(value) for value in speakers.values()))
    return list(speakers.values()), valid


def attribution_summary(counts, reference_units):
    """Micro-average counts; zero evidence is unknown, not perfect accuracy."""
    totals = {key: sum(row[key] for row in counts) for key in ATTRIBUTION_COUNTS}
    scored = totals["scored_units"]
    return {"method": ATTRIBUTION_METHOD, **totals,
            "accuracy": totals["correct_units"] / scored if scored else None,
            "coverage": scored / reference_units if reference_units else None}


def _speaker_attribution(references, hypotheses, cost, unattributed):
    # Grouped transcripts have no timestamps: shared units have no unique owner.
    ref_counts = [Counter(text) for text in references]
    hyp_counts = [Counter(text) for text in hypotheses]
    owners = defaultdict(list)
    for index, counts in enumerate(ref_counts):
        for token in counts:
            owners[token].append(index)
    unique = {token: indexes[0] for token, indexes in owners.items() if len(indexes) == 1}
    hyp_total = Counter(token for text in hypotheses for token in text)
    # Excess repetitions cannot be assigned to specific reference occurrences.
    scorable = {token: owner for token, owner in unique.items()
                if hyp_total[token] <= ref_counts[owner][token]}
    scored = sum(hyp_total[token] for token in scorable)
    credit = [[sum(count for token, count in hyp_counts[j].items() if scorable.get(token) == i)
               if j < len(hypotheses) and not (unattributed and j == 0) else 0
               for j in range(len(cost))] for i in range(len(cost))]
    # Keep the cpWER optimum; among tied permutations maximize attributable credit.
    rows, columns = linear_sum_assignment([
        [(scored + 1) * value - credit[i][j] for j, value in enumerate(row)]
        for i, row in enumerate(cost)
    ])
    counts = {"correct_units": sum(credit[i][j] for i, j in zip(rows, columns)),
              "scored_units": scored,
              "unique_reference_units": sum(ref_counts[owner][token] for token, owner in unique.items()),
              "excess_reference_units": sum(ref_counts[owner][token] for token, owner in unique.items()
                                            if token not in scorable)}
    return attribution_summary([counts], sum(map(len, references)))


def score_speakers(reference, prediction, chinese, mixed=False):
    references, reference_valid = parse_speakers(reference)
    hypotheses, valid = parse_speakers(prediction)
    if not reference_valid:
        raise ValueError("SOT reference must contain ordered, nonempty speaker labels")

    def units(text):
        text = normalize(text)
        if mixed:
            # Chinese characters and whitespace-delimited English words count once.
            text = re.sub(r'([\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f])', r' \1 ', text)
            return text.split()
        return list(text.replace(" ", "")) if chinese else text.split()

    references = [units(text) for text in references]
    hypotheses = [units(text) for text in hypotheses]
    hypotheses = [text for text in hypotheses if text]
    size = max(len(references), len(hypotheses))
    cost = [[Levenshtein.distance(references[i] if i < len(references) else [],
                                  hypotheses[j] if j < len(hypotheses) else [])
             for j in range(size)] for i in range(size)]
    rows, columns = linear_sum_assignment(cost)
    first_tag = TAG.search(prediction)
    unattributed = bool(units(prediction[:first_tag.start()] if first_tag else prediction))
    return {"errors": sum(cost[i][j] for i, j in zip(rows, columns)),
            "reference_units": sum(map(len, references)), "reference_speakers": len(references),
            "predicted_speakers": len(hypotheses), "format_valid": valid,
            "empty_output": not hypotheses,
            "speaker_attribution": _speaker_attribution(references, hypotheses, cost, unattributed)}


def summarize_sot(items, chinese):
    mixed = items[0]['language'] == 'zh-en'
    scores = [score_speakers(row["reference"], row["prediction"], chinese, mixed=mixed) for row in items]
    errors = sum(row["errors"] for row in scores)
    units = sum(row["reference_units"] for row in scores)
    return {"task": "speaker_attributed_asr", "language": items[0]["language"],
            "metric": "cpMER" if mixed else "cpCER" if chinese else "cpWER", "utterances": len(items),
            "errors": errors, "reference_units": units, "error_rate": errors / units,
            "speaker_count_accuracy": sum(row["reference_speakers"] == row["predicted_speakers"] for row in scores) / len(scores),
            "format_valid_rate": sum(row["format_valid"] for row in scores) / len(scores),
            "empty_output_rate": sum(row["empty_output"] for row in scores) / len(scores),
            "speaker_attribution": attribution_summary([row["speaker_attribution"] for row in scores], units)}


def rescore_predictions(predictions, output):
    """Score saved SOT outputs on CPU, preserving the original inference artifacts."""
    from open_audio_llm.data.qwen3_asr import native_language

    payload = predictions.read_bytes()
    groups = defaultdict(list)
    for line in payload.decode("utf-8").splitlines():
        row = json.loads(line)
        if row["task"] == "speaker_attributed_asr":
            groups[row["source"]].append(row)
    if not groups:
        raise ValueError("No speaker_attributed_asr predictions to score")
    report = {"predictions_sha256": hashlib.sha256(payload).hexdigest(),
              "metrics": {source: summarize_sot(items, chinese=native_language(items[0]["language"]) == "Chinese")
                          for source, items in groups.items()}}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Rescore saved SOT predictions without model inference")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.predictions.resolve() == args.output.resolve():
        parser.error("output must differ from predictions")
    rescore_predictions(args.predictions, args.output)


if __name__ == "__main__":
    main()
