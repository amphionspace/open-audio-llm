"""Permutation-invariant transcription scores without discarding untagged output."""

import re

from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

from .qwen3_asr import normalize

TAG = re.compile(r"\[S(\d+)\]")


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


def score_speakers(reference, prediction, chinese):
    references, reference_valid = parse_speakers(reference)
    hypotheses, valid = parse_speakers(prediction)
    if not reference_valid:
        raise ValueError("SOT reference must contain ordered, nonempty speaker labels")

    def units(text):
        text = normalize(text)
        return list(text.replace(" ", "")) if chinese else text.split()

    references = [units(text) for text in references]
    hypotheses = [units(text) for text in hypotheses]
    hypotheses = [text for text in hypotheses if text]
    size = max(len(references), len(hypotheses))
    cost = [[Levenshtein.distance(references[i] if i < len(references) else [],
                                  hypotheses[j] if j < len(hypotheses) else [])
             for j in range(size)] for i in range(size)]
    rows, columns = linear_sum_assignment(cost)
    return {"errors": sum(cost[i][j] for i, j in zip(rows, columns)),
            "reference_units": sum(map(len, references)), "reference_speakers": len(references),
            "predicted_speakers": len(hypotheses), "format_valid": valid,
            "empty_output": not hypotheses}


def summarize_sot(items, chinese):
    scores = [score_speakers(row["reference"], row["prediction"], chinese) for row in items]
    errors = sum(row["errors"] for row in scores)
    units = sum(row["reference_units"] for row in scores)
    return {"task": "speaker_attributed_asr", "language": items[0]["language"],
            "metric": "cpCER" if chinese else "cpWER", "utterances": len(items),
            "errors": errors, "reference_units": units, "error_rate": errors / units,
            "speaker_count_accuracy": sum(row["reference_speakers"] == row["predicted_speakers"] for row in scores) / len(scores),
            "format_valid_rate": sum(row["format_valid"] for row in scores) / len(scores),
            "empty_output_rate": sum(row["empty_output"] for row in scores) / len(scores)}
