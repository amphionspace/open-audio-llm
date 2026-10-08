"""DER and time-constrained cp error rates for timestamped SOT.

tcpWER and DER are computed by MeetEval. Token text is split with the same
units as cp (Chinese characters, English words). MeetEval then places those
tokens inside each utterance with its character-based timing and applies the
collar itself. DER uses md-eval with a 0.25 s collar and overlap included.
Enrolled labels that start with T stay fixed. Anonymous S labels may permute.
"""

from __future__ import annotations

import re
import subprocess

from meeteval.der.md_eval import md_eval_22
from meeteval.io.rttm import RTTM
from meeteval.io.seglst import SegLST
from meeteval.io.uem import UEM
from meeteval.wer.wer.time_constrained import (
    tcp_word_error_rate,
    time_constrained_siso_word_error_rate,
)

from .sot import transcription_units

LINE = re.compile(r"\[([ST][1-9]\d*)\]\[(\d+\.\d{2})-(\d+\.\d{2})\] (.+)")

TCP_COLLARS = (5.0, 1.0)
DER_COLLAR = 0.25
TIME_SOURCE = "meeteval_character_based"


def parsed_segments(text):
    body = text.split("<asr_text>", 1)[-1]
    segments = []
    for line in body.splitlines():
        match = LINE.fullmatch(line.strip())
        if not match:
            continue
        label, start, end, words = match.groups()
        start, end = float(start), float(end)
        if end <= start:
            continue
        segments.append({"speaker": label, "start": start, "end": end, "text": words})
    return segments


def _words(text, language):
    tokens = transcription_units(text, language in {"zh", "Chinese"}, language == "zh-en")
    return " ".join(tokens)


def _seglst(segments, language):
    rows = []
    for segment in segments:
        words = _words(segment["text"], language)
        if not words:
            continue
        rows.append({
            "speaker": segment["speaker"],
            "start_time": segment["start"],
            "end_time": segment["end"],
            "words": words,
        })
    return SegLST(rows)


def _tcp(reference_segments, hypothesis_segments, fixed, collar, language):
    if not fixed:
        result = tcp_word_error_rate(
            _seglst(reference_segments, language),
            _seglst(hypothesis_segments, language),
            collar=collar,
        )
        return int(result.errors), int(result.length)

    errors = length = 0
    for label in sorted(fixed):
        reference = [segment for segment in reference_segments if segment["speaker"] == label]
        hypothesis = [segment for segment in hypothesis_segments if segment["speaker"] == label]
        if not reference and not hypothesis:
            continue
        result = time_constrained_siso_word_error_rate(
            _seglst(reference, language), _seglst(hypothesis, language), collar=collar,
        )
        errors += int(result.errors)
        length += int(result.length)
    reference = [segment for segment in reference_segments if segment["speaker"] not in fixed]
    hypothesis = [segment for segment in hypothesis_segments if segment["speaker"] not in fixed]
    if reference or hypothesis:
        result = tcp_word_error_rate(
            _seglst(reference, language), _seglst(hypothesis, language), collar=collar,
        )
        errors += int(result.errors)
        length += int(result.length)
    return errors, length


def _rttm(segments):
    lines = []
    for segment in segments:
        start = max(0.0, float(segment["start"]))
        end = float(segment["end"])
        if end <= start:
            continue
        lines.append(
            f"SPEAKER rec 1 {start:.3f} {end - start:.3f} <NA> <NA> {segment['speaker']} <NA> <NA>"
        )
    if not lines:
        lines.append("SPEAKER rec 1 0.000 0.000 <NA> <NA> __none__ <NA> <NA>")
    return RTTM.parse("\n".join(lines) + "\n")


def _empty_der():
    return {"missed": 0.0, "false_alarm": 0.0, "confusion": 0.0, "total": 0.0}


def _add_der(total, part):
    for key in total:
        total[key] += part[key]


def _der_call(reference_segments, hypothesis_segments, duration):
    if not reference_segments and not hypothesis_segments:
        return _empty_der()
    if not reference_segments:
        false_alarm = 0.0
        for segment in hypothesis_segments:
            start, end = max(0.0, segment["start"]), min(float(duration), segment["end"])
            if end > start:
                false_alarm += end - start
        return {"missed": 0.0, "false_alarm": false_alarm, "confusion": 0.0, "total": 0.0}
    try:
        result = md_eval_22(
            _rttm(reference_segments),
            _rttm(hypothesis_segments),
            collar=DER_COLLAR,
            regions="all",
            uem=UEM.parse(f"rec 1 0.000 {float(duration):.3f}\n"),
        )
    except subprocess.CalledProcessError as exc:
        if exc.returncode != 255:
            raise
        return _empty_der()
    return {
        "missed": float(result.missed_speaker_time),
        "false_alarm": float(result.falarm_speaker_time),
        "confusion": float(result.speaker_error_time),
        "total": float(result.scored_speaker_time),
    }


def _der(reference_segments, hypothesis_segments, duration, fixed):
    if not fixed:
        return _der_call(reference_segments, hypothesis_segments, duration)
    combined = _empty_der()
    for label in sorted(fixed):
        _add_der(combined, _der_call(
            [segment for segment in reference_segments if segment["speaker"] == label],
            [segment for segment in hypothesis_segments if segment["speaker"] == label],
            duration,
        ))
    _add_der(combined, _der_call(
        [segment for segment in reference_segments if segment["speaker"] not in fixed],
        [segment for segment in hypothesis_segments if segment["speaker"] not in fixed],
        duration,
    ))
    return combined


def score_recording(reference, prediction, duration, *, language, fixed_targets, target_count=0):
    reference_segments = parsed_segments(reference)
    hypothesis_segments = parsed_segments(prediction)
    fixed = set()
    if fixed_targets:
        fixed = {f"T{index}" for index in range(1, target_count + 1)}
        fixed.update(segment["speaker"] for segment in (*reference_segments, *hypothesis_segments)
                     if segment["speaker"].startswith("T"))
    tcp = {}
    for collar in TCP_COLLARS:
        errors, units = _tcp(
            reference_segments, hypothesis_segments, fixed, collar, language,
        )
        tcp[f"{collar:g}"] = {
            "collar_seconds": collar, "errors": errors, "reference_units": units,
            "error_rate": errors / units if units else None,
        }
    der = _der(reference_segments, hypothesis_segments, duration, fixed)
    der["error_rate"] = (
        (der["missed"] + der["false_alarm"] + der["confusion"]) / der["total"] if der["total"] else None
    )
    der["collar_seconds"] = DER_COLLAR
    der["overlap"] = "included"
    der["mapping"] = "fixed_targets" if fixed_targets else "anonymous"
    return {"tcp": tcp, "der": der}


def summarize_meeting_scores(rows, *, fixed_targets):
    scores = [score_recording(
        row["reference"], row["prediction"], row["duration"], language=row["language"],
        fixed_targets=fixed_targets, target_count=row.get("count", 0),
    ) for row in rows]
    tcp = {}
    for collar in TCP_COLLARS:
        key = f"{collar:g}"
        errors = sum(row["tcp"][key]["errors"] for row in scores)
        units = sum(row["tcp"][key]["reference_units"] for row in scores)
        tcp[key] = {
            "collar_seconds": collar, "errors": errors, "reference_units": units,
            "error_rate": errors / units if units else None,
            "time_source": TIME_SOURCE,
        }
    missed = sum(row["der"]["missed"] for row in scores)
    false_alarm = sum(row["der"]["false_alarm"] for row in scores)
    confusion = sum(row["der"]["confusion"] for row in scores)
    total = sum(row["der"]["total"] for row in scores)
    language = rows[0]["language"]
    cp_metric = "cpMER" if language == "zh-en" else "cpCER" if language in {"zh", "Chinese"} else "cpWER"
    tcp_metric = cp_metric.replace("cp", "tcp")
    return {
        "cp_metric": cp_metric,
        "tcp_metric": tcp_metric,
        "tcp_error_rate_5s": tcp["5"]["error_rate"],
        "tcp_error_rate_1s": tcp["1"]["error_rate"],
        "tcp": tcp,
        "der": (missed + false_alarm + confusion) / total if total else None,
        "der_details": {
            "collar_seconds": DER_COLLAR, "overlap": "included",
            "mapping": "fixed_targets" if fixed_targets else "anonymous",
            "missed": missed, "false_alarm": false_alarm, "confusion": confusion, "total": total,
            "error_rate": (missed + false_alarm + confusion) / total if total else None,
        },
    }
