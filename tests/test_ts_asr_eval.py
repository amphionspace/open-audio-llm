import copy

import pytest

from open_audio_llm.eval.ts_asr import retention_gate, summarize


def row(source, reference, prediction, task="asr"):
    return {"source": source, "reference": reference, "prediction": prediction,
            "language": "zh", "task": task}


def test_ts_metrics_separate_domains_and_negative_false_alarms():
    metrics = summarize([
        row("clean", "你好", "你好。"), row("meeting", "你好", "你"),
        row("ts", "你好", "", "ts_asr"),
        row("ts-neg", "", "你好", "ts_asr"), row("ts-neg", "", "", "ts_asr"),
    ])
    assert metrics["clean"]["error_rate"] == 0
    assert metrics["meeting"]["error_rate"] == 0.5
    assert metrics["ts"]["miss_rate"] == 1
    assert metrics["ts-neg"]["error_rate"] is None
    assert metrics["ts-neg"]["false_alarm_rate"] == 0.5


@pytest.mark.parametrize("language", ["Mandarin", "Southwestern"])
def test_kespeech_is_scored_as_chinese_characters_and_gated(language):
    baseline_row = row("kespeech", "你好世界", "你好世界")
    baseline_row["language"] = language
    candidate_row = dict(baseline_row, prediction="你好世")
    baseline = {"protocol": {}, "metrics": summarize([baseline_row])}
    candidate = {"protocol": {}, "metrics": summarize([candidate_row])}
    assert candidate["metrics"]["kespeech"]["metric"] == "CER"
    assert candidate["metrics"]["kespeech"]["error_rate"] == 0.25
    assert not retention_gate(baseline, candidate)["passed"]


def test_retention_gate_cannot_hide_a_domain_regression_in_average():
    baseline = {"protocol": {"records_sha256": "same"}, "metrics": summarize([
        row("clean", "你好", "你"), row("meeting", "你好", "你好"),
    ])}
    candidate = {"protocol": baseline["protocol"], "metrics": summarize([
        row("clean", "你好", "你好"), row("meeting", "你好", "你"),
    ])}
    gate = retention_gate(baseline, candidate)
    assert not gate["passed"]
    assert gate["checks"]["clean"]["passed"]
    assert not gate["checks"]["meeting"]["passed"]
    assert retention_gate(baseline, candidate, max_cer_increase=0.5)["passed"]
    changed = copy.deepcopy(candidate)
    changed["protocol"]["records_sha256"] = "different"
    with pytest.raises(ValueError, match="protocol"):
        retention_gate(baseline, changed)
    changed = copy.deepcopy(candidate)
    changed["metrics"].pop("meeting")
    with pytest.raises(ValueError, match="sources differ"):
        retention_gate(baseline, changed)


def test_retention_gate_requires_chinese_asr_and_valid_tolerance():
    summary = {"protocol": {}, "metrics": summarize([row("ts", "你好", "你好", "ts_asr")])}
    with pytest.raises(ValueError, match="ordinary-ASR"):
        retention_gate(summary, summary)
    with pytest.raises(ValueError, match="non-negative"):
        retention_gate(summary, summary, -0.1)
