import pytest

pytest.importorskip("rapidfuzz")

from open_audio_llm.eval.qwen3_asr import contains, summarize


def test_hotword_matching_respects_english_words():
    assert contains("Alice's friend.", "Alice", False)
    assert not contains("cartography", "art", False)
    assert contains("欢迎到北京！", "北京", True)


def test_error_rates_and_distractor_counts():
    rows = [{
        "dataset_id": "en", "condition": "hotwords", "language": "en",
        "reference": "Hello Alice.", "prediction": "Hello Bob.",
        "positives": ["Alice"], "negatives": ["Bob", "art"],
    }, {
        "dataset_id": "zh", "condition": "hotwords", "language": "Chinese",
        "reference": "你好，北京！", "prediction": "你好，背景。",
        "positives": ["北京"], "negatives": ["背景"],
    }]
    result = summarize(rows)
    assert result["en/hotwords"]["error_rate"] == 0.5
    assert result["en/hotwords"]["hotword_recall"] == 0
    assert result["en/hotwords"]["distractor_false_alarm_rate"] == 0.5
    assert result["zh/hotwords"]["metric"] == "CER"
    assert result["zh/hotwords"]["errors"] == 2
