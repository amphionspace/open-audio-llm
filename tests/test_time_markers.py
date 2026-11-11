from types import SimpleNamespace

import numpy as np
import pytest

from open_audio_llm.audio.time_markers import (
    ENV,
    marked_placeholders,
    marker_layout,
    time_marker_seconds,
)


def test_layout_marks_every_complete_interval_like_moss():
    # 120 s at 13 tokens/s: a marker after every 65 tokens, the last one on the clip's end.
    layout = marker_layout(1560, 5)
    assert layout[:4] == [65, "5", 65, "10"]
    assert layout[-2:] == [65, "120"]
    assert sum(p for p in layout if isinstance(p, int)) == 1560
    # Shorter than one interval: no marker; a partial tail keeps its tokens after the last marker.
    assert marker_layout(64, 5) == [64]
    assert marker_layout(70, 5) == [65, "5", 5]
    assert marker_layout(100, 0) == [100]


def test_placeholders_keep_the_audio_token_count():
    span = marked_placeholders("A", 140, 5)
    assert span.count("A") == 140
    assert span[65:66] == ["5"] and span[131:133] == ["1", "0"]


def test_marker_seconds_reads_the_environment(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert time_marker_seconds() == 0
    monkeypatch.setenv(ENV, "5")
    assert time_marker_seconds() == 5
    monkeypatch.setenv(ENV, "-1")
    with pytest.raises(ValueError):
        time_marker_seconds()


def test_template_writes_markers_between_audio_placeholders(monkeypatch):
    from open_audio_llm.integrations.ms_swift import register_qwen3_asr as module

    if module.Template is object:
        pytest.skip("ms-swift not installed")
    template = object.__new__(module.Qwen3ASRTemplate)
    template._hop_length = 160
    wav = np.zeros(16000 * 12, dtype=np.float32)  # 12 s -> 156 tokens, markers after 5 s and 10 s
    inputs = SimpleNamespace(audios=["sample.wav"], _audio_llm_waveforms=[wav])
    monkeypatch.setattr(module.Qwen3ASRTemplate, "_mix_source", lambda self, inputs: None)
    monkeypatch.delenv(ENV, raising=False)
    plain = template.replace_tag("audio", 0, inputs)
    monkeypatch.setenv(ENV, "5")
    marked = template.replace_tag("audio", 0, inputs)
    n = module._get_n_audio_tokens(wav, 160)
    assert plain == ["<|audio_start|>"] + ["<|audio_pad|>"] * n + ["<|audio_end|>"]
    assert marked == ["<|audio_start|>"] + marked_placeholders("<|audio_pad|>", n, 5) + ["<|audio_end|>"]
    assert marked.count("<|audio_pad|>") == n and "5" in marked and "1" in marked


def test_plugin_refuses_time_markers_with_ts_sep(monkeypatch):
    from open_audio_llm.integrations.vllm.plugin import register

    monkeypatch.setenv(ENV, "5")
    monkeypatch.setenv("AMPHION_TSASR_INSERT_SEP", "1")
    with pytest.raises(ValueError, match="enable one"):
        register()
