import pytest

pytest.importorskip("rapidfuzz")

from open_audio_llm.scripts.grpo_mining import score


def item(system, solution):
    return {"system": system, "solution": solution}


def test_keeps_only_presence_or_hotword_variation():
    empty = item("", "language None<asr_text>")
    assert score(empty, ["language None<asr_text>", "language Chinese<asr_text>嗯"])[1]
    assert not score(empty, ["language None<asr_text>"] * 2)[1]
    speech = item("", "language Chinese<asr_text>你好世界")
    assert not score(speech, ["language Chinese<asr_text>你好世界", "language Chinese<asr_text>你好视界"])[1]
    listed = item("Hotwords: 张伟,章玮", "language Chinese<asr_text>找张伟")
    assert score(listed, ["language Chinese<asr_text>找张伟", "language Chinese<asr_text>找章玮"])[1]
