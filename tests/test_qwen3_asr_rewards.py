import pytest

pytest.importorskip("rapidfuzz")

from open_audio_llm.integrations.ms_swift.rewards import qwen3_asr


def test_accuracy_scores_mixed_units_after_the_header():
    ref = "language Chinese<asr_text>我们用 Python 写代码。"
    assert qwen3_asr.accuracy(ref, ref) == 1.0
    # One substituted unit out of 7: 我 们 用 python 写 代 码.
    assert qwen3_asr.accuracy("language Chinese<asr_text>我们用 python 写代吗", ref) == pytest.approx(1 - 1 / 7)
    assert qwen3_asr.accuracy("language None<asr_text>", ref) == 0.0


def test_empty_reference_rewards_only_silence():
    ref = "language None<asr_text>"
    assert qwen3_asr.accuracy("language None<asr_text>", ref) == 1.0
    assert qwen3_asr.accuracy("language Chinese<asr_text>嗯", ref) == 0.0
    assert qwen3_asr.presence("language None<asr_text>", ref) == 1.0
    assert qwen3_asr.presence("language Chinese<asr_text>嗯", ref) == -1.0


def test_presence_penalizes_a_missed_utterance_only():
    ref = "language English<asr_text>hello there"
    assert qwen3_asr.presence("language None<asr_text>", ref) == -1.0
    assert qwen3_asr.presence("language English<asr_text>hello", ref) == 0.0


def test_repetition_loops_are_scored_as_written():
    ref = "language Chinese<asr_text>好的"
    assert qwen3_asr.accuracy("language Chinese<asr_text>" + "好的" * 5, ref) == 0.0


def test_hotword_recall_minus_distractor_false_alarms():
    ref = "language Chinese<asr_text>请联系张伟处理"
    words = "张伟,章玮,李娜"
    assert qwen3_asr.hotword(ref, ref, words) == 1.0
    # The near-homophone distractor written instead of the spoken name.
    assert qwen3_asr.hotword("language Chinese<asr_text>请联系章玮处理", ref, words) == -1.0
    assert qwen3_asr.hotword("language Chinese<asr_text>请联系处理", ref, words) == 0.0
    assert qwen3_asr.hotword(ref, ref, "N/A") is None
    assert qwen3_asr.hotword(ref, ref, None) is None


def test_hotword_list_comes_from_the_prompt_not_the_dataset_column():
    shown = [{"role": "system", "content": "Hotwords: 张伟,章玮"}, {"role": "user", "content": "<audio>"}]
    hidden = [{"role": "system", "content": ""}, {"role": "user", "content": "<audio>"}]
    assert qwen3_asr.prompt_hotwords(shown) == "张伟,章玮"
    assert qwen3_asr.prompt_hotwords(hidden) is None


def test_ts_prompt_hotword_line_is_read():
    from open_audio_llm.tsasr.ts_prompt import TS_CONCAT_SYSTEM, ts_system

    shown = [{"role": "system", "content": ts_system("张伟,章玮")}, {"role": "user", "content": "<audio>"}]
    plain_ts = [{"role": "system", "content": TS_CONCAT_SYSTEM}, {"role": "user", "content": "<audio>"}]
    assert qwen3_asr.prompt_hotwords(shown) == "张伟,章玮"
    assert qwen3_asr.prompt_hotwords(plain_ts) is None


def test_latin_hotwords_match_whole_words():
    ref = "language English<asr_text>Call Anna today"
    assert qwen3_asr.hotword("language English<asr_text>call anna today", ref, "Anna,Ann") == 1.0
    assert qwen3_asr.hotword("language English<asr_text>call ann today", ref, "Anna,Ann") == -1.0


def test_format_requires_the_native_header():
    assert qwen3_asr.well_formed("language Chinese<asr_text>你好")
    assert qwen3_asr.well_formed("language None<asr_text>")
    assert not qwen3_asr.well_formed("你好")
