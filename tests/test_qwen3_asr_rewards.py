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
    from open_audio_llm.tsasr.ts_prompt import TS_SYSTEM, ts_system

    shown = [{"role": "system", "content": ts_system("张伟,章玮")}, {"role": "user", "content": "<audio>"}]
    plain_ts = [{"role": "system", "content": TS_SYSTEM}, {"role": "user", "content": "<audio>"}]
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


def test_context_reads_each_target_at_its_own_mention():
    ref = "language Chinese<asr_text>开了一家织音琴行。她解释过，织是纺织的织，不是知己的知。"
    target = [{"start": ref.split("<asr_text>")[1].index("织音"), "text": "织音琴行", "alt": "知音琴行"}]
    assert qwen3_asr.context(ref, ref, target) == 1.0
    # The common spelling at the mention, though the explanation that repeats 织 is transcribed right.
    wrong = "language Chinese<asr_text>开了一家知音琴行。她解释过，织是纺织的织，不是知己的知。"
    assert qwen3_asr.context(wrong, ref, target) == -1.0
    other = "language Chinese<asr_text>开了一家之音琴行。她解释过，织是纺织的织，不是知己的知。"
    assert qwen3_asr.context(other, ref, target) == 0.0
    assert qwen3_asr.context(ref, ref, None) is None and qwen3_asr.context(ref, ref, []) is None


def test_context_survives_edits_around_the_target_and_averages_targets():
    ref = "language Chinese<asr_text>我姓章，立早章，章程的章"
    text = ref.split("<asr_text>")[1]
    targets = [{"start": 2, "text": "章", "alt": "张"}, {"start": text.index("立早") + 2, "text": "章", "alt": None}]
    # A word dropped before the target and one inserted after it move nothing.
    assert qwen3_asr.context("language Chinese<asr_text>姓章啊，立早章，章程的章", ref, targets) == 1.0
    assert qwen3_asr.context("language Chinese<asr_text>我姓张，立早章，章程的章", ref, targets) == 0.0  # -1 and +1
    # Without a known alternative a wrong spelling scores 0, not -1.
    assert qwen3_asr.context("language Chinese<asr_text>我姓张，立早张，章程的章", ref, targets) == -0.5


def test_context_penalizes_over_correction_on_confirm_items():
    ref = "language Chinese<asr_text>他说的是张三的张，弓长张"
    confirm = [{"start": 4, "text": "张三", "alt": "章三"}]
    assert qwen3_asr.context("language Chinese<asr_text>他说的是章三的张，弓长张", ref, confirm) == -1.0


def test_context_rejects_a_target_that_is_not_in_the_reference():
    ref = "language Chinese<asr_text>我姓章"
    with pytest.raises(ValueError, match="not at 0"):
        qwen3_asr.context(ref, ref, [{"start": 0, "text": "章", "alt": "张"}])


def test_context_reward_takes_targets_from_grpo_meta():
    import json

    from open_audio_llm.integrations.ms_swift.rewards.plugin import Qwen3ASRContextReward, orms

    ref = "language Chinese<asr_text>我姓章，立早章"
    metas = [json.dumps({"context_targets": [{"start": 2, "text": "章", "alt": "张"}]}), json.dumps({"origin": "replay"})]
    out = "language Chinese<asr_text>我姓张，立早章"
    assert Qwen3ASRContextReward()([out, out], [ref, ref], grpo_meta=metas) == [-1.0, None]
    if orms:
        assert orms["qwen3_asr_context"] is Qwen3ASRContextReward
