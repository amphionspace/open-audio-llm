import pytest

pytest.importorskip("rapidfuzz")

from open_audio_llm.rl.tsasr_grpo.reward import reward  # noqa: E402
from open_audio_llm.rl.tsasr_grpo.text import errors, parse_output  # noqa: E402


def test_scoring_ignores_punctuation_and_case_like_the_benchmark():
    assert errors("今天天气很好", "今天，天气很好。", "zh") == (0, 6)
    assert errors("hello world", "Hello, world!", "en") == (0, 2)


def test_absent_target_rewards_only_silence():
    assert reward("", "", "zh") == 1.0
    assert reward("", "。", "zh") == 1.0
    assert reward("", "别人的话", "zh") == 0.0


def test_present_target_orders_correct_partial_miss_and_wrong_speaker():
    target = "今天天气很好"
    correct = reward(target, target, "zh", miss_reward=-0.5)
    partial = reward(target, "今天天气", "zh", miss_reward=-0.5)
    miss = reward(target, "", "zh", miss_reward=-0.5)
    wrong = reward(target, "明天我们一起去公园散步吧", "zh", miss_reward=-0.5)
    assert correct == 1.0
    assert correct > partial > miss
    assert -1.0 <= wrong < 0.0


def test_miss_reward_is_configurable():
    assert reward("今天天气很好", "", "zh", miss_reward=0.0) == 0.0


def test_parse_output_strips_the_language_header():
    assert parse_output("language Chinese<asr_text>你好<|im_end|>") == "你好"
    assert parse_output("language None<asr_text><|im_end|>") == ""
    assert parse_output("language None") == ""
