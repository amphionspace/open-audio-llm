import importlib

import pytest

from open_audio_llm.integrations.ms_swift.rewards.base import (
    hotword_match_accuracy,
    parse_structured_answer,
)
from open_audio_llm.integrations.ms_swift.rewards.plugin import (
    ASRFormatReward,
    HotwordReward,
)


def test_structured_answer_parses_language_hotwords_and_transcription():
    parsed = parse_structured_answer(
        "Language: zh-cn\nHotwords: 北京,清华\nTranscription: 我在北京"
    )

    assert parsed["language"] == "zh-cn"
    assert parsed["hotwords"] == ["北京", "清华"]
    assert parsed["transcription"] == "我在北京"


def test_hotword_candidate_match_accuracy():
    assert hotword_match_accuracy(["北京"], ["北京"], ["北京", "上海"]) == 1.0
    assert hotword_match_accuracy(["上海"], ["北京"], ["北京", "上海"]) == 0.0


def test_grpo_rewards_include_format_and_candidate_hotwords():
    assert ASRFormatReward()(
        ["Language: zh-cn\nHotwords: N/A\nTranscription: hello"]
    ) == [1.0]

    reward = HotwordReward()(
        ["Language: zh-cn\nHotwords: 北京\nTranscription: x"],
        ["Language: zh-cn\nHotwords: 北京\nTranscription: x"],
        candidate_hotwords=["北京,上海"],
    )

    assert reward == [1.0]


def test_legacy_checkpoint_key_remap():
    pytest.importorskip("transformers")
    module = importlib.import_module(
        "open_audio_llm.integrations.hf.convert_legacy_checkpoint"
    )

    assert (
        module.remap_legacy_key("encoder.layers.0.weight")
        == "audio_tower.encoder.layers.0.weight"
    )
    assert (
        module.remap_legacy_key("encoder_projector.proj.1.weight")
        == "connector.proj.1.weight"
    )
    assert module.remap_legacy_key("llm.model.embed_tokens.weight") == (
        "language_model.model.embed_tokens.weight"
    )
    assert module.remap_legacy_key("prompt_embedding.weight") == (
        "slot_merger.prompt_embedding.weight"
    )
