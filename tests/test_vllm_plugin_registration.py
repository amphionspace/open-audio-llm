import pytest

from open_audio_llm.integrations.vllm.plugin import register


def test_plugin_refuses_two_qwen3_asr_classes_for_one_architecture(monkeypatch):
    monkeypatch.setenv('OPEN_AUDIO_LLM_ENABLE_QWEN3_ASR_EMBEDS', '1')
    monkeypatch.setenv('AMPHION_TSASR_INSERT_SEP', '1')
    with pytest.raises(ValueError, match='enable one'):
        register()
