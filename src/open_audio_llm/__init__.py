"""Composable Audio-LLM package."""

__all__ = [
    "AudioLLMConfig",
    "AudioLLMForConditionalGeneration",
]


def __getattr__(name):
    if name == "AudioLLMConfig":
        from .configuration_audio_llm import AudioLLMConfig

        return AudioLLMConfig
    if name == "AudioLLMForConditionalGeneration":
        from .modeling_audio_llm import AudioLLMForConditionalGeneration

        return AudioLLMForConditionalGeneration
    raise AttributeError(name)
