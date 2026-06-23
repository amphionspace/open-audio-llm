"""Optional legacy AmphionASR vLLM plugin registration."""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "AmphionASRForConditionalGeneration",
        "open_audio_llm.integrations.vllm.plugin.legacy_amphion_asr.amphion_asr:"
        "AmphionASRForVLLM",
    )
