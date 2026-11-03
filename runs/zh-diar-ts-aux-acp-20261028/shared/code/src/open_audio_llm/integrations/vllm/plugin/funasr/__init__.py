"""Optional FunASR vLLM plugin registration."""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "FunASRForConditionalGeneration",
        "open_audio_llm.integrations.vllm.plugin.funasr.funasr_model:FunASRForVLLM",
    )
