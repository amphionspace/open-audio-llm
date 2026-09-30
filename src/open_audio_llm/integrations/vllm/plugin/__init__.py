"""vLLM general plugin entry point.

The default rollout path uses vLLM's generic Transformers backend plus
``patch_registry_subprocess.py``. The direct registry path is kept behind an
explicit opt-in because vLLM 0.18 can segfault while importing its registry in
some environments.
"""


def register():
    import os

    enable_audio_llm = (
        os.environ.get("OPEN_AUDIO_LLM_ENABLE_EXPERIMENTAL_VLLM_REGISTRY") == "1"
    )
    enable_qwen3_asr_embeds = (
        os.environ.get("OPEN_AUDIO_LLM_ENABLE_QWEN3_ASR_EMBEDS") == "1"
    )
    enable_legacy_amphion_asr = (
        os.environ.get("OPEN_AUDIO_LLM_ENABLE_LEGACY_AMPHION_ASR") == "1"
    )
    enable_funasr = os.environ.get("OPEN_AUDIO_LLM_ENABLE_FUNASR_VLLM") == "1"
    enable_target_sot = os.environ.get("OPEN_AUDIO_LLM_TARGET_SOT_VLLM") == "1"
    if not (
        enable_audio_llm
        or enable_qwen3_asr_embeds
        or enable_legacy_amphion_asr
        or enable_funasr
        or enable_target_sot
    ):
        return

    from vllm import ModelRegistry

    if enable_target_sot:
        ModelRegistry.register_model(
            "Qwen3ASRTargetSOTForVLLM",
            "open_audio_llm.integrations.vllm.plugin.qwen3_asr_embeds:Qwen3ASRTargetSOTForVLLM",
        )

    if enable_audio_llm:
        ModelRegistry.register_model(
            "AudioLLMForConditionalGeneration",
            "open_audio_llm.integrations.vllm.plugin.model:AudioLLMForVLLM",
        )
    if enable_qwen3_asr_embeds:
        ModelRegistry.register_model(
            "Qwen3ASRForConditionalGeneration",
            "open_audio_llm.integrations.vllm.plugin.qwen3_asr_embeds:"
            "Qwen3ASRForVLLMWithEmbeds",
        )
    if enable_legacy_amphion_asr:
        ModelRegistry.register_model(
            "AmphionASRForConditionalGeneration",
            "open_audio_llm.integrations.vllm.plugin.legacy_amphion_asr.amphion_asr:"
            "AmphionASRForVLLM",
        )
    if enable_funasr:
        ModelRegistry.register_model(
            "FunASRForConditionalGeneration",
            "open_audio_llm.integrations.vllm.plugin.funasr.funasr_model:"
            "FunASRForVLLM",
        )
