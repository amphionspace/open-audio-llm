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
    enable_ts_sep = os.environ.get("AMPHION_TSASR_INSERT_SEP", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if not (
        enable_audio_llm
        or enable_qwen3_asr_embeds
        or enable_legacy_amphion_asr
        or enable_funasr
        or enable_ts_sep
    ):
        return

    from vllm import ModelRegistry

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
    if enable_ts_sep:
        # v3 TS-ASR: one concat waveform, independent Mel/conv, learned SEP.
        ModelRegistry.register_model(
            "Qwen3ASRForConditionalGeneration",
            "open_audio_llm.tsasr.vllm_backend:Qwen3ASRForConditionalGeneration",
        )
