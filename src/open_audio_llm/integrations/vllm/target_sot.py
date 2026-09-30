"""Conditional SOT: checkpoint audio frontend + vLLM-only text generation.

The frontend loads only audio weights; no Transformers text model is created.
"""

from io import BytesIO
from pathlib import Path

import torch

from open_audio_llm.audio.target_sot import (
    attach_separator,
    checkpoint_tensors,
    encode_segments,
    extract_segments,
)
from open_audio_llm.data.target_sot import target_prompt


def read_waveform(path, sampling_rate=16000):
    from fractions import Fraction

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    waveform, rate = sf.read(BytesIO(path) if isinstance(path, bytes) else path,
                             dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if rate != sampling_rate:
        ratio = Fraction(sampling_rate, rate)
        waveform = resample_poly(waveform, ratio.numerator, ratio.denominator)
    return np.asarray(waveform, dtype=np.float32)


class TargetSOTVLLM:
    def __init__(self, model_dir, *, max_model_len=16384, gpu_memory_utilization=.7,
                 frontend_device="cpu"):
        import os
        from importlib.metadata import version

        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
            Qwen3ASRConfig,
        )
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRAudioEncoder,
        )
        from transformers import AutoProcessor

        if version("qwen-asr") != "0.0.6":
            raise ValueError("Target SOT frontend requires qwen-asr==0.0.6")
        if version("vllm") not in {"0.17.0", "0.18.0"}:
            raise ValueError("Target SOT adapter targets vllm 0.17.0/0.18.0")
        model_dir = str(Path(model_dir).resolve())
        config = Qwen3ASRConfig.from_pretrained(model_dir)
        if not getattr(config, "target_sot_audio", False):
            raise ValueError("Checkpoint was not trained with target_sot_audio")
        audio_config = config.thinker_config.audio_config
        audio_config._attn_implementation = "sdpa"
        self.tower = Qwen3ASRAudioEncoder(audio_config)
        attach_separator(self.tower)
        self.tower.load_state_dict(dict(checkpoint_tensors(model_dir, "thinker.audio_tower.")), strict=True)
        self.tower.to(frontend_device).eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        os.environ["OPEN_AUDIO_LLM_TARGET_SOT_VLLM"] = "1"
        from open_audio_llm.integrations.vllm.plugin import register

        register()
        from vllm import LLM

        self.engine = LLM(model=model_dir, hf_overrides={"architectures": ["Qwen3ASRTargetSOTForVLLM"]},
                          enable_mm_embeds=True, limit_mm_per_prompt={"audio": 1},
                          max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization)
        self.max_model_len = max_model_len
        self.backend = "vllm"

    def transcribe(self, mixture, enrollments, mode="all", *, max_tokens=4096):
        from vllm import SamplingParams

        from open_audio_llm.data.sot import TIMESTAMP_SYSTEM

        if enrollments:
            prompt = target_prompt(len(enrollments), mode)
        elif mode == "all":
            prompt = TIMESTAMP_SYSTEM
        else:
            raise ValueError("targets_only requires enrollments")
        waves = [read_waveform(p) for p in [*enrollments, mixture]]
        if any(not 1 <= len(w) / 16000 <= 5 for w in waves[:-1]):
            raise ValueError("Supply enrollment clips of 1--5 seconds")
        features, lengths = extract_segments(self.processor.feature_extractor, waves)
        with torch.inference_mode():
            embeddings = encode_segments(self.tower, features[0].to(next(self.tower.parameters())), lengths).cpu()
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": "<|audio_start|><|audio_pad|><|audio_end|>"}]
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = len(self.processor.tokenizer.encode(text)) - 1 + len(embeddings)
        if prompt_tokens + max_tokens > self.max_model_len:
            raise ValueError("Mixture, enrollments and output exceed max_model_len; no audio was truncated")
        result = self.engine.generate(
            [{"prompt": text, "multi_modal_data": {"audio": {"audio_embeds": embeddings}}}],
            SamplingParams(temperature=0, max_tokens=max_tokens), use_tqdm=False,
        )[0].outputs[0]
        return {"prediction": result.text, "finish_reason": result.finish_reason,
                "backend": self.backend, "duration": len(waves[-1]) / 16000}
