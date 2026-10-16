"""Long-enrollment TS-ASR (AmphionASR v5): PyTorch audio frontend + vLLM text decoding.

The frontend (enrollment encoder, conditioned audio tower, ECAPA; longenroll/frontend.py) turns
each request into post-projector audio embeddings. vLLM's Qwen3-ASR LLM decodes them through the
``audio_embeds`` input of Qwen3ASRForVLLMWithEmbeds; vLLM's own audio tower is loaded with the
checkpoint but never run. Prompts are the training token ids, with the single <|audio_pad|> that
vLLM expands to the embedding length.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch

EOS_IDS = (151643, 151645)  # <|endoftext|>, <|im_end|>: Qwen3-ASR generation_config.eos_token_id


class LongEnrollVLLM:
    backend = "vllm"

    def __init__(self, model_dir, *, max_model_len=4096, gpu_memory_utilization=0.4, max_num_seqs=256,
                 frontend_device="cuda", frontend_batch=16, feature_workers=4, seed=0):
        from importlib.metadata import version

        from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

        from open_audio_llm.longenroll.features import LongEnrollFeatures
        from open_audio_llm.longenroll.frontend import LongEnrollFrontend

        if version("qwen-asr") != "0.0.6":
            raise ValueError("The long-enrollment frontend requires qwen-asr==0.0.6")
        if version("vllm") not in {"0.17.0", "0.18.0"}:
            raise ValueError("The long-enrollment adapter targets vllm 0.17.0/0.18.0")
        model_dir = str(Path(model_dir).resolve())
        processor = Qwen3ASRProcessor.from_pretrained(model_dir)
        self.features = LongEnrollFeatures(processor)
        self.tokenizer = processor.tokenizer
        self.frontend = LongEnrollFrontend.from_pretrained(model_dir, device=frontend_device)
        self.frontend_batch = int(frontend_batch)
        self.feature_workers = int(feature_workers)
        # The engine process loads this package's vLLM plugin through the entry point; the
        # variable selects the Qwen3-ASR class that accepts audio_embeds.
        os.environ["OPEN_AUDIO_LLM_ENABLE_QWEN3_ASR_EMBEDS"] = "1"
        from open_audio_llm.integrations.vllm.plugin import register

        register()
        from vllm import LLM

        self.engine = LLM(model=model_dir, dtype="bfloat16", enable_mm_embeds=True,
                          limit_mm_per_prompt={"audio": 1}, max_model_len=max_model_len,
                          max_num_seqs=max_num_seqs, gpu_memory_utilization=gpu_memory_utilization, seed=seed)
        self.max_model_len = int(max_model_len)
        self.versions = {"vllm": version("vllm"), "qwen-asr": version("qwen-asr"), "torch": torch.__version__}

    def transcribe(self, requests, *, max_new_tokens=256):
        """requests: [{"task": "ts"|"asr"|"hotword", "mixture": wave, "enrollment": wave|None,
        "hotwords": str}] -> [{"raw", "token_ids", "finish_reason"}] in order, plus timings."""
        from vllm import SamplingParams

        from open_audio_llm.longenroll.frontend import embed_requests

        items, embeds, feature_s, frontend_s = embed_requests(self.frontend, self.features, requests,
                                                              self.frontend_batch, self.feature_workers)
        prompts = []
        for item, embed in zip(items, embeds):
            ids = self.features.prompt.build(1, None, system=item["system"])[0]
            if len(ids) - 1 + item["n_audio"] + max_new_tokens > self.max_model_len:
                raise ValueError("Mixture and output exceed max_model_len; no audio was truncated")
            # [1, n_audio, 2048]: the leading axis counts audio items (one per prompt).
            prompts.append({"prompt_token_ids": ids,
                            "multi_modal_data": {"audio": {"audio_embeds": embed.cpu()[None]}}})
        started = time.perf_counter()
        outputs = self.engine.generate(
            prompts, SamplingParams(temperature=0.0, max_tokens=max_new_tokens, skip_special_tokens=False),
            use_tqdm=False)
        llm_s = time.perf_counter() - started
        results = []
        for output in outputs:
            ids = list(output.outputs[0].token_ids)
            cut = next((i for i, token in enumerate(ids) if token in EOS_IDS), len(ids))
            ids = ids[:cut]
            results.append({"raw": self.tokenizer.decode(ids, skip_special_tokens=False), "token_ids": ids,
                            "finish_reason": output.outputs[0].finish_reason})
        timings = {"feature_s": feature_s, "frontend_s": frontend_s, "llm_s": llm_s}
        return results, items, timings
