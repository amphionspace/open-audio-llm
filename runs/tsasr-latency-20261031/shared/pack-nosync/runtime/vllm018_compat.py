"""Load the vendored Qwen3-ASR vLLM backend (patched for vLLM 0.18).

Does not modify AmphionASR, conda, or site-packages.
"""
from __future__ import annotations

import os


def bootstrap_vllm_runtime() -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # Direct vision/mm_encoder import can segfault unless CUDA platform is up first.
    from vllm.config import MultiModalConfig, VllmConfig, get_current_vllm_config  # noqa: F401
    from vllm.platforms import current_platform  # noqa: F401
    from vllm.distributed import get_tensor_model_parallel_rank  # noqa: F401
    from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: F401


def register_vendored_backend() -> None:
    """Overwrite qwen_asr's registry entry with the in-repo vLLM 0.18 copy."""
    from vllm import ModelRegistry
    from vendor.qwen3_asr_vllm import Qwen3ASRForConditionalGeneration

    ModelRegistry.register_model(
        "Qwen3ASRForConditionalGeneration",
        Qwen3ASRForConditionalGeneration,
    )
    print("[eval] vendored qwen3 vLLM backend (v0.18)", flush=True)
