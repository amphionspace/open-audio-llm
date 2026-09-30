"""Match inference audio attention to HF SDPA training (one global window).

ms-swift trains with empty ``attn_impl`` → transformers SDPA. The encoder
builds ``cu_seqlens`` from ``n_window_infer=800`` (~8s) but never applies
that mask; SDPA ignores ``cu_seq_lens_*``. vLLM FlashAttention honors the
windows, so mix frames after enrollment cannot see the enroll prompt.

Setting ``n_window_infer`` huge makes FA use a single window. Same idea as
Amphion ``_match_hf_sdpa_audio_attention``.

Opt out: ``COT_AUDIO_CHUNKED_ATTN=1`` (keep config.json windows).
"""
from __future__ import annotations

import logging
import os
from typing import Any

GLOBAL_N_WINDOW_INFER = 10**9
_ENV_CHUNKED = "COT_AUDIO_CHUNKED_ATTN"

logger = logging.getLogger(__name__)


def chunked_attn_requested() -> bool:
    raw = os.environ.get(_ENV_CHUNKED, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def vllm_hf_overrides() -> dict[str, Any]:
    """Nested overrides applied in the parent *and* pickled into EngineCore."""
    if chunked_attn_requested():
        return {}
    return {
        "thinker_config": {
            "audio_config": {
                "n_window_infer": GLOBAL_N_WINDOW_INFER,
            }
        }
    }


def describe_mode() -> str:
    if chunked_attn_requested():
        return (
            f"audio encoder attention: chunked (config n_window_infer; "
            f"{_ENV_CHUNKED}=1)"
        )
    return (
        f"audio encoder attention: global "
        f"(n_window_infer={GLOBAL_N_WINDOW_INFER}; "
        f"set {_ENV_CHUNKED}=1 to keep 800-frame windows)"
    )


def _bump_n_window_infer(obj: Any) -> int | None:
    if obj is None or not hasattr(obj, "n_window_infer"):
        return None
    try:
        old = int(obj.n_window_infer)
    except (TypeError, ValueError):
        return None
    if old >= GLOBAL_N_WINDOW_INFER:
        return old
    obj.n_window_infer = GLOBAL_N_WINDOW_INFER
    return old


def apply_to_hf_model(model: Any) -> list[str]:
    """Force ``n_window_infer`` on a transformers Qwen3-ASR module tree."""
    if chunked_attn_requested() or model is None:
        return []
    patched: list[str] = []
    seen: set[int] = set()

    def _try(obj: Any, label: str) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        old = _bump_n_window_infer(obj)
        if old is not None and old < GLOBAL_N_WINDOW_INFER:
            patched.append(f"{label}:{old}")

    _try(model, type(model).__name__)
    cfg = getattr(model, "config", None)
    _try(cfg, "config")
    thinker = getattr(cfg, "thinker_config", None)
    _try(thinker, "thinker_config")
    _try(getattr(thinker, "audio_config", None), "thinker_config.audio_config")
    _try(getattr(cfg, "audio_config", None), "audio_config")

    modules = getattr(model, "modules", None)
    if callable(modules):
        for mod in modules():
            _try(mod, type(mod).__name__)

    if patched:
        logger.info(
            "global audio attn patched %s → %s",
            patched,
            GLOBAL_N_WINDOW_INFER,
        )
    return patched
