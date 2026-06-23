"""Runtime compatibility patches for ms-swift training.

The patches are deliberately version-gated and idempotent. They address known
GRPO paths where ms-swift forwards dataset/reward metadata into model
generation kwargs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

GRPO_NON_MODEL_KEYS = frozenset(
    {
        "solution",
        "prompt_id",
        "request_id",
        "reward_model",
        "reward",
        "rollout_infos",
        "add_eos",
        "candidate_hotwords",
        "task",
        "dataset_id",
        "duration",
    }
)


def strip_grpo_non_model_keys(kwargs: dict) -> dict:
    """Return generation kwargs without GRPO-only metadata fields."""

    return {key: value for key, value in kwargs.items() if key not in GRPO_NON_MODEL_KEYS}


def apply_patches() -> None:
    """Apply safe ms-swift patches when ms-swift is installed."""

    try:
        import swift  # noqa: F401
    except ImportError:
        return

    _patch_prepare_generate_kwargs()
    _patch_qwen25_omni_position_ids()


def _patch_prepare_generate_kwargs() -> None:
    try:
        from swift.template.base import Template
    except Exception:
        return

    if getattr(Template.prepare_generate_kwargs, "_open_audio_llm_patched", False):
        return

    original = Template.prepare_generate_kwargs

    def patched(self, generate_kwargs, *, model=None):
        result = original(self, generate_kwargs, model=model)
        for key in GRPO_NON_MODEL_KEYS:
            result.pop(key, None)
        return result

    patched._open_audio_llm_patched = True
    Template.prepare_generate_kwargs = patched
    logger.info("Applied ms-swift GRPO generate kwargs patch")


def _patch_qwen25_omni_position_ids() -> None:
    try:
        from swift.template.templates.qwen import Qwen2_5OmniTemplate
    except Exception:
        return

    if getattr(
        Qwen2_5OmniTemplate._get_position_ids,
        "_open_audio_llm_patched",
        False,
    ):
        return

    original = Qwen2_5OmniTemplate._get_position_ids

    def patched(self, inputs):
        result = original(self, inputs)
        if not isinstance(result, dict):
            return {"position_ids": result}
        return result

    patched._open_audio_llm_patched = True
    Qwen2_5OmniTemplate._get_position_ids = patched
    logger.info("Applied ms-swift Qwen2.5-Omni position_ids patch")
