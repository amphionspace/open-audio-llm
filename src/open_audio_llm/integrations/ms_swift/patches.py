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
        "audio_slot_count",
        "grpo_meta",
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
    _patch_grpo_rollout_log()


def _patch_grpo_rollout_log() -> None:
    try:
        from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
    except ImportError:
        return
    if getattr(GRPOTrainer._log_rollout, "_open_audio_llm_patched", False):
        return
    GRPOTrainer._log_rollout = log_rollout_with_meta(GRPOTrainer._log_rollout)


def log_rollout_with_meta(original):
    """Wrap ``GRPOTrainer._log_rollout`` to add each prompt's ``grpo_meta`` as a log column.

    ms-swift writes every ``self._logs`` column to completions.jsonl and the W&B table but gathers only
    ``solution`` from the samples. ``grpo_meta`` (origin plan, source, record, online-mix build; see
    grpo_prompts.GRPOPromptPlan) lets a later analysis split rollouts by plan, source, look-alike
    interferers or durations without re-rendering the audio. Every rank gathers it, so it is added only
    when all local samples carry it (they do whenever the prompts come from GRPOPromptPlan).
    """

    def _log_rollout(self, samples):
        original(self, samples)
        if not self.log_completions or not samples or not all("grpo_meta" in s.extra for s in samples):
            return
        from collections import deque

        if "grpo_meta" not in self._logs:
            self._logs["grpo_meta"] = deque(maxlen=self.args.generation_batch_size)
        self._logs["grpo_meta"].extend(
            self._gather_and_flatten([s.extra["grpo_meta"] for s in samples], flatten_level=0))

    _log_rollout._open_audio_llm_patched = True
    return _log_rollout


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
