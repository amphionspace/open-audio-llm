"""ms-swift template helpers for AudioLLM samples."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

from open_audio_llm.data.augment import augment_features

try:  # pragma: no cover - optional ms-swift dependency
    from swift.template import Template
    from swift.template.vision_utils import load_audio, load_batch
    from swift.utils import get_env_args
except Exception:  # pragma: no cover
    Template = object
    load_audio = None
    load_batch = None

    def get_env_args(name, type_func, default):
        return default


@dataclass
class AudioLLMSample:
    messages: list[dict[str, Any]]
    audios: list[str]
    task: str = "asr"
    solution: str | None = None
    candidate_hotwords: str | None = None


def normalize_audio_sample(row: dict[str, Any]) -> AudioLLMSample:
    audios = row.get("audios") or row.get("audio") or []
    if isinstance(audios, str):
        audios = [audios]
    return AudioLLMSample(
        messages=row.get("messages", []),
        audios=list(audios),
        task=row.get("task", "asr"),
        solution=row.get("solution"),
        candidate_hotwords=row.get("candidate_hotwords"),
    )


class AudioLLMTemplate(Template):
    """ms-swift template that extracts audio features for AudioLLM models."""

    placeholder_tokens = ["<speech>"]

    def generate(self, model, *args, **kwargs):
        output = super().generate(model, *args, **kwargs)
        if kwargs.get("input_features") is None or kwargs.get("feature_lens") is None:
            return output
        # AudioLLM returns only completions when generating from audio embeddings.
        # Restore the text prefix expected by ms-swift's get_generate_ids.
        sequences = output if isinstance(output, torch.Tensor) else output.sequences
        input_ids = kwargs["input_ids"]
        input_ids = input_ids.repeat_interleave(
            sequences.shape[0] // input_ids.shape[0], dim=0
        )
        sequences = torch.cat([input_ids, sequences], dim=-1)
        if isinstance(output, torch.Tensor):
            return sequences
        output.sequences = sequences
        return output

    def init_env_args(self) -> None:
        if hasattr(super(), "init_env_args"):
            super().init_env_args()
        feature_extractor = getattr(self.processor, "feature_extractor", None)
        if feature_extractor is None:
            from transformers import WhisperFeatureExtractor

            feature_size = get_env_args("feature_size", int, 128)
            feature_extractor = WhisperFeatureExtractor(
                feature_size=feature_size,
                sampling_rate=16000,
            )
            self._feature_extractor = feature_extractor
        self.sampling_rate = get_env_args(
            "sampling_rate",
            int,
            int(getattr(feature_extractor, "sampling_rate", 16000)),
        )

    @property
    def feature_extractor(self):
        return getattr(self, "_feature_extractor", None) or self.processor.feature_extractor

    def replace_tag(self, media_type, index: int, inputs) -> list[str]:
        if media_type != "audio":
            raise ValueError(f"AudioLLMTemplate only supports audio, got {media_type}")
        return ["<start_speech><speech><end_speech>"]

    def _encode(self, inputs) -> dict[str, Any]:
        encoded = super()._encode(inputs)
        if not getattr(inputs, "audios", None):
            return encoded
        if load_batch is None or load_audio is None:
            raise RuntimeError("ms-swift vision_utils is required to load audio")

        audios = load_batch(
            inputs.audios,
            load_func=partial(load_audio, sampling_rate=self.sampling_rate),
        )
        audio_inputs = self.feature_extractor(
            audios,
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        attention_mask = audio_inputs.pop("attention_mask")
        encoded["feature_lens"] = attention_mask.sum(dim=-1).long()
        # WhisperFeatureExtractor returns (B, n_mels, T); the model consumes B,T,F.
        max_frames = int(encoded["feature_lens"].max().item())
        features = augment_features(
            audio_inputs["input_features"][..., :max_frames],
            encoded["feature_lens"],
            (getattr(inputs, "extra_kwargs", None) or {}).get("audio_augmentation"),
        )
        encoded["input_features"] = (
            features.transpose(1, 2).contiguous()
        )
        return encoded

    def _data_collator(
        self,
        batch: list[dict[str, Any]],
        *,
        padding_to: int | None = None,
    ) -> dict[str, Any]:
        result = super()._data_collator(batch, padding_to=padding_to)
        feature_batches = [
            item["input_features"]
            for item in batch
            if item.get("input_features") is not None
        ]
        length_batches = [
            item["feature_lens"] for item in batch if item.get("feature_lens") is not None
        ]
        if not feature_batches:
            return result

        slot_counts = {features.shape[0] for features in feature_batches}
        if len(slot_counts) != 1:
            raise ValueError(
                "Mixed audio-slot counts in one batch are not supported; "
                f"got {sorted(slot_counts)}"
            )
        slot_count = slot_counts.pop()
        max_frames = max(features.shape[1] for features in feature_batches)
        feature_batches = [
            F.pad(features, (0, 0, 0, max_frames - features.shape[1]))
            for features in feature_batches
        ]
        if slot_count == 1:
            result["input_features"] = torch.cat(feature_batches, dim=0)
            result["feature_lens"] = torch.cat(length_batches, dim=0)
            return result

        result["input_features"] = [
            torch.cat([features[slot_idx : slot_idx + 1] for features in feature_batches])
            for slot_idx in range(slot_count)
        ]
        result["feature_lens"] = [
            torch.cat([lengths[slot_idx : slot_idx + 1] for lengths in length_batches])
            for slot_idx in range(slot_count)
        ]
        return result
