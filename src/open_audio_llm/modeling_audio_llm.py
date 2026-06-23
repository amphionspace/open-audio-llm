"""Hugging Face model for composable Audio-LLM checkpoints."""

from __future__ import annotations

import importlib
from typing import Optional, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from open_audio_llm.configuration_audio_llm import AudioLLMConfig
from open_audio_llm.merge import SlotMerger
from open_audio_llm.registry import audio_tower_registry, connector_registry

# Register built-ins without making optional encoder dependencies mandatory.
importlib.import_module("open_audio_llm.audio.base")
importlib.import_module("open_audio_llm.connectors.mlp_downsample")
importlib.import_module("open_audio_llm.connectors.pooling")
importlib.import_module("open_audio_llm.connectors.qformer")

try:  # pragma: no cover - optional dependency path
    importlib.import_module("open_audio_llm.audio.qwen3_asr")
except Exception:
    pass

try:  # pragma: no cover - optional dependency path
    importlib.import_module("open_audio_llm.audio.qwen3_omni")
except Exception:
    pass


_GENERATE_METADATA_KEYS = frozenset(
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


class AudioLLMPreTrainedModel(PreTrainedModel):
    config_class = AudioLLMConfig
    base_model_prefix = "audio_llm"
    supports_gradient_checkpointing = True


class AudioLLMForConditionalGeneration(AudioLLMPreTrainedModel):
    """Audio tower + connector + HF causal LM.

    The forward path supports one or more audio slots. A single tensor input is
    treated as the main audio slot; a list of tensors is treated as ordered
    slots, for example `[enrollment, mixed]` in TS-ASR.
    """

    def __init__(self, config: AudioLLMConfig):
        super().__init__(config)
        self.audio_tower = audio_tower_registry.build(
            config.audio_tower_config.type,
            config.audio_tower_config,
        )
        self.connector = connector_registry.build(
            config.connector_config.type,
            config.connector_config,
        )
        text_config = config.text_config
        if isinstance(text_config, dict):
            if text_config:
                text_config = AutoConfig.for_model(**text_config)
            else:
                text_config = AutoConfig.for_model(
                    "gpt2",
                    n_embd=config.connector_config.output_dim,
                    n_layer=2,
                    n_head=2,
                    vocab_size=32000,
                )
        self.language_model = AutoModelForCausalLM.from_config(text_config)
        hidden_size = self._hidden_size(self.language_model.config)
        self.slot_merger = SlotMerger(config, hidden_size=hidden_size)
        self.post_init()

    @staticmethod
    def _hidden_size(config) -> int:
        return int(
            getattr(config, "hidden_size", None)
            or getattr(config, "n_embd", None)
            or getattr(config, "d_model", None)
        )

    @staticmethod
    def _normalize_to_btf(features: torch.Tensor, expected_dim: int) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected 3-D features, got {tuple(features.shape)}")
        if features.shape[-1] == expected_dim:
            return features.contiguous()
        if features.shape[1] == expected_dim:
            return features.transpose(1, 2).contiguous()
        if features.shape[-1] in (80, 128):
            return features.contiguous()
        return features.transpose(1, 2).contiguous()

    def _encode_one(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = lengths.to(dtype=torch.long, device=features.device)
        expected_dim = int(getattr(self.config.audio_tower_config, "input_dim", 0))
        features = self._normalize_to_btf(features, expected_dim)
        max_len = int(lengths.max().item()) if lengths.numel() else features.shape[1]
        features = features[:, :max_len, :]
        tower_dtype = next(self.audio_tower.parameters(), features).dtype
        hidden, hidden_lengths = self.audio_tower(features.to(tower_dtype), lengths)
        return self.connector(hidden, hidden_lengths)

    @staticmethod
    def _as_list(value):
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _encode_audio_slots(
        self,
        input_features: Union[torch.Tensor, list[torch.Tensor]],
        feature_lens: Union[torch.Tensor, list[torch.Tensor]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        feature_list = self._as_list(input_features)
        length_list = self._as_list(feature_lens)
        if len(feature_list) != len(length_list):
            raise ValueError("input_features and feature_lens slot counts differ")
        projected, projected_lens = [], []
        for features, lengths in zip(feature_list, length_list):
            slot_hidden, slot_lens = self._encode_one(features, lengths)
            projected.append(slot_hidden)
            projected_lens.append(slot_lens)
        return projected, projected_lens

    def _use_eager_attention_for_text_fallback(self) -> None:
        # vLLM's generic Transformers backend injects attention state only for
        # the wrapped top-level model. The nested HF language model must use its
        # own eager attention path during text-only rollout initialization.
        config = getattr(self.language_model, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"

    @staticmethod
    def _is_vllm_transformers_call(kwargs: dict) -> bool:
        return "attention_instances" in kwargs or "position_ids" in kwargs

    def _language_model_hidden_forward(
        self,
        *,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[tuple],
        inputs_embeds: Optional[torch.FloatTensor],
        use_cache: Optional[bool],
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        kwargs: dict,
    ):
        base_model = getattr(self.language_model, "model", None)
        if base_model is None:
            return None
        model_kwargs = {key: value for key, value in kwargs.items() if key != "attention_instances"}
        return base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
            **model_kwargs,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_features: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        feature_lens: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if past_key_values is not None or inputs_embeds is not None:
            self._use_eager_attention_for_text_fallback()
            if self._is_vllm_transformers_call(kwargs):
                hidden_outputs = self._language_model_hidden_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    kwargs=kwargs,
                )
                if hidden_outputs is not None:
                    return hidden_outputs
            return self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

        if input_features is None or feature_lens is None:
            if (
                input_ids is not None
                and self.config.default_speech_token_id is not None
                and (input_ids == self.config.default_speech_token_id).any()
            ):
                raise ValueError("input_features and feature_lens are required")
            self._use_eager_attention_for_text_fallback()
            if self._is_vllm_transformers_call(kwargs):
                hidden_outputs = self._language_model_hidden_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    kwargs=kwargs,
                )
                if hidden_outputs is not None:
                    return hidden_outputs
            return self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        audio_features, audio_lengths = self._encode_audio_slots(input_features, feature_lens)
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        merged_embeds, merged_mask, merged_labels = self.slot_merger(
            audio_features,
            audio_lengths,
            text_embeds,
            input_ids,
            attention_mask,
            labels,
        )
        outputs = self.language_model(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            labels=merged_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            return outputs
        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_features: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        feature_lens: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
        **kwargs,
    ):
        kwargs = {
            key: value for key, value in kwargs.items() if key not in _GENERATE_METADATA_KEYS
        }
        if input_features is None or feature_lens is None:
            if (
                input_ids is not None
                and self.config.default_speech_token_id is not None
                and (input_ids == self.config.default_speech_token_id).any()
            ):
                raise ValueError("input_features and feature_lens are required")
            self._use_eager_attention_for_text_fallback()
            return self.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        audio_features, audio_lengths = self._encode_audio_slots(input_features, feature_lens)
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        merged_embeds, merged_mask, _ = self.slot_merger(
            audio_features,
            audio_lengths,
            text_embeds,
            input_ids,
            attention_mask,
        )
        return self.language_model.generate(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            **kwargs,
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)
