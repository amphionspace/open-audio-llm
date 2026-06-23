"""Qwen3-ASR vLLM adapter with pre-computed audio embedding support.

vLLM's native Qwen3-ASR implementation accepts raw audio and internally
produces ``input_audio_features``. RAG-ASR can already expose the post-projector
audio frames through Triton, so this adapter adds an ``audio_embeds`` branch
that feeds those frames directly into the LLM placeholder positions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRMultiModalDataParser,
    Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo,
    _qwen3asr_field_config,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import PromptReplacement, PromptUpdate


def _qwen3asr_embed_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    fields = _qwen3asr_field_config(hf_inputs)
    fields["audio_embeds"] = MultiModalFieldConfig.batched("audio")
    return fields


def _audio_embed_lengths(audio_embeds) -> list[int]:
    if audio_embeds is None:
        return []
    if isinstance(audio_embeds, torch.Tensor):
        if audio_embeds.ndim == 4:
            return [int(item.squeeze(0).shape[-2]) for item in audio_embeds]
        if audio_embeds.ndim == 3:
            return [int(item.shape[-2]) for item in audio_embeds]
        if audio_embeds.ndim == 2:
            return [int(audio_embeds.shape[0])]
        return []
    lengths: list[int] = []
    for item in audio_embeds:
        if isinstance(item, torch.Tensor):
            if item.ndim == 3 and item.shape[0] == 1:
                item = item.squeeze(0)
            lengths.append(int(item.shape[-2]))
    return lengths


def _split_audio_embeds(audio_embeds) -> list[torch.Tensor]:
    if isinstance(audio_embeds, torch.Tensor):
        if audio_embeds.ndim == 4:
            return [item.squeeze(0) for item in audio_embeds]
        if audio_embeds.ndim == 3:
            return list(audio_embeds.unbind(0))
        if audio_embeds.ndim == 2:
            return [audio_embeds]
        raise ValueError(f"unexpected audio_embeds ndim={audio_embeds.ndim}")

    out: list[torch.Tensor] = []
    for item in audio_embeds:
        if item.ndim == 3 and item.shape[0] == 1:
            item = item.squeeze(0)
        out.append(item)
    return out


class Qwen3ASREmbedMultiModalDataParser(Qwen3ASRMultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, torch.Tensor):
            return DictEmbeddingItems(
                {"audio_embeds": data},
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_qwen3asr_embed_field_config,
            )
        if isinstance(data, dict) and "audio_embeds" in data:
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_qwen3asr_embed_field_config,
            )
        return super()._parse_audio_data(data)


class Qwen3ASREmbedProcessingInfo(Qwen3ASRProcessingInfo):
    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return Qwen3ASREmbedMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class Qwen3ASREmbedDummyInputsBuilder(Qwen3ASRDummyInputsBuilder):
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        return super().get_dummy_mm_data(seq_len, mm_counts, mm_options)


class Qwen3ASREmbedMultiModalProcessor(Qwen3ASRMultiModalProcessor):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # For pre-computed embeddings there is no raw audio for the HF
        # processor to featurise. Tokenise the prompt only; multimodal kwargs
        # are carried separately by vLLM's MultiModalKwargsItems.
        if not (mm_data.get("audios") or mm_data.get("audio")):
            tokenizer = self.info.get_tokenizer()
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _qwen3asr_embed_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        out_mm_data = out_mm_kwargs.get_data()
        audio_embeds = out_mm_data.get("audio_embeds")
        embed_lens = _audio_embed_lengths(audio_embeds)
        if not embed_lens:
            return super()._get_prompt_updates(
                mm_items,
                hf_processor_mm_kwargs,
                out_mm_kwargs,
            )

        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        audio_token_id = tokenizer.get_vocab()[processor.audio_token]

        def get_replacement(item_idx: int):
            return [audio_token_id] * embed_lens[item_idx]

        return [
            PromptReplacement(
                modality="audio",
                target=processor.audio_token,
                replacement=get_replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASREmbedMultiModalProcessor,
    info=Qwen3ASREmbedProcessingInfo,
    dummy_inputs=Qwen3ASREmbedDummyInputsBuilder,
)
class Qwen3ASRForVLLMWithEmbeds(Qwen3ASRForConditionalGeneration):
    """Qwen3-ASR with an ``audio_embeds`` branch for vLLM mm embeds."""

    def _parse_and_validate_audio_input(self, **kwargs: object):
        audio_embeds = kwargs.pop("audio_embeds", None)
        if audio_embeds is not None:
            return {
                "type": "audio_embeds",
                "audio_embeds": audio_embeds,
            }
        return super()._parse_and_validate_audio_input(**kwargs)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if (
                input_key in ("input_audio_features", "audio_embeds")
                and "audio" not in mm_input_by_modality
            ):
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
                    **kwargs
                )
        return mm_input_by_modality

    def _process_audio_input(self, audio_input, *args, **kwargs):
        if audio_input["type"] == "audio_embeds":
            return _split_audio_embeds(audio_input["audio_embeds"])
        return super()._process_audio_input(audio_input, *args, **kwargs)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += tuple(audio_embeddings)
        return multimodal_embeddings

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        seq_len = len(input_tokens)
        if not mm_features:
            llm_positions = (
                torch.arange(seq_len, dtype=torch.long).view(1, -1).expand(3, -1)
            )
            return llm_positions.clone(), 0

        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0

        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset
            feature_data = mm_feature.data

            if "audio_feature_lengths" in feature_data:
                audio_feature_length = feature_data["audio_feature_lengths"].data
                if isinstance(audio_feature_length, torch.Tensor):
                    audio_feature_length = audio_feature_length.item()
                from vllm.model_executor.models.qwen3_asr import (
                    _get_feat_extract_output_lengths,
                )

                audio_len = _get_feat_extract_output_lengths(
                    torch.tensor(audio_feature_length)
                ).item()
            elif "audio_embeds" in feature_data:
                audio_embeds = feature_data["audio_embeds"].data
                audio_len = int(audio_embeds.shape[0])
            else:
                raise KeyError("audio_feature_lengths or audio_embeds")

            text_len = offset - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_positions = (
                torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1)
                + st_idx
            )
            llm_pos_ids_list.append(text_positions)
            st_idx = st_idx + text_len

            audio_positions = (
                torch.arange(audio_len, dtype=torch.long).view(1, -1).expand(3, -1)
                + st_idx
            )
            llm_pos_ids_list.append(audio_positions)

            st = offset + audio_len

        if st < seq_len:
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = seq_len - st
            final_text_positions = (
                torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1)
                + st_idx
            )
            llm_pos_ids_list.append(final_text_positions)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if llm_positions.shape[1] != seq_len:
            raise RuntimeError("Position ids length mismatch with input ids length")

        mrope_position_delta = (llm_positions.max() + 1 - seq_len).item()
        return llm_positions, mrope_position_delta
