"""Qwen3-ASR in vLLM with elapsed-time markers in the audio placeholder span.

Same span as the training template (``open_audio_llm.audio.time_markers``): the audio token run of each clip with
the elapsed-second digits after every ``OPEN_AUDIO_LLM_AUDIO_TIME_MARKER_SECONDS`` seconds. vLLM 0.18 builds the
prompt one of two ways and both must carry the markers:
- without the processor cache the HF processor expands ``<|audio_pad|>`` into a plain run and vLLM only locates the
  replacement afterwards, so ``_call_hf_processor`` rewrites each run into the marked span;
- with the cache (the server default) the prompt is tokenized as text and vLLM applies the replacement itself.
The replacement is the marked span with ``select_token_id``: only the audio tokens are embedding positions, so the
encoder output still fills exactly the audio tokens and the digits stay ordinary text tokens.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

from open_audio_llm.audio.time_markers import marked_placeholders, time_marker_seconds


class Qwen3ASRTimeMarkerProcessor(Qwen3ASRMultiModalProcessor):
    def _marker_ids(self):
        tokenizer = self.info.get_tokenizer()
        audio_token_id = tokenizer.get_vocab()[self.info.get_hf_processor().audio_token]
        digits = {}
        for d in "0123456789":
            ids = tokenizer.encode(d, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"digit {d!r} is not a single token: {ids}")
            digits[d] = ids[0]
        return audio_token_id, digits

    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs) -> BatchFeature:
        outputs = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
        every = time_marker_seconds()
        if not every or not (mm_data.get("audios") or mm_data.get("audio")):
            return outputs
        audio_token_id, digits = self._marker_ids()
        rows = []
        for row in outputs["input_ids"].tolist():
            out, i = [], 0
            while i < len(row):
                if row[i] != audio_token_id:
                    out.append(row[i])
                    i += 1
                    continue
                j = i
                while j < len(row) and row[j] == audio_token_id:
                    j += 1
                out += marked_placeholders(audio_token_id, j - i, every, digit=digits.__getitem__)
                i = j
            rows.append(out)
        if len({len(r) for r in rows}) != 1:
            raise ValueError("time markers gave rows of different lengths")
        outputs["input_ids"] = torch.tensor(rows, dtype=outputs["input_ids"].dtype)
        if "attention_mask" in outputs:
            outputs["attention_mask"] = torch.ones_like(outputs["input_ids"])
        return outputs

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        updates = super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        every = time_marker_seconds()
        if not every:
            return updates
        audio_token_id, digits = self._marker_ids()

        def marked(base):
            def replacement(item_idx: int):
                run = base(item_idx)
                if any(token != audio_token_id for token in run):
                    raise ValueError("expected a plain audio-token run to mark")
                return PromptUpdateDetails.select_token_id(
                    marked_placeholders(audio_token_id, len(run), every, digit=digits.__getitem__),
                    embed_token_id=audio_token_id,
                )
            return replacement

        out = []
        for update in updates:
            if isinstance(update, PromptReplacement) and update.modality == "audio":
                update = PromptReplacement(modality=update.modality, target=update.target,
                                           replacement=marked(update.replacement))
            out.append(update)
        return out


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRTimeMarkerProcessor,
    info=Qwen3ASRProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class Qwen3ASRForVLLMWithTimeMarkers(Qwen3ASRForConditionalGeneration):
    """Qwen3-ASR whose prompts carry elapsed-time markers among the audio tokens."""
