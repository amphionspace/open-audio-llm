# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM 0.32 Qwen3-ASR with the TS-ASR enrollment separator.

Subclasses vLLM's native Qwen3-ASR (``vllm.model_executor.models.qwen3_asr``). With
``AMPHION_TSASR_INSERT_SEP`` off this is the native model; the EAGLE proposer hooks
(``SupportsEagle3``) come from the native class. With it on, each request's waveform is
``[enrollment | mixture]``: the two are featurised and convolved independently and the
learned SEP embedding joins them before the shared audio transformer (``sep_token.py``).
Per request, ``mm_processor_kwargs.enroll_seconds`` sets where the enrollment ends.
"""

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniAudioFeatureInputs,
)
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRDummyInputsBuilder,
)
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration as NativeQwen3ASRForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRMultiModalProcessor as NativeQwen3ASRMultiModalProcessor,
)
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRProcessingInfo as NativeQwen3ASRProcessingInfo,
)
from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3OmniMoeAudioEncoder
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdate

from open_audio_llm.tsasr.concat_audio import configured_enroll_sec
from open_audio_llm.tsasr.sep_token import (
    REQUEST_ENROLL_SEC,
    AudioSepToken,
    audio_token_count,
    conv_pack_features,
    default_enroll_n_frames,
    expand_enroll_mix_lens,
    install_independent_mel_fe,
    rebuild_cu_seqlens,
    stitch_sep_between_segments,
    transformer_window_aftercnn,
    tsasr_sep_enabled,
)

install_independent_mel_fe()


# One Whisper window; the enrollment is extracted as its own clip.
MAX_ENROLL_SECONDS = 30.0


def _audio_pad_count(mel_len: int, enroll: int | None = None) -> int:
    if enroll is None:
        enroll = default_enroll_n_frames()
    return audio_token_count(
        int(mel_len), insert_sep=bool(enroll), enroll_n_frames=int(enroll)
    )


def _int_list(values) -> list[int]:
    if torch.is_tensor(values):
        return [int(x) for x in values.reshape(-1).tolist()]
    return [int(x) for x in values]


def _audio_pad_counts(mel_lens, enroll_lens=None) -> list[int]:
    vals = _int_list(mel_lens)
    enrolls = [None] * len(vals) if enroll_lens is None else _int_list(enroll_lens)
    return [_audio_pad_count(v, e) for v, e in zip(vals, enrolls, strict=True)]


class Qwen3ASRAudioEncoder(Qwen3OmniMoeAudioEncoder):
    """Native encoder; the enrollment and the mixture are convolved apart and joined by SEP."""

    def __init__(self, config, prefix: str = ""):
        super().__init__(config, prefix=prefix)
        self.sep_token = AudioSepToken(config.d_model)

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        aftercnn_lens: torch.Tensor | None = None,
        enroll_n_frames: torch.Tensor | int | None = None,
    ):
        # aftercnn_lens is recomputed per enrollment/mixture segment below.
        del aftercnn_lens
        if enroll_n_frames is None:
            enroll_n_frames = default_enroll_n_frames()
        seg_lens, groups = expand_enroll_mix_lens(feature_lens, enroll_n_frames)
        hidden_states, aftercnn_lens = conv_pack_features(self, input_features, seg_lens)
        if any(use_sep for _, use_sep in groups):
            hidden_states, aftercnn_lens = stitch_sep_between_segments(
                hidden_states, aftercnn_lens, groups, self.sep_token.weight
            )
        window_aftercnn = transformer_window_aftercnn(self)
        # Lengths arrive on the host (keep_on_cpu fields), so cu_seqlens and max_seqlen are
        # built there. The flash-attention wrapper calls max_seqlen.item() in every layer; on a
        # device tensor that is one GPU sync per layer, and on a time-sliced GPU each sync waits
        # for the other processes' slices.
        cu_seqlens = rebuild_cu_seqlens(aftercnn_lens, window_aftercnn, "cpu")
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)
        cu_seqlens = cu_seqlens.to(hidden_states.device, non_blocking=True)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens, max_seqlen)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.proj2(hidden_states)


class Qwen3ASRProcessingInfo(NativeQwen3ASRProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        # Per-request split point; consumed by the multimodal processor, not HF.
        kwargs.pop("enroll_seconds", None)
        return super().get_hf_processor(**kwargs)

    def get_feature_extractor(self, **kwargs: object):
        feature_extractor = super().get_feature_extractor(**kwargs)
        # install_independent_mel_fe() splits enrollment and mixture inside
        # WhisperFeatureExtractor only; any other extractor would featurise them jointly.
        if tsasr_sep_enabled() and not isinstance(feature_extractor, WhisperFeatureExtractor):
            raise TypeError(
                "TS-ASR needs WhisperFeatureExtractor (preprocessor_config.json), "
                f"got {type(feature_extractor).__name__}"
            )
        return feature_extractor


class Qwen3ASRMultiModalProcessor(NativeQwen3ASRMultiModalProcessor):
    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_kwargs = dict(hf_kwargs)
        seconds = hf_kwargs.pop("enroll_seconds", None)
        if not tsasr_sep_enabled():
            if seconds is not None:
                raise ValueError("enroll_seconds needs the TS-ASR service (AMPHION_TSASR_INSERT_SEP=1)")
            return super()._call_hf_processor(hf_data, hf_kwargs)
        audios = hf_data.get("audio") or []
        if len(audios) > 1:
            # The native processor featurises several clips one by one and keeps only the
            # features and masks, which would drop each clip's enrollment length.
            raise ValueError("a TS-ASR prompt takes exactly one [enrollment | mixture] audio")
        # Every TS request carries its own split, so all items in a batch have
        # enroll_feature_lengths; requests without it use the server default.
        seconds = configured_enroll_sec() if seconds is None else float(seconds)
        if not 0 < seconds <= MAX_ENROLL_SECONDS:
            raise ValueError(f"enroll_seconds must be in (0, {MAX_ENROLL_SECONDS}]")
        # The enrollment must end on a Mel hop: the mixture is extracted from the
        # next sample, and vLLM pads only the whole clip. Clients zero-pad the
        # enrollment to a multiple of the hop (at most 10 ms), which leaves its Mel unchanged.
        feature_extractor = self.info.get_feature_extractor()
        hop_seconds = feature_extractor.hop_length / feature_extractor.sampling_rate
        hops = seconds / hop_seconds
        if abs(hops - round(hops)) > 1e-6:
            raise ValueError(
                f"enroll_seconds={seconds} is not a multiple of {hop_seconds} s; "
                "zero-pad the enrollment to whole hops"
            )
        token = REQUEST_ENROLL_SEC.set(seconds)
        try:
            return super()._call_hf_processor(hf_data, hf_kwargs)
        finally:
            REQUEST_ENROLL_SEC.reset(token)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        if "enroll_feature_lengths" in hf_inputs:
            fields["enroll_feature_lengths"] = MultiModalFieldConfig.batched(
                "audio", keep_on_cpu=True
            )
        return fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        audio_token_id = self.info.get_tokenizer().get_vocab()[processor.audio_token]

        out_mm_data = out_mm_kwargs.get_data()
        audio_feature_lengths = out_mm_data.get("audio_feature_lengths")
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        enroll_feature_lengths = out_mm_data.get("enroll_feature_lengths")
        if audio_feature_lengths is not None:
            audio_output_lengths = _audio_pad_counts(audio_feature_lengths, enroll_feature_lengths)
        elif feature_attention_mask is not None:
            audio_output_lengths = _audio_pad_counts(
                feature_attention_mask.sum(-1), enroll_feature_lengths
            )
        else:
            audio_output_lengths = []

        def get_replacement(item_idx: int):
            num_features = audio_output_lengths[item_idx]
            if num_features == 0:
                audio = mm_items.get_items("audio", AudioProcessorItems).get(item_idx)
                raise ValueError(
                    f"The audio {audio} (len={len(audio)}) is too short "
                    "to be represented inside the model"
                )
            return [audio_token_id] * num_features

        return [
            PromptReplacement(
                modality="audio", target=[audio_token_id], replacement=get_replacement
            )
        ]


class _AudioInput(NamedTuple):
    features: Qwen2_5OmniAudioFeatureInputs
    # Per-item enrollment Mel frames from TS requests; None for plain ASR.
    enroll_feature_lengths: torch.Tensor | None


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3ASRMultiModalProcessor,
    info=Qwen3ASRProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class Qwen3ASRForConditionalGeneration(NativeQwen3ASRForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # The native __init__ with Qwen3ASRAudioEncoder in place of Qwen3OmniMoeAudioEncoder;
        # building the native tower first and replacing it would allocate it twice.
        nn.Module.__init__(self)
        self.vllm_config = vllm_config  # needed for torch compile forward context
        thinker_config = vllm_config.model_config.hf_config.thinker_config
        self.config = thinker_config
        # The EAGLE proposer copies the target's config.image_token_index into the draft
        # config when loading a draft for any multimodal target; the text-only draft never reads it.
        self.config.image_token_index = thinker_config.audio_token_id
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.quant_config = vllm_config.quant_config

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = Qwen3ASRAudioEncoder(
                thinker_config.audio_config,
                prefix=maybe_prefix(prefix, "audio_tower"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3ForCausalLM(
                vllm_config=vllm_config.with_hf_config(
                    thinker_config.text_config, architectures=["Qwen3ForCausalLM"]
                ),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        enroll_lens = kwargs.pop("enroll_feature_lengths", None)
        mm_input_by_modality = super()._parse_and_validate_multimodal_inputs(**kwargs)
        if "audio" in mm_input_by_modality:
            mm_input_by_modality["audio"] = _AudioInput(
                mm_input_by_modality["audio"], enroll_lens
            )
        return mm_input_by_modality

    def _process_audio_input(self, audio_input: _AudioInput) -> tuple[torch.Tensor, ...]:
        features, enroll_lens = audio_input
        # Host lengths (keep_on_cpu): the encoder splits audio by them in Python without a
        # GPU sync at every .tolist() (see Qwen3ASRAudioEncoder.forward).
        audio_feature_lengths = torch.as_tensor(features["audio_feature_lengths"]).cpu()
        if torch.is_tensor(enroll_lens):
            enroll_lens = enroll_lens.cpu()
        input_features = features["input_features"]
        audio_features = self.audio_tower(
            input_features.to(self.audio_tower.dtype),
            feature_lens=audio_feature_lengths,
            enroll_n_frames=default_enroll_n_frames() if enroll_lens is None else enroll_lens,
        )
        return audio_features.split(_audio_pad_counts(audio_feature_lengths, enroll_lens))
