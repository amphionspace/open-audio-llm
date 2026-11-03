"""Legacy AmphionASR registration plugin for ms-swift.

Registers model architecture, loader, and template so that ms-swift can
load AmphionASR HuggingFace checkpoints for SFT / GRPO training.

Usage::

    swift sft --model <hf_model_dir> \
        --external_plugins src/open_audio_llm/integrations/ms_swift/register_legacy_amphion_asr.py ...

Design follows the built-in ``Qwen2AudioTemplate`` / ``Qwen2AudioLoader``
pattern — the model's ``forward()`` handles audio encoding and feature
merging internally, so the template only needs to extract mel features and
pass them through.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
from transformers import PreTrainedModel

from swift.model import (
    Model,
    ModelGroup,
    ModelLoader,
    ModelMeta,
    MultiModelKeys,
    register_model,
    register_model_arch,
)
from swift.template import Template, TemplateMeta, register_template
from swift.template.vision_utils import load_audio, load_batch
from swift.utils import get_env_args, get_logger

logger = get_logger()

# -----------------------------------------------------------------------
# Workaround: PyTorch 2.10 added strict=True to LRScheduler._update_lr's
# zip() which breaks when DeepSpeed reorganises optimizer param groups.
# Replace _update_lr with a faithful copy of the upstream 2.10 logic that
# only differs in dropping the strict flag. NOTE: the previous version of
# this patch silently dropped `self.last_epoch += 1`, which pinned
# last_epoch at -1, made cosine warmup return a tiny *negative* lr
# (~-1e-8), and caused models to drift in the wrong direction throughout
# training (train+eval loss steadily rose). See investigation in
# exp/qwen3asr_4b_ts_magicdata_sft logging.jsonl.
# -----------------------------------------------------------------------
import torch.optim.lr_scheduler as _lr_sched

_orig_update_lr = getattr(_lr_sched.LRScheduler, '_update_lr', None)
if _orig_update_lr is not None:
    def _patched_update_lr(self, epoch=None):
        with _lr_sched._enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, '_get_closed_form_lr'):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            if isinstance(lr, dict):
                for key, val in lr.items():
                    param_group[key] = val
            else:
                _lr_sched._update_param_group_val(param_group, 'lr', lr)

        self._last_lr = [
            group.get('lr', group.get('d_model'))
            for group in self.optimizer.param_groups
        ]

    _lr_sched.LRScheduler._update_lr = _patched_update_lr
    logger.info('Patched LRScheduler._update_lr (last_epoch fix + non-strict zip).')

# -----------------------------------------------------------------------
# 1. Architecture mapping
# -----------------------------------------------------------------------

register_model_arch(
    MultiModelKeys(
        'amphion_asr',
        language_model='language_model',
        aligner=['multi_modal_projector', 'prompt_embedding'],
        vision_tower='audio_encoder',
    ))

# -----------------------------------------------------------------------
# 2. Model loader
# -----------------------------------------------------------------------


class AmphionASRLoader(ModelLoader):
    """Load an AmphionASR HuggingFace checkpoint.

    The model directory contains ``auto_map`` in ``config.json`` so
    ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``
    resolves to ``AmphionASRForConditionalGeneration`` automatically.
    """

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from transformers import AutoModelForCausalLM
        self.auto_model_cls = self.auto_model_cls or AutoModelForCausalLM
        return super().get_model(model_dir, *args, **kwargs)


# -----------------------------------------------------------------------
# 3. Template
# -----------------------------------------------------------------------


class AmphionASRTemplate(Template):
    """Template for AmphionASR audio-to-text models.

    Mirrors ``Qwen2AudioTemplate`` — audio features are extracted via
    ``WhisperFeatureExtractor`` and passed to the model as
    ``input_features`` / ``feature_lens``.  The model handles encoding,
    projection, and merging with text embeddings in its ``forward()``.
    """

    placeholder_tokens = ['<speech>']

    def init_env_args(self) -> None:
        super().init_env_args()
        fe = getattr(self.processor, 'feature_extractor', None)
        if fe is None:
            from transformers import WhisperFeatureExtractor
            model_dir = getattr(self.processor, 'name_or_path', None)
            if model_dir:
                try:
                    fe = WhisperFeatureExtractor.from_pretrained(model_dir)
                except Exception:
                    fe = None
            if fe is None:
                fe = WhisperFeatureExtractor(sampling_rate=16000)
            self._feature_extractor = fe
        self.sampling_rate = get_env_args('sampling_rate', int, fe.sampling_rate)

    def replace_tag(
        self,
        media_type: Literal['image', 'video', 'audio'],
        index: int,
        inputs,
    ) -> list:
        assert media_type == 'audio'
        return ['<start_speech><speech><end_speech>']

    @property
    def feature_extractor(self):
        return getattr(self, '_feature_extractor', None) or self.processor.feature_extractor

    def _encode(self, inputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        if inputs.audios:
            try:
                audios = load_batch(
                    inputs.audios,
                    load_func=partial(load_audio, sampling_rate=self.sampling_rate),
                )
            except Exception as e:
                # librosa raises on truncated / unsupported / missing files.
                # Re-raise as ValueError so ms-swift's LazyLLMDataset catches
                # it (see swift/dataset/utils.py:LazyLLMDataset.__getitem__),
                # logs a warning, and randomly resamples another row instead
                # of poisoning the batch with zero-length features.
                raise ValueError(
                    f'AmphionASRTemplate: load_audio failed for '
                    f'{inputs.audios!r}: {e}') from e

            audio_inputs = self.feature_extractor(
                audios,
                sampling_rate=self.sampling_rate,
                return_attention_mask=True,
                return_tensors='pt',
            )
            mask = audio_inputs.pop('attention_mask')
            feat_lens = mask.sum(dim=-1).long()
            # Guard against the modeling_qwen3_asr.py:690 corner case where
            # feature_lens == 0 makes chunk_lengths collapse to a 0-d tensor
            # that subsequent indexing (chunk_lengths[tail_chunk_index] = ...)
            # cannot write into. The convert.py audio_sanity_check is the
            # primary defence; this is the in-flight backstop for any sample
            # that slipped through (e.g. file silently truncated after the
            # JSONL was generated).
            if (feat_lens <= 0).any():
                raise ValueError(
                    f'AmphionASRTemplate: zero-length feature_lens '
                    f'{feat_lens.tolist()} for audios={inputs.audios!r}')
            encoded['feature_lens'] = feat_lens
            # Model expects (B, T, n_mels); feature_extractor returns (B, n_mels, T)
            encoded['input_features'] = audio_inputs['input_features'].transpose(1, 2)
        return encoded

    def _data_collator(
        self,
        batch: List[Dict[str, Any]],
        *,
        padding_to: Optional[int] = None,
    ) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        input_features = [
            b['input_features'] for b in batch
            if b.get('input_features') is not None
        ]
        feature_lens = [
            b['feature_lens'] for b in batch
            if b.get('feature_lens') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_lens'] = torch.concat(feature_lens)
        return res


# -----------------------------------------------------------------------
# 4. Registration
# -----------------------------------------------------------------------

register_model(
    ModelMeta(
        'amphion_asr',
        [
            ModelGroup([
                Model('', 'amphion-asr'),
            ], 'amphion_asr'),
        ],
        AmphionASRLoader,
        template='amphion_asr',
        model_arch='amphion_asr',
        architectures=['AmphionASRForConditionalGeneration'],
        is_multimodal=True,
        requires=['qwen_asr', 'librosa'],
        tags=['audio'],
    ))

register_template(
    TemplateMeta(
        'amphion_asr',
        prefix=[],
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        suffix=['<|im_end|>'],
        default_system=None,
        template_cls=AmphionASRTemplate,
    ))


# -----------------------------------------------------------------------
# Convenience entry point
# -----------------------------------------------------------------------

def register_all():
    """No-op — registrations happen at import time via module-level calls."""
