"""Qwen3-ASR-1.7B registration plugin for ms-swift.

Registers model architecture, loader, and template so that ms-swift can
load Qwen3-ASR-1.7B HuggingFace checkpoints for SFT / GRPO training.

Usage::

    swift sft --model /path/to/Qwen3-ASR-1.7B \
        --external_plugins src/open_audio_llm/integrations/ms_swift/register_qwen3_asr.py ...

Model structure
---------------
The Qwen3-ASR-1.7B model wraps everything under a ``thinker`` sub-module:

    thinker.audio_tower   — Whisper-style encoder + proj1/proj2 (frozen)
    thinker.model         — Qwen3 causal LLM (LoRA target)
    thinker.lm_head       — tied output projection

Audio tokens (from tokenizer_config.json)::

    <|audio_start|>  id=151669
    <|audio_pad|>    id=151676   ← placeholder in prompt
    <|audio_end|>    id=151670

Chat template (exact token sequence)
-------------------------------------
This template strictly follows the original Qwen3-ASR ``chat_template.json``
behavior, which:

  1. Puts **all text** in the system turn.
  2. Puts **all audio tokens** (from every message) in the user turn.
  3. **Completely ignores user-turn text** — the user turn is pure audio.

Training and vLLM-inference token sequences are therefore identical::

    <|im_start|>system
    {enrollment notice if applicable}
    {Language: X  if applicable}
    {Hotwords: w1,w2  if applicable}
    <|im_end|>
    <|im_start|>user
    <|audio_start|><|audio_pad|>...<|audio_pad|><|audio_end|>  ← enrollment (N₁ pad tokens)
    <|audio_start|><|audio_pad|>...<|audio_pad|><|audio_end|>  ← target (N₂ pad tokens)
    <|im_end|>
    <|im_start|>assistant
    language English<asr_text>{transcription text}
    <|im_end|>

Runtime sample format expected by this native-model template::

    {
      "messages": [
        {"role": "system",    "content": "Given the speaker's voice in the first audio.\\nHotwords: xxx"},
        {"role": "user",      "content": "<audio><audio>"},   ← ms-swift expands each <audio> → N pad tokens
        {"role": "assistant", "content": "language Chinese<asr_text>转写文本"}
      ],
      "audios": ["enroll.wav", "target.wav"]
    }

For basic ASR (no enrollment, no hotwords, no language), system is empty
and user has a single ``<audio>`` placeholder.

Compared to Amphion-4B
-----------------------
Amphion-4B embeds instruction text directly in the user turn
(``"Given the speaker's voice:<audio>\\nTranscribe...\\nHotwords: xxx\\n<audio>"``).
This plugin follows Qwen3-ASR's original design: all text → system,
all audio → user.

Model forward signature::

    forward(input_ids, attention_mask, input_features, feature_attention_mask, ...)

    input_features        : (B, n_mels, T)  — native feature extractor format, NOT transposed
    feature_attention_mask: (B, T) bool/int — 1 for valid frames, 0 for padding

CRITICAL — audio placeholder expansion
--------------------------------------
``Qwen3ASRThinkerForConditionalGeneration.forward()`` uses ``masked_scatter`` to
inject audio encoder output embeddings into the ``<|audio_pad|>`` token positions.
The number of ``<|audio_pad|>`` tokens in the sequence MUST equal the encoder output
length for each audio, computed by ``_get_feat_extract_output_lengths(n_frames)``.

The original ``Qwen3ASRProcessor`` does this expansion automatically.  In ms-swift
we must replicate it in ``replace_tag``: instead of returning a single placeholder
``<|audio_start|><|audio_pad|><|audio_end|>``, return
``<|audio_start|> + <|audio_pad|> × N + <|audio_end|>`` where N is computed from
the audio file's duration.

Without this expansion only the first audio frame is used per audio, the model gets
no useful audio signal, and loss plateaus around 3.0 with no convergence.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F
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

from open_audio_llm.data.augment import augment_features

# ---------------------------------------------------------------------------
# Audio token count helpers (mirrors Qwen3ASRProcessor logic)
# ---------------------------------------------------------------------------

def _get_feat_extract_output_lengths(n_frames: int) -> int:
    """Map WhisperFeatureExtractor frame count → audio encoder output token count.

    This is a direct port of ``_get_feat_extract_output_lengths`` from
    ``qwen_asr/core/transformers_backend/processing_qwen3_asr.py``.
    It accounts for the three Conv2D layers (stride 2 each) and the
    chunked attention inside ``Qwen3ASRAudioEncoder.forward()``.
    """
    input_lengths_leave = n_frames % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (n_frames // 100) * 13
    return int(output_lengths)


def _get_n_audio_tokens(wav, hop_length: int, max_length: int) -> int:
    """Count placeholders from the same decoded waveform used for features."""
    # Fixed-window padding keeps the final partial hop in the attention mask.
    # Speed perturbation frequently produces these non-aligned lengths.
    n_frames = min((len(wav) + hop_length - 1) // hop_length, max_length)
    return max(1, _get_feat_extract_output_lengths(n_frames))


logger = get_logger()

# -----------------------------------------------------------------------
# 1. Architecture mapping
# -----------------------------------------------------------------------

register_model_arch(
    MultiModelKeys(
        'amphion_asr_1.7b',
        language_model='thinker.model',
        vision_tower='thinker.audio_tower',
        # proj1/proj2 live inside audio_tower, so no separate aligner key.
    ))


# -----------------------------------------------------------------------
# 2. Model loader
# -----------------------------------------------------------------------

class Qwen3ASRLoader(ModelLoader):
    """Load a Qwen3-ASR HuggingFace checkpoint.

    ``qwen_asr`` registers ``Qwen3ASRConfig`` / ``Qwen3ASRForConditionalGeneration``
    with HuggingFace's ``AutoConfig`` / ``AutoModel`` at import time.  We just need
    to import it first (with the sys.path guard to avoid the nagisa/k2 collision),
    then use ``AutoModel.from_pretrained`` as usual.

    Also handles ModelScope / HuggingFace Hub cache layouts where the actual model
    lives in a nested subdirectory (e.g. ``…/Qwen/Qwen3-ASR-1___7B/``).
    """

    @staticmethod
    def _import_qwen_asr() -> None:
        """Import qwen_asr to register Qwen3ASRConfig/Model with AutoConfig/AutoModel.

        Must be called before any AutoConfig.from_pretrained() or
        AutoModel.from_pretrained() call, because qwen3_asr is not built into
        the transformers library and requires the qwen_asr package to register it.

        Guards against the nagisa / k2 collision caused by AmphionASR's src/
        directory shadowing qwen_asr's internal 'import model'.

        Also patches Qwen3ASRForConditionalGeneration.get_input_embeddings /
        set_input_embeddings, which are not implemented by the top-level wrapper
        class but are required by the transformers library.
        """
        import sys
        from pathlib import Path

        # In source checkouts, remove only this repository's ./src while
        # importing qwen_asr so qwen_asr's internal top-level imports cannot
        # be shadowed by legacy source-layout packages.
        _this_file = Path(__file__).resolve()
        _source_root = _this_file.parents[3]
        _remove_path = str(_source_root) if _source_root.name == 'src' else None
        _saved = sys.path.copy()
        if _remove_path is not None:
            while _remove_path in sys.path:
                sys.path.remove(_remove_path)
        try:
            import qwen_asr as _qwen_asr  # noqa: F401
        finally:
            sys.path[:] = _saved

        # Patch missing methods on the top-level wrapper class.
        # Qwen3ASRForConditionalGeneration wraps Qwen3ASRThinkerForConditionalGeneration
        # (self.thinker) that implements these, but the wrapper itself does not.
        # ms-swift / PEFT / transformers all require these to be present on the
        # top-level model for training.
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRForConditionalGeneration as _Cls,
            Qwen3ASRAudioEncoder as _AudioCls,
        )
        # qwen-asr 0.0.6 exposes a stale conv1 accessor although its first
        # convolution is conv2d1. Swift also disables input hooks on frozen
        # towers where no hook was installed in the first place.
        def _get_audio_input_embeddings(self):
            return self.conv2d1

        def _set_audio_input_embeddings(self, value):
            self.conv2d1 = value

        def _disable_audio_input_require_grads(self):
            hook = getattr(self, '_require_grads_hook', None)
            if hook is not None:
                hook.remove()
                del self._require_grads_hook

        _AudioCls.get_input_embeddings = _get_audio_input_embeddings
        _AudioCls.set_input_embeddings = _set_audio_input_embeddings
        _AudioCls.disable_input_require_grads = _disable_audio_input_require_grads
        if not getattr(_Cls, '_swift_patched', False):
            def _get_input_embeddings(self):
                return self.thinker.get_input_embeddings()

            def _set_input_embeddings(self, value):
                return self.thinker.set_input_embeddings(value)

            def _forward(self, *args, **kwargs):
                return self.thinker.forward(*args, **kwargs)

            _Cls.get_input_embeddings = _get_input_embeddings
            _Cls.set_input_embeddings = _set_input_embeddings
            _Cls.forward = _forward
            _Cls._swift_patched = True

    def get_config(self, model_dir: str):
        # AutoConfig.from_pretrained() is called here by ms-swift.
        # qwen3_asr must be registered in AutoConfig BEFORE this call.
        self._import_qwen_asr()
        return super().get_config(model_dir)

    def get_model(self, model_dir: str, *args, **kwargs) -> PreTrainedModel:
        from pathlib import Path
        from transformers import AutoModel

        # ------------------------------------------------------------------
        # 1. Resolve ModelScope / HF Hub cache nested path.
        #    e.g. /models/Qwen3-ASR-1.7B  →  /models/Qwen3-ASR-1.7B/Qwen/Qwen3-ASR-1___7B
        # ------------------------------------------------------------------
        model_path = Path(model_dir)
        if not (model_path / 'config.json').exists():
            found = next(
                (p.parent for p in sorted(model_path.rglob('config.json'))
                 if '.lock' not in str(p)),
                None,
            )
            if found:
                logger.info(f'Qwen3ASRLoader: resolved nested model dir: {model_dir!r} -> {found}')
                model_dir = str(found)

        # ------------------------------------------------------------------
        # 2. Import qwen_asr (idempotent; get_config already called it).
        # ------------------------------------------------------------------
        self._import_qwen_asr()

        # ------------------------------------------------------------------
        # 3. Load via AutoModel (registered by qwen_asr at import time).
        # ------------------------------------------------------------------
        self.auto_model_cls = self.auto_model_cls or AutoModel
        return super().get_model(model_dir, *args, **kwargs)


# -----------------------------------------------------------------------
# 3. Template
# -----------------------------------------------------------------------

class Qwen3ASRTemplate(Template):
    """Template for Qwen3-ASR-1.7B audio-to-text model.

    Each audio is represented by the placeholder token ``<|audio_pad|>``,
    which ``replace_tag`` wraps into ``<|audio_start|><|audio_pad|><|audio_end|>``.
    Audio features are extracted with WhisperFeatureExtractor (128 mel bins)
    and passed to the model as ``input_features`` / ``feature_lens``.
    """

    placeholder_tokens = ['<|audio_pad|>']

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
                fe = WhisperFeatureExtractor(
                    feature_size=128, sampling_rate=16000
                )
            self._feature_extractor = fe
        self.sampling_rate = get_env_args('sampling_rate', int, fe.sampling_rate)
        # hop_length / max_length used by _get_n_audio_tokens() in replace_tag
        self._hop_length = getattr(fe, 'hop_length', 160)
        self._max_length = getattr(fe, 'nb_max_frames', 3000)

    @property
    def feature_extractor(self):
        return getattr(self, '_feature_extractor', None) or self.processor.feature_extractor

    def replace_tag(
        self,
        media_type: Literal['image', 'video', 'audio'],
        index: int,
        inputs,
    ) -> list:
        assert media_type == 'audio'
        wavs = getattr(inputs, '_audio_llm_waveforms', None)
        wav = (wavs[index] if wavs is not None else
               load_audio(inputs.audios[index], sampling_rate=self.sampling_rate))
        # Expand <|audio_pad|> to exactly the number of tokens the audio encoder
        # will produce, so that masked_scatter in forward() can replace every
        # placeholder position with a real audio frame embedding.
        # (One placeholder → only 1 frame used → audio ignored → loss stuck at ~3.)
        n_tokens = _get_n_audio_tokens(
            wav,
            hop_length=self._hop_length,
            max_length=self._max_length,
        )
        return ['<|audio_start|>'] + ['<|audio_pad|>'] * n_tokens + ['<|audio_end|>']

    def _encode(self, inputs) -> Dict[str, Any]:
        if not inputs.audios:
            return super()._encode(inputs)
        try:
            audios = load_batch(
                inputs.audios,
                load_func=partial(load_audio, sampling_rate=self.sampling_rate),
            )
        except Exception as e:
            raise ValueError(
                f'Qwen3ASRTemplate: load_audio failed for '
                f'{inputs.audios!r}: {e}') from e

        # replace_tag and feature extraction share this decode. Keep the
        # waveforms off extra_kwargs so they cannot leak into training rows.
        inputs._audio_llm_waveforms = audios
        try:
            encoded = super()._encode(inputs)
        finally:
            del inputs._audio_llm_waveforms

        audio_inputs = self.feature_extractor(
            audios,
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # feature_attention_mask: (B, T) bool mask — 1=valid frame, 0=padding.
        # This is what Qwen3ASRForConditionalGeneration.forward() expects.
        feature_attention_mask = audio_inputs['attention_mask']

        # Guard against zero-length audio (silently truncated files).
        feat_lens = feature_attention_mask.sum(dim=-1)
        if (feat_lens <= 0).any():
            raise ValueError(
                f'Qwen3ASRTemplate: zero-length audio '
                f'(feat_lens={feat_lens.tolist()}) for audios={inputs.audios!r}')

        # Keep input_features in the native (B, n_mels, T) format from the feature extractor.
        # Qwen3ASRThinkerForConditionalGeneration.get_audio_features() iterates over the batch
        # and slices each audio as input_feature[:, :feature_len] — i.e., (n_mels, T) → ✓
        # Do NOT transpose here (Amphion-4B transposes because its encoder handles (B,T,F)
        # internally, but Qwen3-ASR's encoder expects (n_mels, T) directly).
        max_frames = int(feat_lens.max().item())
        encoded['input_features'] = (
            augment_features(
                audio_inputs['input_features'][..., :max_frames], feat_lens,
                (getattr(inputs, 'extra_kwargs', None) or {}).get('audio_augmentation'),
            ).contiguous()
        )
        encoded['feature_attention_mask'] = (
            feature_attention_mask[..., :max_frames].contiguous()
        )
        return encoded

    def _data_collator(
        self,
        batch: List[Dict[str, Any]],
        *,
        padding_to: Optional[int] = None,
    ) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        if any('catalog_task' in b for b in batch):
            res['catalog_task'] = torch.tensor([b['catalog_task'] for b in batch], dtype=torch.long)
        input_features = [
            b['input_features'] for b in batch
            if b.get('input_features') is not None
        ]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch
            if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            max_frames = max(features.shape[-1] for features in input_features)
            input_features = [
                F.pad(features, (0, max_frames - features.shape[-1]))
                for features in input_features
            ]
            feature_attention_mask = [
                F.pad(mask, (0, max_frames - mask.shape[-1]))
                for mask in feature_attention_mask
            ]
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)
        return res


# -----------------------------------------------------------------------
# 4. Registration
# -----------------------------------------------------------------------

register_model(
    ModelMeta(
        'amphion_asr_1.7b',
        [
            ModelGroup([
                Model('', 'qwen3-asr'),
            ], 'amphion_asr_1.7b'),
        ],
        Qwen3ASRLoader,
        template='amphion_asr_1.7b',
        model_arch='amphion_asr_1.7b',
        architectures=['Qwen3ASRForConditionalGeneration'],
        is_multimodal=True,
        requires=['qwen_asr', 'librosa'],
        tags=['audio'],
    ))

register_template(
    TemplateMeta(
        'amphion_asr_1.7b',
        # System turn: all instruction text (enrollment notice, Language, Hotwords).
        # Matches the original chat_template.json which ALWAYS renders the system
        # turn and puts NO text in the user turn.
        prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
        # User turn: ONLY audio placeholder tokens — no text.
        # Runtime messages contain one placeholder per audio slot.
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        suffix=['<|im_end|>'],
        default_system='',   # empty system for basic ASR; data populates it
        template_cls=Qwen3ASRTemplate,
    ))


# -----------------------------------------------------------------------
# Convenience entry point
# -----------------------------------------------------------------------

def register_all():
    """No-op — registrations happen at import time via module-level calls."""
