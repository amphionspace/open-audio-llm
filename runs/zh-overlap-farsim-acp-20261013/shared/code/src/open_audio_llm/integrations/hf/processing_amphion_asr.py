"""Processor for AmphionASR.

Combines audio feature extraction with an ``AutoTokenizer``
(Qwen-family) and provides helpers for building the chat-template
prompt that the model expects.

Feature extraction strategy depends on the encoder type:
- **Qwen3 / OmniMoe** → ``WhisperFeatureExtractor`` (Whisper mel scale)
- **Zipformer** → ``torchaudio.compliance.kaldi.fbank`` (Kaldi mel scale)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer, WhisperFeatureExtractor
from transformers.processing_utils import ProcessorMixin

from .constants import (
    DEFAULT_SPEECH_TOKEN,
    END_SPEECH_TOKEN,
    END_TEXT_TOKEN,
    SPECIAL_TOKENS,
    START_SPEECH_TOKEN,
    START_TEXT_TOKEN,
)


# ======================================================================
# Kaldi-compatible Fbank (matches lhotse FbankConfig defaults)
# ======================================================================

def kaldi_fbank_extract(
    audio: Union[np.ndarray, torch.Tensor],
    num_mel_bins: int = 80,
    sample_frequency: float = 16000.0,
) -> torch.Tensor:
    """Extract Kaldi-compatible log-mel filterbank features.

    Uses ``torchaudio.compliance.kaldi.fbank`` with the same defaults as
    ``lhotse.features.FbankConfig(num_mel_bins=80)`` to guarantee
    feature-level compatibility with Zipformer training.

    Parameters
    ----------
    audio : 1-D waveform (numpy or torch)
    num_mel_bins : number of mel bins
    sample_frequency : expected sample rate

    Returns
    -------
    features : ``(T, num_mel_bins)`` float32 tensor (already log-mel)
    """
    import torchaudio

    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # (1, samples)

    features = torchaudio.compliance.kaldi.fbank(
        audio,
        sample_frequency=sample_frequency,
        frame_length=25.0,
        frame_shift=10.0,
        num_mel_bins=num_mel_bins,
        preemphasis_coefficient=0.97,
        window_type="povey",
        dither=0.0,
        snip_edges=False,
        energy_floor=1e-10,
        raw_energy=True,
        use_energy=False,
        low_freq=20.0,
        high_freq=-400.0,
        remove_dc_offset=True,
    )
    return features  # (T, num_mel_bins)

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content']}}"
    "{% if loop.last %}{{ '<|im_end|>'}}"
    "{% else %}{{ '<|im_end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
)

TASK_PROMPTS = {
    "asr": "Transcribe the following audio:{speech}",
    "asr_en": "Transcribe the following English audio:{speech}",
    "asr_zh": "Transcribe the following Chinese audio:{speech}",
    "asr_hotwords": "Hotwords:{hotwords}\nTranscribe the following audio:{speech}",
    "ser": "Classify the emotion of the following audio:{speech}",
    "sec": "Describe the emotion of the following audio:{speech}",
}


class AmphionASRProcessor(ProcessorMixin):
    """Processor that bundles audio feature extraction and tokenisation.

    Usage::

        processor = AmphionASRProcessor.from_pretrained("path/to/model")
        inputs = processor(audio=waveform, text="Transcribe", sr=16000)
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor: WhisperFeatureExtractor = None,
        tokenizer: AutoTokenizer = None,
        feature_extractor_type: str = "whisper",
    ):
        if feature_extractor is None:
            feature_extractor = WhisperFeatureExtractor(
                sampling_rate=16000,
            )
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        super().__init__(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.feature_extractor_type = feature_extractor_type
        self._ensure_special_tokens()

    def _ensure_special_tokens(self):
        """Add Amphion special tokens to the tokenizer if missing."""
        existing = set(self.tokenizer.additional_special_tokens or [])
        to_add = [t for t in SPECIAL_TOKENS if t not in existing]
        if to_add:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": list(existing | set(to_add))}
            )

    @staticmethod
    def build_speech_placeholder() -> str:
        return f"{START_SPEECH_TOKEN}{DEFAULT_SPEECH_TOKEN}{END_SPEECH_TOKEN}"

    def get_task_prompt(self, task: str = "asr", **kwargs) -> str:
        """Build the user-role prompt for a given task."""
        if task not in TASK_PROMPTS:
            raise ValueError(f"Unknown task {task!r}. Available: {sorted(TASK_PROMPTS)}")
        return TASK_PROMPTS[task].format(
            speech=self.build_speech_placeholder(), **kwargs
        )

    def build_chat_messages(
        self,
        task: str = "asr",
        answer: Optional[str] = None,
        **kwargs,
    ) -> list:
        """Build a list of chat messages for the model.

        Parameters
        ----------
        task : str
            Task key (``"asr"``, ``"asr_en"``, etc.).
        answer : str, optional
            If provided, an assistant turn is appended (for training).
        """
        user_content = self.get_task_prompt(task, **kwargs)
        messages = [{"role": "user", "content": user_content}]
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
        return messages

    def __call__(
        self,
        audio: Union[np.ndarray, List[np.ndarray], torch.Tensor] = None,
        text: Union[str, List[str]] = None,
        task: str = "asr",
        sampling_rate: int = 16000,
        return_tensors: str = "pt",
        padding: bool = True,
        answer: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Process audio + text into model-ready tensors.

        Parameters
        ----------
        audio : array-like
            Raw waveform(s) at ``sampling_rate``.
        text : str or list[str], optional
            If provided, used directly as the text prompt.  Otherwise
            ``task`` is used to build the prompt automatically.
        task : str
            Task key used when ``text`` is *not* provided.
        answer : str, optional
            Assistant answer (for training data preparation).
        """
        result = {}

        # --- Audio features ---------------------------------------------------
        if audio is not None:
            if self.feature_extractor_type == "kaldi_fbank":
                input_features, feature_lens = self._extract_kaldi_fbank(
                    audio, sampling_rate
                )
            else:
                input_features, feature_lens = self._extract_whisper(
                    audio, sampling_rate, return_tensors, padding
                )
            result["input_features"] = input_features  # (B, T, F)
            result["feature_lens"] = feature_lens

        # --- Text tokenisation ------------------------------------------------
        if text is None:
            messages = self.build_chat_messages(task=task, answer=answer, **kwargs)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=(answer is None),
                chat_template=CHAT_TEMPLATE,
            )

        if isinstance(text, str):
            text = [text]

        tok_out = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding="longest",
            truncation=True,
        )
        result["input_ids"] = tok_out["input_ids"]
        result["attention_mask"] = tok_out["attention_mask"]

        return result

    # --- Feature extraction backends -----------------------------------------

    def _extract_whisper(
        self, audio, sampling_rate, return_tensors, padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_out = self.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors=return_tensors,
            padding=padding,
            return_attention_mask=True,
        )
        input_features = feat_out["input_features"]  # (B, n_mels, T)
        attn_mask = feat_out["attention_mask"]  # (B, T)
        feature_lens = attn_mask.sum(dim=-1).long()
        return input_features.transpose(1, 2), feature_lens  # (B, T, F)

    def _extract_kaldi_fbank(
        self, audio, sampling_rate
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract Kaldi-compatible fbank for one or more waveforms."""
        if isinstance(audio, np.ndarray) and audio.ndim == 1:
            audios = [audio]
        elif isinstance(audio, list):
            audios = audio
        elif isinstance(audio, torch.Tensor) and audio.dim() == 1:
            audios = [audio.numpy()]
        else:
            audios = [audio]

        feats_list = []
        for wav in audios:
            f = kaldi_fbank_extract(wav, sample_frequency=float(sampling_rate))
            feats_list.append(f)

        lens = torch.tensor([f.shape[0] for f in feats_list], dtype=torch.long)
        max_len = int(lens.max().item())
        n_mels = feats_list[0].shape[1]

        padded = torch.zeros(len(feats_list), max_len, n_mels)
        for i, f in enumerate(feats_list):
            padded[i, : f.shape[0], :] = f

        return padded, lens

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "input_features", "feature_lens"]
