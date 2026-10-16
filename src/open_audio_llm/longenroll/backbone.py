"""Qwen3-ASR-1.7B pieces used by the long-enrollment TS-ASR model (AmphionASR v5).

The audio tower is re-driven layer by layer on a padded batch so that GCA can
sit between self-attention and FFN, and so that the encoder runs once.
Per-utterance attention is global (same as HF SDPA training, which ignores
``cu_seqlens``); padding is masked out.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

AUDIO_SAMPLE_RATE = 16000
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"
AUDIO_PAD = "<|audio_pad|>"
NO_SPEECH_ANSWER = "language None<asr_text>"
AUDIO_TOKENS_PER_SEC = 13  # conv: 100 mel frames (1 s) -> 13 encoder frames


def _import_qwen_asr():
    from qwen_asr.core.transformers_backend import (
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRProcessor,
    )

    return Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor


def load_qwen3_asr(model_dir: str, dtype=torch.bfloat16, device: str | torch.device = "cpu"):
    """Returns (model, processor) in eval mode with requires_grad off; LongEnrollTSASR
    switches on the parts it trains."""
    model_cls, processor_cls = _import_qwen_asr()
    model = model_cls.from_pretrained(model_dir, dtype=dtype, attn_implementation="sdpa")
    processor = processor_cls.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, processor


def feat_extract_output_lengths(input_lengths):
    """Mel frames -> audio tokens (same formula as Qwen3-ASR, 100 frames -> 13)."""
    if torch.is_tensor(input_lengths):
        input_lengths = input_lengths.to(torch.long)
    leave = input_lengths % 100
    feat_lengths = (leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


class MelExtractor:
    """Whisper log-mel exactly as the Qwen3-ASR processor computes it for one clip."""

    def __init__(self, feature_extractor):
        self.fe = feature_extractor
        self.hop = int(getattr(feature_extractor, "hop_length", 160))

    def __call__(self, wav: np.ndarray) -> Tensor:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if wav.size < self.hop * 8:
            wav = np.pad(wav, (0, self.hop * 8 - wav.size))
        out = self.fe(
            [wav],
            sampling_rate=AUDIO_SAMPLE_RATE,
            return_attention_mask=True,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        n = int(out["attention_mask"][0].sum().item())
        return out["input_features"][0, :, :n].contiguous().float()  # [128, T]


class PromptBuilder:
    """Qwen3-ASR chat layout. Only the assistant answer + <|im_end|> is labeled."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.audio_pad_id = tokenizer.convert_tokens_to_ids(AUDIO_PAD)
        self.im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.im_end_id

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def build(self, n_audio_tokens: int, answer: Optional[str], system: str = "") -> tuple[list[int], list[int]]:
        head = self._ids(f"{IM_START}system\n{system}{IM_END}\n{IM_START}user\n{AUDIO_START}")
        tail = self._ids(f"{AUDIO_END}{IM_END}\n{IM_START}assistant\n")
        prompt = head + [self.audio_pad_id] * int(n_audio_tokens) + tail
        if answer is None:
            return prompt, [-100] * len(prompt)
        answer_ids = self._ids(answer) + [self.im_end_id]
        return prompt + answer_ids, [-100] * len(prompt) + answer_ids


# ---------------------------------------------------------------- audio tower

def conv_frontend(tower: nn.Module, mels: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Conv2d stack + per-1s-chunk sinusoidal PE (official layout).

    ``mels``: list of [128, T_i]. Returns padded hidden [B, L, D] and lengths [B].
    """
    device = tower.conv2d1.weight.device
    dtype = tower.conv2d1.weight.dtype
    feature_lens = torch.tensor([int(m.shape[-1]) for m in mels], device=device, dtype=torch.long)
    features = torch.cat([m.to(device=device, dtype=dtype) for m in mels], dim=-1)
    aftercnn_lens = feat_extract_output_lengths(feature_lens)
    chunk_size = int(tower.n_window) * 2
    chunk_num = torch.ceil(feature_lens / chunk_size).long()
    chunk_lengths = torch.full((int(chunk_num.sum()),), chunk_size, dtype=torch.long, device=device)
    tail = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail] = feature_lens % chunk_size
    chunk_lengths[chunk_lengths == 0] = chunk_size

    chunk_list = features.T.split(chunk_lengths.tolist(), dim=0)
    padded = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2).unsqueeze(1)
    lens_after = feat_extract_output_lengths(chunk_lengths)
    keep = torch.arange(int(lens_after.max()), device=device)[None] < lens_after[:, None]
    embeds = []
    for block in padded.split(int(getattr(tower, "conv_chunksize", 500) or 500), dim=0):
        x = F.gelu(tower.conv2d1(block))
        x = F.gelu(tower.conv2d2(x))
        x = F.gelu(tower.conv2d3(x))
        embeds.append(x)
    x = torch.cat(embeds, dim=0)
    b, c, f, t = x.shape
    x = tower.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
    pe = tower.positional_embedding.positional_embedding[: x.shape[1]].unsqueeze(0).to(x.dtype)
    packed = (x + pe)[keep]
    per_utt = packed.split(aftercnn_lens.tolist(), dim=0)
    return nn.utils.rnn.pad_sequence(per_utt, batch_first=True), aftercnn_lens


def length_mask(lengths: Tensor, max_len: Optional[int] = None) -> Tensor:
    max_len = int(lengths.max()) if max_len is None else int(max_len)
    return torch.arange(max_len, device=lengths.device)[None] < lengths[:, None]


GCAHook = Callable[[int, Tensor], Tensor]


def encoder_layer_forward(
    layer: nn.Module,
    hidden: Tensor,
    key_mask: Tensor,
    layer_index: int,
    post_attention_hook: Optional[GCAHook] = None,
) -> Tensor:
    """One Qwen3ASRAudioEncoderLayer on [B, L, D] with padding mask.

    ``post_attention_hook(layer_index, hidden)`` returns the residual added
    after self-attention and before the FFN (GCA + speaker bias).
    """
    attn = layer.self_attn
    bsz, length, dim = hidden.shape
    heads = attn.num_heads
    residual = hidden
    x = layer.self_attn_layer_norm(hidden)
    q = attn.q_proj(x).view(bsz, length, heads, -1).transpose(1, 2)
    k = attn.k_proj(x).view(bsz, length, heads, -1).transpose(1, 2)
    v = attn.v_proj(x).view(bsz, length, heads, -1).transpose(1, 2)
    mask = key_mask[:, None, None, :]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=attn.scaling)
    out = attn.out_proj(out.transpose(1, 2).reshape(bsz, length, dim))
    hidden = residual + out
    if post_attention_hook is not None:
        hidden = hidden + post_attention_hook(layer_index, hidden)
    residual = hidden
    x = layer.final_layer_norm(hidden)
    x = layer.fc2(layer.activation_fn(layer.fc1(x)))
    return residual + x


def run_encoder_layers(
    tower: nn.Module,
    hidden: Tensor,
    key_mask: Tensor,
    start: int,
    end: int,
    post_attention_hook: Optional[GCAHook] = None,
    collect: Sequence[int] = (),
) -> tuple[Tensor, dict[int, Tensor]]:
    """Run layers [start, end). ``collect`` holds depths d (output after d
    layers, 0 = conv output) whose hidden states are returned."""
    wanted = set(int(d) for d in collect)
    states: dict[int, Tensor] = {}
    if start in wanted:
        states[start] = hidden
    for index in range(start, end):
        hidden = encoder_layer_forward(tower.layers[index], hidden, key_mask, index, post_attention_hook)
        if index + 1 in wanted:
            states[index + 1] = hidden
    return hidden, states


def tower_head(tower: nn.Module, hidden: Tensor) -> Tensor:
    hidden = tower.ln_post(hidden)
    return tower.proj2(tower.act(tower.proj1(hidden)))
