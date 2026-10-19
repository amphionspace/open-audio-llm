"""Run the Qwen3-ASR text decoder on one packed row per microbatch, without padding."""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Per-token inputs ms-swift templates may add; packing would leave them misaligned.
UNPACKED_TOKEN_KEYS = {"position_ids", "text_position_ids", "loss_scale", "token_type_ids",
                       "mm_token_type_ids", "channel"}


def pack_text(batch):
    """Concatenate the unpadded text rows of a collated batch; audio inputs are untouched.

    transformers runs FlashAttention-2 varlen when ``attention_mask`` is absent and
    ``cu_seq_lens_*`` are given. Passing them here avoids rebuilding them from
    ``position_ids`` in every layer, which would wait for the GPU each time.
    """
    unknown = sorted(UNPACKED_TOKEN_KEYS & batch.keys())
    if unknown:
        raise ValueError(f"Cannot pack per-token inputs {unknown}")
    mask = batch.pop("attention_mask").bool()
    lengths = mask.sum(-1)
    starts = F.pad(lengths.cumsum(0), (1, 0))
    labels = batch["labels"][mask]
    # A padded row never uses its first label (the shift drops it); in one packed row the
    # previous sample's last token would predict it.
    labels[starts[:-1]] = -100
    batch["input_ids"] = batch["input_ids"][mask][None]
    batch["labels"] = labels[None]
    batch["position_ids"] = (mask.cumsum(-1) - 1)[mask][None]
    batch["cu_seq_lens_q"] = batch["cu_seq_lens_k"] = starts.to(torch.int32)
    batch["max_length_q"] = batch["max_length_k"] = int(lengths.max())
    return batch


class PackedTextCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, rows):
        return pack_text(self.collator(rows))


def enable_packed_text(model):
    """Switch only the thinker's text decoder to FlashAttention-2; the audio tower keeps its own."""
    from transformers.utils import is_flash_attn_2_available

    if not is_flash_attn_2_available():
        raise ImportError("packed_text needs flash-attn 2 for varlen attention")
    config = model.thinker.model.config
    if config is model.thinker.audio_tower.config:
        raise ValueError("The text decoder shares its config with the audio tower")
    previous = config._attn_implementation
    config._attn_implementation = "flash_attention_2"

    def restore():
        config._attn_implementation = previous

    return restore
