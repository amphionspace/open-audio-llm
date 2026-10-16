"""Long Enrollment Encoder: enrollment of any length -> 32 tokens [B, 32, H].

Stage 1 (per 30 s window, ported from TargetConditionedRVQEncoder):
  32 shared windowed queries, single-head attention whose weights come only
  from the coarse key feature. The same weights pool every value layer.
  Key  = W_k(pool(coarse))                 [32, H]
  Value= W_v(concat(pool(layer_l) for l))  [32, H]
  Windows are concatenated in time -> variable-length KV memory (~32 per 30 s).
  The windows are the same 30 s windows the frozen enrollment encoder attends in.

Stage 2:
  8 global queries  : cross-attn over memory (K = V = Value). Enrollment only,
                      cacheable per speaker.
  24 local queries  : (1) cross-attn over the mixture's encoder states at
                      depth ``mix_condition_layer``; (2) that output queries
                      the whole memory with K = Key (coarse), V = Value (full).

Continuous features replace the 16 RVQ code embeddings of the TTS version.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn


def window_bounds(length: int, window: int, min_tail: int) -> list[tuple[int, int]]:
    """[start, end) windows of ``window`` frames over ``length`` frames. A tail
    shorter than ``min_tail`` is merged into the previous window; a sequence no
    longer than ``window`` is a single window (never padded)."""
    if length <= window:
        return [(0, length)]
    bounds = [(start, min(start + window, length)) for start in range(0, length, window)]
    if bounds[-1][1] - bounds[-1][0] < min_tail:
        tail = bounds.pop()
        bounds[-1] = (bounds[-1][0], tail[1])
    return bounds


class CrossAttentionBlock(nn.Module):
    """Pre-Norm cross-attention + FFN (same as the TTS reference)."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.RMSNorm(hidden_size)
        self.key_norm = nn.RMSNorm(hidden_size)
        self.value_norm = nn.RMSNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, key_mask: Optional[Tensor],
                return_attention: bool = False):
        attended, weights = self.attention(
            self.query_norm(queries),
            self.key_norm(keys),
            self.value_norm(values),
            key_padding_mask=(~key_mask.bool()) if key_mask is not None else None,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        hidden = queries + attended
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden, weights


@dataclass
class EnrollMemory:
    """Variable-length enrollment KV. Depends only on enrollment audio."""

    keys: Tensor    # [B, M, H]
    values: Tensor  # [B, M, H]
    mask: Tensor    # [B, M] bool

    def index_select(self, index: Tensor) -> "EnrollMemory":
        return EnrollMemory(self.keys.index_select(0, index), self.values.index_select(0, index),
                            self.mask.index_select(0, index))


class LongEnrollmentEncoder(nn.Module):
    def __init__(self, hidden_size: int, cfg):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.compressor_dim = int(cfg.compressor_dim)
        self.chunk_queries = int(cfg.chunk_queries)
        self.local_window_slots = float(cfg.local_window_slots)
        self.global_count = int(cfg.global_queries)
        self.local_count = int(cfg.local_queries)
        self.key_layer = int(cfg.key_layer)
        self.value_layers = [int(x) for x in cfg.value_layers]
        if self.global_count + self.local_count != 32:
            raise ValueError("global_queries + local_queries must be 32")
        if self.local_window_slots <= 0:
            raise ValueError("local_window_slots must be > 0")
        H = self.hidden_size

        # Encoder hidden norms differ a lot across depths; normalize per depth.
        self.depth_norms = nn.ModuleDict({
            str(d): nn.RMSNorm(H) for d in sorted(set(self.value_layers) | {self.key_layer})
        })
        self.chunk_query = nn.Parameter(torch.randn(self.chunk_queries, self.compressor_dim) * 0.02)
        self.chunk_query_norm = nn.RMSNorm(self.compressor_dim)
        self.raw_key_projection = nn.Linear(H, self.compressor_dim, bias=False)
        self.raw_key_norm = nn.RMSNorm(self.compressor_dim)
        self.memory_key_projection = nn.Linear(H, H, bias=False)
        self.memory_value_projection = nn.Linear(H * len(self.value_layers), H, bias=False)

        heads, dropout = int(cfg.num_heads), float(cfg.dropout)
        self.global_queries = nn.Parameter(torch.randn(self.global_count, H) * 0.02)
        self.local_queries = nn.Parameter(torch.randn(self.local_count, H) * 0.02)
        self.global_block = CrossAttentionBlock(H, heads, dropout)
        self.mix_block = CrossAttentionBlock(H, heads, dropout)
        self.local_block = CrossAttentionBlock(H, heads, dropout)

    @property
    def feature_depths(self) -> list[int]:
        return sorted(set(self.value_layers) | {self.key_layer})

    # ---------------------------------------------------------- stage 1
    def _local_mask(self, lengths: Tensor, max_frames: int) -> Tensor:
        """Query q sees frames whose normalized position is within
        ``local_window_slots / 32`` of its center (q + 0.5) / 32."""
        frame = torch.arange(max_frames, device=lengths.device, dtype=torch.float32)
        slot = torch.arange(self.chunk_queries, device=lengths.device, dtype=torch.float32)
        normalized = (frame[None, None, :] + 0.5) / lengths[:, None, None].float()
        centers = (slot[None, :, None] + 0.5) / self.chunk_queries
        radius = self.local_window_slots / self.chunk_queries
        valid = frame[None, None, :] < lengths[:, None, None]
        mask = ((normalized - centers).abs() <= radius) & valid
        missing = ~mask.any(dim=-1)
        if missing.any():  # very short chunk: give each query its nearest frame
            nearest = (normalized - centers).abs().masked_fill(~valid, float("inf")).argmin(dim=-1)
            mask.scatter_(2, nearest.unsqueeze(-1), True)
        return mask

    def compress_chunks(self, depth_states: dict[int, Tensor], lengths: Tensor,
                        return_attention: bool = False):
        """depth_states[d]: [N, T, H] encoder states of N chunks. -> keys, values [N, 32, H]."""
        coarse = self.depth_norms[str(self.key_layer)](depth_states[self.key_layer].float())
        raw_keys = self.raw_key_norm(self.raw_key_projection(coarse))
        queries = self.chunk_query_norm(self.chunk_query)[None].expand(coarse.size(0), -1, -1)
        logits = torch.matmul(queries, raw_keys.transpose(1, 2)) / math.sqrt(self.compressor_dim)
        mask = self._local_mask(lengths, coarse.size(1))
        attention = torch.softmax(logits.float().masked_fill(~mask, float("-inf")), dim=-1)
        pooled_key = torch.matmul(attention, coarse)
        pooled_values = [
            torch.matmul(attention, self.depth_norms[str(d)](depth_states[d].float()))
            for d in self.value_layers
        ]
        keys = self.memory_key_projection(pooled_key)
        values = self.memory_value_projection(torch.cat(pooled_values, dim=-1))
        return (keys, values, attention) if return_attention else (keys, values)

    @staticmethod
    def gather_memory(chunk_keys: Tensor, chunk_values: Tensor, owners: Sequence[int], batch_size: int) -> EnrollMemory:
        """Chunks [N, 32, H] -> per-owner time-ordered memory, padded."""
        owner_t = torch.as_tensor(list(owners), device=chunk_keys.device)
        key_seqs, value_seqs = [], []
        for owner in range(batch_size):
            idx = torch.nonzero(owner_t.eq(owner), as_tuple=False).flatten()
            if idx.numel() == 0:
                raise ValueError(f"enrollment {owner} has no chunks")
            key_seqs.append(chunk_keys.index_select(0, idx).flatten(0, 1))
            value_seqs.append(chunk_values.index_select(0, idx).flatten(0, 1))
        lengths = torch.tensor([k.size(0) for k in key_seqs], device=chunk_keys.device)
        keys = nn.utils.rnn.pad_sequence(key_seqs, batch_first=True)
        values = nn.utils.rnn.pad_sequence(value_seqs, batch_first=True)
        mask = torch.arange(keys.size(1), device=keys.device)[None] < lengths[:, None]
        return EnrollMemory(keys, values, mask)

    # ---------------------------------------------------------- stage 2
    def global_tokens(self, memory: EnrollMemory, return_attention: bool = False):
        queries = self.global_queries[None].expand(memory.values.size(0), -1, -1)
        tokens, weights = self.global_block(queries, memory.values, memory.values, memory.mask, return_attention)
        return (tokens, weights) if return_attention else tokens

    def local_tokens(self, memory: EnrollMemory, mix_states: Tensor, mix_mask: Tensor,
                     return_attention: bool = False):
        queries = self.local_queries[None].expand(memory.keys.size(0), -1, -1)
        mix_states = mix_states.float()
        queries, _ = self.mix_block(queries, mix_states, mix_states, mix_mask)
        tokens, weights = self.local_block(queries, memory.keys, memory.values, memory.mask, return_attention)
        return (tokens, weights) if return_attention else tokens

    def forward(self, memory: EnrollMemory, mix_states: Tensor, mix_mask: Tensor,
                global_tokens: Optional[Tensor] = None) -> Tensor:
        if global_tokens is None:
            global_tokens = self.global_tokens(memory)
        local = self.local_tokens(memory, mix_states, mix_mask)
        return torch.cat([global_tokens.to(local.dtype), local], dim=1)
