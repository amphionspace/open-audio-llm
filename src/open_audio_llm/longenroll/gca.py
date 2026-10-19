"""Gated cross-attention in the GCA layers of the Qwen3-ASR audio encoder
(after self-attention, before the FFN), and the ECAPA short path added to
every frame of the encoder input:

    x_0 = conv(mel) + tanh(s) * spk_active * SpkProj(ecapa)
    h_l = h_l + tanh(g_l) * long_active * GCA_l(h_l, memory32)      l in GCA layers

Gates start at 0.01, so at init the encoder is the original Qwen3-ASR encoder.

``output_norm`` normalizes each frame of the GCA output to unit RMS (no affine) before the gate.
Without it the gate and the value/output projections share one scale: under Adam the gain moved
into a few amplified directions of W_out W_o W_v (top singular value 0.7 -> 120-230 in 90k steps)
while the gates shrank to ~1e-3, where a 1e-4 Adam step changes them by ~10% per step.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
from torch import Tensor, nn


def normalize_layers(indices: Iterable[int], num_layers: int) -> tuple[int, ...]:
    layers = set()
    for index in indices:
        index = int(index)
        if index < 0:
            index += num_layers
        if not 0 <= index < num_layers:
            raise ValueError(f"GCA layer out of range: {index}/{num_layers}")
        layers.add(index)
    if not layers:
        raise ValueError("need at least one GCA layer")
    return tuple(sorted(layers))


class EncoderGCA(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, output_norm: bool = False):
        super().__init__()
        self.query_norm = nn.RMSNorm(hidden_size)
        self.memory_norm = nn.RMSNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Parameter-free: the gate stays the only scale of the injected residual.
        self.output_norm = nn.RMSNorm(hidden_size, elementwise_affine=False) if output_norm else None

    def forward(self, hidden: Tensor, memory: Tensor, token_mask: Optional[Tensor] = None) -> Tensor:
        """hidden [B, L, H], memory [B, 32, H], token_mask [B, 32] (True = visible)."""
        memory = self.memory_norm(memory)
        out, _ = self.attention(
            self.query_norm(hidden), memory, memory,
            key_padding_mask=(~token_mask) if token_mask is not None else None,
            need_weights=False,
        )
        out = self.out_proj(out)
        return self.output_norm(out) if self.output_norm is not None else out


class SpeakerPath(nn.Module):
    """ECAPA 192-d -> MLP -> 1024-d bias, added to every frame of the encoder input."""

    def __init__(self, ecapa_dim: int, hidden_size: int, gate_init: float):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(ecapa_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, ecapa: Tensor) -> Tensor:
        """ecapa [B, 192], L2-normalized (rescaled here to unit per-dim RMS) -> [B, H]."""
        return torch.tanh(self.gate) * self.proj(ecapa * math.sqrt(ecapa.size(-1)))
