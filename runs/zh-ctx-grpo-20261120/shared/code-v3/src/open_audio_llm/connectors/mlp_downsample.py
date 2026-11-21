"""MLP connector with frame concatenation downsampling."""

from __future__ import annotations

import torch
from torch import nn

from .base import Connector
from open_audio_llm.registry import connector_registry


class SwooshR(nn.Module):
    """Pure PyTorch SwooshR approximation used to avoid k2 as a hard dependency."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.exp(x - 1.0)) - 0.08 * x - 0.035


@connector_registry.register("mlp_downsample")
class MLPDownsampleConnector(Connector):
    def __init__(self, config):
        super().__init__()
        self.downsample_rate = int(getattr(config, "downsample_rate", 1))
        input_dim = int(getattr(config, "input_dim"))
        output_dim = int(getattr(config, "output_dim"))
        dropout = float(getattr(config, "dropout", 0.1))
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim * self.downsample_rate, output_dim),
            SwooshR(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor):
        if self.downsample_rate > 1:
            bsz, seq_len, dim = hidden.shape
            usable = seq_len // self.downsample_rate * self.downsample_rate
            hidden = hidden[:, :usable, :]
            hidden = hidden.reshape(bsz, usable // self.downsample_rate, dim * self.downsample_rate)
        return self.proj(hidden), lengths.to(torch.long) // self.downsample_rate
