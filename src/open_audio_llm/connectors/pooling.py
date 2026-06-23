"""Pooling connector extension point."""

from __future__ import annotations

import torch
from torch import nn

from .base import Connector
from open_audio_llm.registry import connector_registry


@connector_registry.register("mean_pool")
class MeanPoolConnector(Connector):
    def __init__(self, config):
        super().__init__()
        self.downsample_rate = 1
        self.proj = nn.Linear(int(config.input_dim), int(config.output_dim))

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor):
        pooled = []
        for i, length in enumerate(lengths.tolist()):
            pooled.append(hidden[i, : max(int(length), 1)].mean(dim=0, keepdim=True))
        return torch.stack(pooled, dim=0).squeeze(1).unsqueeze(1), lengths.new_ones(lengths.shape)
