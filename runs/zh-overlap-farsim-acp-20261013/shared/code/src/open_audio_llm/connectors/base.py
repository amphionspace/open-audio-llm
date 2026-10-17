"""Connector interface."""

from __future__ import annotations

import torch
from torch import nn


class Connector(nn.Module):
    downsample_rate: int = 1

    def forward(
        self,
        hidden: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
