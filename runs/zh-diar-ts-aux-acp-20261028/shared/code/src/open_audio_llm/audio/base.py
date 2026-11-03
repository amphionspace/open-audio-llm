"""Audio tower interface and simple built-in adapters."""

from __future__ import annotations

from typing import Union

import torch
from torch import nn

from open_audio_llm.registry import audio_tower_registry


class AudioTower(nn.Module):
    """Base class for audio encoders used by `AudioLLM`.

    Implementations must return `(hidden, hidden_lengths)` where hidden has
    shape `(B, T, D)`.
    """

    output_dim: int

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def get_output_lengths(
        input_lengths: Union[int, torch.Tensor],
    ) -> Union[int, torch.Tensor]:
        raise NotImplementedError


@audio_tower_registry.register("identity")
class IdentityAudioTower(AudioTower):
    """A dependency-free tower useful for smoke tests and feature passthrough."""

    def __init__(self, config):
        super().__init__()
        self.output_dim = int(getattr(config, "output_dim", getattr(config, "input_dim", 0)))

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected features with shape (B,T,F), got {features.shape}")
        return features, lengths.to(dtype=torch.long, device=features.device)

    @staticmethod
    def get_output_lengths(input_lengths):
        return input_lengths
