"""Qwen3-Omni audio tower adapter."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from .base import AudioTower
from open_audio_llm.registry import audio_tower_registry


@audio_tower_registry.register("qwen3omni")
@audio_tower_registry.register("qwen3omni_captioner")
class Qwen3OmniAudioTower(AudioTower):
    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoderConfig,
        )
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
        )

        cfg_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        cfg_dict.pop("type", None)
        cfg = Qwen3OmniMoeAudioEncoderConfig(**cfg_dict)
        self.encoder = Qwen3OmniMoeAudioEncoder(cfg)
        self.output_dim = int(getattr(cfg, "output_dim", getattr(cfg, "d_model", 0)))
        self._strip_projection()

    def _strip_projection(self):
        for attr in ("proj1", "act", "proj2"):
            if hasattr(self.encoder, attr):
                setattr(self.encoder, attr, torch.nn.Identity())

    def forward(self, features: torch.Tensor, lengths: torch.Tensor):
        seqs: list[torch.Tensor] = []
        lens: list[int] = []
        for b in range(features.shape[0]):
            t = min(max(int(lengths[b].item()), 0), features.shape[1])
            feat_ft = features[b, :t].transpose(0, 1).contiguous()
            out = self.encoder(input_features=feat_ft, feature_lens=lengths[b : b + 1])
            hidden = out.last_hidden_state
            seqs.append(hidden)
            lens.append(hidden.shape[0])
        return (
            pad_sequence(seqs, batch_first=True, padding_value=0.0),
            torch.tensor(lens, device=features.device, dtype=torch.long),
        )

    @staticmethod
    def get_output_lengths(input_lengths):
        leave = input_lengths % 100
        feat = (leave - 1) // 2 + 1
        return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
