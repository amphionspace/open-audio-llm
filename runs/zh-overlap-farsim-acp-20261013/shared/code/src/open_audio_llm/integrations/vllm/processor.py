"""vLLM processor helpers."""

from __future__ import annotations


def estimate_audio_tokens(audio_tower, feature_lengths, connector_downsample: int = 1):
    tower_lengths = audio_tower.get_output_lengths(feature_lengths)
    return tower_lengths // max(int(connector_downsample), 1)
