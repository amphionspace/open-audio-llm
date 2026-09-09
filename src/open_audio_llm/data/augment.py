"""On-the-fly augmentation hooks.

The functions are intentionally small and optional. Projects can replace them
with Lhotse or torchaudio-backed implementations without changing dataset
schema.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable



@dataclass
class AugmentConfig:
    speed_prob: float = 0.0
    rir_prob: float = 0.0
    noise_prob: float = 0.0
    spec_aug_prob: float = 0.0
    speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1)
    noise_snr_db: tuple[float, float] = (5.0, 20.0)
    time_mask_width: int = 30
    frequency_mask_width: int = 16

    def __post_init__(self):
        for name in ("speed_prob", "rir_prob", "noise_prob", "spec_aug_prob"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not self.speed_factors or any(f <= 0 for f in self.speed_factors):
            raise ValueError("speed_factors must be positive")
        if len(self.noise_snr_db) != 2 or self.noise_snr_db[0] > self.noise_snr_db[1]:
            raise ValueError("noise_snr_db must be an ordered pair")
        if self.time_mask_width < 0 or self.frequency_mask_width < 0:
            raise ValueError("SpecAugment mask widths must be non-negative")


AugmentFn = Callable[[dict[str, Any], random.Random], dict[str, Any]]


def apply_augmentations(
    sample: dict[str, Any],
    rng: random.Random,
    hooks: list[AugmentFn] | None = None,
) -> dict[str, Any]:
    for hook in hooks or []:
        sample = hook(sample, rng)
    return sample


def augment_features(features, lengths, policy):
    """Mask valid mel frames (B,F,T) without changing lengths or padding."""
    if not policy:
        return features
    config = AugmentConfig(**policy["config"])
    rng = random.Random(policy["seed"])
    if rng.random() >= config.spec_aug_prob:
        return features
    features = features.clone()
    for index, length in enumerate(lengths):
        frames = min(int(length), features.shape[-1])
        for axis, size, maximum in (
            (-1, frames, config.time_mask_width),
            (-2, features.shape[-2], config.frequency_mask_width),
        ):
            width = rng.randint(0, min(size, maximum))
            start = rng.randint(0, size - width)
            if axis == -1:
                features[index, :, start : start + width] = 0
            else:
                features[index, start : start + width, :frames] = 0
    return features
