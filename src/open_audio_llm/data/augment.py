"""On-the-fly augmentation hooks.

The functions are intentionally small and optional. Projects can replace them
with Lhotse or torchaudio-backed implementations without changing dataset
schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable


@dataclass
class AugmentConfig:
    speed_prob: float = 0.0
    rir_prob: float = 0.0
    noise_prob: float = 0.0
    spec_aug_prob: float = 0.0


AugmentFn = Callable[[dict[str, Any], random.Random], dict[str, Any]]


def apply_augmentations(
    sample: dict[str, Any],
    rng: random.Random,
    hooks: list[AugmentFn] | None = None,
) -> dict[str, Any]:
    for hook in hooks or []:
        sample = hook(sample, rng)
    return sample
