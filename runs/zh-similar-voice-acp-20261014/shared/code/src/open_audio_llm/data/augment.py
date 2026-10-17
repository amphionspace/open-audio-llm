"""On-the-fly augmentation hooks.

The functions are intentionally small and optional. Projects can replace them
with Lhotse or torchaudio-backed implementations without changing dataset
schema.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentConfig:
    speed_prob: float = 0.0
    rir_prob: float = 0.0
    noise_prob: float = 0.0
    spec_aug_prob: float = 0.0
    speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1)
    noise_snr_db: tuple[float, float] = (5.0, 20.0)
    # Datasets already recorded in the target noise condition (e.g. far-field meetings).
    noise_exclude_datasets: tuple[str, ...] = ()
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


def augment_waveform(
    audio, sampling_rate, rng, config, noise_loader=None, rir_loader=None
):
    """Apply waveform transforms in memory using only the sample's RNG."""
    from fractions import Fraction

    from scipy.signal import fftconvolve, resample_poly

    audio = np.asarray(audio, dtype=np.float32).copy()
    if rng.random() < config.speed_prob:
        factor = rng.choice(config.speed_factors)
        ratio = Fraction(1 / factor).limit_denominator(1000)
        audio = resample_poly(audio, ratio.numerator, ratio.denominator)
    if rng.random() < config.rir_prob:
        if rir_loader is None:
            raise ValueError("rir_prob requires a Catalog RIR source")
        impulse = rir_loader(rng, sampling_rate)
        energy = np.sqrt(np.sum(impulse**2))
        if energy > 0:
            audio = fftconvolve(audio, impulse / energy)[: len(audio)]
    if rng.random() < config.noise_prob:
        if noise_loader is None:
            raise ValueError("noise_prob requires a Catalog noise source")
        noise = noise_loader(rng, sampling_rate)
        if len(noise) < len(audio):
            noise = np.tile(noise, (len(audio) + len(noise) - 1) // len(noise))
        start = rng.randrange(len(noise) - len(audio) + 1)
        noise = noise[start : start + len(audio)]
        signal_power, noise_power = np.mean(audio**2), np.mean(noise**2)
        if noise_power > 0:
            snr = rng.uniform(*config.noise_snr_db)
            audio = audio + noise * np.sqrt(
                signal_power / (noise_power * 10 ** (snr / 10))
            )
    return np.asarray(audio, dtype=np.float32)


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
