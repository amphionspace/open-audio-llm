"""On-the-fly augmentation hooks.

The functions are intentionally small and optional. Projects can replace them
with Lhotse or torchaudio-backed implementations without changing dataset
schema.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

import numpy as np

from .ts_online import set_level

# Early reflections end ~50 ms after the direct path (the usual C50/D50 boundary).
LATE_REVERB_S = 0.05
# Direct-path window for DRR: +-2.5 ms around the peak is the common convention.
DIRECT_PATH_S = 0.0025
# Deployed and evaluated audio arrives as PCM16, which cannot exceed full scale.
FULL_SCALE = 1.0


@dataclass
class AugmentConfig:
    speed_prob: float = 0.0
    rir_prob: float = 0.0
    # Mutually exclusive with rir_prob: one convolution with a far-field RIR.
    far_rir_prob: float = 0.0
    noise_prob: float = 0.0
    spec_aug_prob: float = 0.0
    speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1)
    noise_snr_db: tuple[float, float] = (5.0, 20.0)
    # Noise level of the enrollment slot only; None uses noise_snr_db. An enrollment buried in noise
    # or babble no longer names a speaker, so keep it at or above the benchmark's worst one (10 dB).
    enrollment_noise_snr_db: tuple[float, float] | None = None
    # Datasets already recorded in the target noise condition (e.g. far-field meetings).
    noise_exclude_datasets: tuple[str, ...] = ()
    # Far-field: when the speech was reverberated, pass the noise through the same
    # room's late reverberation so it arrives diffuse instead of dry.
    reverb_noise: bool = False
    # Keep only RIR microphones whose DRR (dB) lies in this range; None keeps all.
    # [-6, -1] matched AliMeeting far-field recordings by ear (2026-10-14).
    rir_drr_db: tuple[float, float] | None = None
    # Whole-clip RMS (dBFS) drawn for reverberated clips, after noise; None keeps the
    # input level. Real far-field arrays record far quieter than close-talk speech.
    far_level_dbfs: tuple[float, float] | None = None
    # Share of the remaining clips given a whole-clip RMS (dBFS) drawn from level_dbfs, after
    # noise. Slots with a level of their own (online mixes, nonspeech windows, the enrollment)
    # keep it. Empty-target sources draw random levels, so text clips left at their recorded
    # level teach the model that quiet means nobody speaks (CHiME-4 empties, 2026-11-03).
    level_prob: float = 0.0
    level_dbfs: tuple[float, float] | None = None
    time_mask_width: int = 30
    frequency_mask_width: int = 16

    def __post_init__(self):
        for name in ("speed_prob", "rir_prob", "far_rir_prob", "noise_prob", "spec_aug_prob", "level_prob"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.rir_prob + self.far_rir_prob > 1:
            raise ValueError("rir_prob + far_rir_prob must be at most 1")
        if not self.speed_factors or any(f <= 0 for f in self.speed_factors):
            raise ValueError("speed_factors must be positive")
        if len(self.noise_snr_db) != 2 or self.noise_snr_db[0] > self.noise_snr_db[1]:
            raise ValueError("noise_snr_db must be an ordered pair")
        if self.enrollment_noise_snr_db is not None and (
            len(self.enrollment_noise_snr_db) != 2 or self.enrollment_noise_snr_db[0] > self.enrollment_noise_snr_db[1]
        ):
            raise ValueError("enrollment_noise_snr_db must be an ordered pair")
        if self.rir_drr_db is not None and (
            len(self.rir_drr_db) != 2 or self.rir_drr_db[0] > self.rir_drr_db[1]
        ):
            raise ValueError("rir_drr_db must be an ordered pair")
        for name in ("far_level_dbfs", "level_dbfs"):
            pair = getattr(self, name)
            if pair is not None and (len(pair) != 2 or pair[0] > pair[1] or pair[1] > 0):
                raise ValueError(f"{name} must be an ordered pair <= 0 dBFS")
        if self.level_prob and self.level_dbfs is None:
            raise ValueError("level_prob requires level_dbfs")
        if self.time_mask_width < 0 or self.frequency_mask_width < 0:
            raise ValueError("SpecAugment mask widths must be non-negative")


def for_slot(config, slot_name):
    """The augmentation one slot gets: the enrollment may have its own noise level.

    The enrollment level is part of the TS contract (enrollment_quiet_prob), so level_prob
    never moves it.
    """
    if slot_name != "enrollment":
        return config
    if config.enrollment_noise_snr_db is not None:
        config = replace(config, noise_snr_db=tuple(config.enrollment_noise_snr_db))
    return replace(config, level_prob=0.0) if config.level_prob else config


def limit_peak(audio):
    """Scale a clip past full scale back to it; clips within full scale are unchanged."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= FULL_SCALE:
        return audio
    return (audio * (FULL_SCALE / peak)).astype(np.float32)


def direct_to_reverberant_db(impulse, sampling_rate):
    """Energy within DIRECT_PATH_S after the peak over the energy after it."""
    impulse = np.asarray(impulse, dtype=np.float64)
    impulse = impulse[int(np.argmax(np.abs(impulse))) :]
    split = max(1, round(DIRECT_PATH_S * sampling_rate))
    reverberant = np.sum(impulse[split:] ** 2)
    if reverberant == 0:
        return float("inf")
    return float(10 * np.log10(np.sum(impulse[:split] ** 2) / reverberant))


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
    roll = rng.random()
    if roll < config.far_rir_prob:
        rir_mode = "far"
    elif roll < config.far_rir_prob + config.rir_prob:
        rir_mode = "ordinary"
    else:
        rir_mode = None
    tail, reverberated = None, False
    if rir_mode is not None:
        if rir_loader is None:
            raise ValueError("RIR convolution requires a Catalog RIR source")
        impulse = rir_loader(rng, sampling_rate, rir_mode)
        # Start at the direct path so the output keeps the input's timing.
        impulse = impulse[int(np.argmax(np.abs(impulse))) :]
        energy = np.sqrt(np.sum(impulse**2))
        if energy > 0:
            audio = fftconvolve(audio, impulse / energy)[: len(audio)]
            reverberated = True
            # Past the early reflections the field is diffuse and roughly the same
            # anywhere in the room, so it stands in for the noise source's RIR.
            tail = impulse[round(LATE_REVERB_S * sampling_rate) :]
            if not np.any(tail):
                tail = None
    if rng.random() < config.noise_prob:
        if noise_loader is None:
            raise ValueError("noise_prob requires a Catalog noise source")
        noise = noise_loader(rng, sampling_rate)
        reverb = tail if config.reverb_noise else None
        # "valid" convolution needs len(tail) - 1 extra samples and skips the onset.
        needed = len(audio) + (len(reverb) - 1 if reverb is not None else 0)
        if len(noise) < needed:
            noise = np.tile(noise, (needed + len(noise) - 1) // len(noise))
        start = rng.randrange(len(noise) - needed + 1)
        noise = noise[start : start + needed]
        if reverb is not None:
            noise = fftconvolve(noise, reverb, mode="valid")
        signal_power, noise_power = np.mean(audio**2), np.mean(noise**2)
        if noise_power > 0:
            snr = rng.uniform(*config.noise_snr_db)
            audio = audio + noise * np.sqrt(
                signal_power / (noise_power * 10 ** (snr / 10))
            )
    if reverberated and config.far_level_dbfs is not None:
        power = np.mean(audio**2)
        if power > 0:
            level = rng.uniform(*config.far_level_dbfs)
            audio = audio * (10 ** (level / 20) / np.sqrt(power))
    # Draw only when enabled, so recipes without it keep their random stream.
    elif config.level_prob and rng.random() < config.level_prob:
        audio = set_level(rng, audio, config.level_dbfs)
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
