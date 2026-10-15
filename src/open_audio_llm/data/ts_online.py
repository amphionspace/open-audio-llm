"""Mix target-speaker samples on the fly from a clean target and other speakers.

A source with ``online_mix`` holds records whose mixture slot is one clean
target utterance and whose enrollment slot is another utterance of the same
speaker. Each sample draws its own interferers, per-speaker reverberation,
SIR, overlap, input level, enrollment length and contamination, so the
model sees new conditions every epoch instead of a fixed offline mixture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this RMS a clip is treated as silence and is not rescaled.
_SILENT_RMS = 1e-6
# Peak ceiling after level scaling, to avoid clipping in the WAV transport.
_MAX_PEAK = 0.99


@dataclass(frozen=True)
class OnlineMixConfig:
    # Probability of 0, 1, 2, ... interferers when the target is present.
    interferer_probs: tuple[float, ...] = (0.25, 0.40, 0.25, 0.10)
    # Target absent: the transcript becomes empty and only interferers (or
    # nothing but the noise of the later augmentation) remain.
    absent_prob: float = 0.15
    absent_interferer_probs: tuple[float, ...] = (0.2, 0.4, 0.3, 0.1)
    sir_db: tuple[float, float] = (-6.0, 10.0)
    # Each speaker is reverberated on its own, so a far target with a dry
    # interferer occurs naturally.
    speaker_rir_prob: float = 0.5
    speaker_far_rir_prob: float = 0.5
    level_dbfs: tuple[float, float] | None = (-50.0, -15.0)
    enrollment_seconds: tuple[float, float] = (1.0, 20.0)
    enrollment_contam_prob: float = 0.1
    enrollment_contam_sir_db: tuple[float, float] = (5.0, 15.0)

    def __post_init__(self):
        for name in ("interferer_probs", "absent_interferer_probs"):
            probs = getattr(self, name)
            if not probs or any(p < 0 for p in probs) or not np.isclose(sum(probs), 1.0):
                raise ValueError(f"{name} must be non-negative and sum to 1")
        for name in ("absent_prob", "speaker_rir_prob", "speaker_far_rir_prob", "enrollment_contam_prob"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("sir_db", "enrollment_seconds", "enrollment_contam_sir_db", "level_dbfs"):
            pair = getattr(self, name)
            if pair is not None and (len(pair) != 2 or pair[0] > pair[1]):
                raise ValueError(f"{name} must be an ordered pair")
        if self.enrollment_seconds[0] <= 0:
            raise ValueError("enrollment_seconds must be positive")


def _rms(audio):
    return float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0


def _draw_count(rng, probs):
    mark = rng.random()
    for count, prob in enumerate(probs):
        mark -= prob
        if mark < 0:
            return count
    return len(probs) - 1


def _fit(rng, audio, length):
    """Random crop or zero-pad ``audio`` to ``length`` at a random offset."""
    if len(audio) >= length:
        start = rng.randrange(len(audio) - length + 1)
        return audio[start : start + length]
    out = np.zeros(length, dtype=np.float32)
    start = rng.randrange(length - len(audio) + 1)
    out[start : start + len(audio)] = audio
    return out


def _place(rng, audio, length):
    """Place an interferer so it overlaps the window partly or fully."""
    out = np.zeros(length, dtype=np.float32)
    # Start anywhere from fully before the window to its last sample, so
    # overlap ranges from a short tail to the whole window.
    start = rng.randrange(-len(audio) + 1, length)
    begin, end = max(start, 0), min(start + len(audio), length)
    out[begin:end] = audio[begin - start : end - start]
    return out


def _scale_to(audio, reference_rms, ratio_db):
    """Scale ``audio`` so reference_rms / rms(audio) equals ``ratio_db``."""
    active = audio[np.abs(audio) > 0]
    rms = _rms(active)
    if rms < _SILENT_RMS or reference_rms < _SILENT_RMS:
        return audio
    return audio * (reference_rms / rms / 10 ** (ratio_db / 20))


def set_level(rng, audio, level_dbfs):
    """Scale the whole clip to a random RMS level, never clipping."""
    if level_dbfs is None:
        return audio
    rms = _rms(audio)
    if rms < _SILENT_RMS:
        return audio
    gain = 10 ** (rng.uniform(*level_dbfs) / 20) / rms
    peak = float(np.max(np.abs(audio))) * gain
    if peak > _MAX_PEAK:
        gain *= _MAX_PEAK / peak
    return (audio * gain).astype(np.float32)


def reverberate(rng, audio, config, rir_loader, sampling_rate):
    from scipy.signal import fftconvolve

    if rir_loader is None or rng.random() >= config.speaker_rir_prob:
        return audio
    mode = "far" if rng.random() < config.speaker_far_rir_prob else "ordinary"
    impulse = rir_loader(rng, sampling_rate, mode)
    energy = np.sqrt(np.sum(impulse**2))
    if energy <= 0:
        return audio
    return fftconvolve(audio, impulse / energy)[: len(audio)].astype(np.float32)


def build_mixture(rng, target, config, *, absent, speech_loader, rir_loader, sampling_rate):
    """Return the mixture window and the number of interferers used.

    ``target`` keeps its length as the window. When the target is absent the
    window holds only interferers, or silence for the later noise step.
    """
    target = np.asarray(target, dtype=np.float32)
    length = len(target)
    probs = config.absent_interferer_probs if absent else config.interferer_probs
    count = _draw_count(rng, probs)
    present = reverberate(rng, target, config, rir_loader, sampling_rate)
    reference = _rms(present[np.abs(present) > 0])
    mixture = np.zeros(length, dtype=np.float32) if absent else present.copy()
    for _ in range(count):
        other = reverberate(rng, speech_loader(rng, sampling_rate), config, rir_loader, sampling_rate)
        other = _place(rng, other, length)
        mixture += _scale_to(other, reference, rng.uniform(*config.sir_db))
    return mixture, count


def crop_enrollment(rng, enrollment, config, sampling_rate):
    """Take a random window of 1–20 s (by default) from the enrollment."""
    enrollment = np.asarray(enrollment, dtype=np.float32)
    seconds = rng.uniform(*config.enrollment_seconds)
    length = min(len(enrollment), max(1, int(round(seconds * sampling_rate))))
    start = rng.randrange(len(enrollment) - length + 1)
    return enrollment[start : start + length]


def contaminate_enrollment(rng, enrollment, config, speech_loader, sampling_rate):
    """Occasionally leak another speaker into the enrollment, as real recordings do."""
    if rng.random() >= config.enrollment_contam_prob:
        return enrollment
    other = _fit(rng, speech_loader(rng, sampling_rate), len(enrollment))
    reference = _rms(enrollment[np.abs(enrollment) > 0])
    return enrollment + _scale_to(other, reference, rng.uniform(*config.enrollment_contam_sir_db))
