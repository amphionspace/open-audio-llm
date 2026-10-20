"""Mix target-speaker samples on the fly from a clean target and other speakers.

A source with ``online_mix`` holds records whose mixture slot is one clean
target utterance and whose enrollment slot is another utterance of the same
speaker. Each sample draws its own interferers, per-speaker reverberation,
SIR, overlap, input level, enrollment length and contamination, so the
model sees new conditions every epoch instead of a fixed offline mixture.
"""

from __future__ import annotations

import math
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
    # Target timing: with entry_prob the window is longer than the target, so the target
    # enters late, leaves early or both; interferers and the noise floor fill the rest.
    entry_prob: float = 0.0
    lead_seconds: tuple[float, float] = (0.5, 8.0)
    trail_seconds: tuple[float, float] = (0.5, 6.0)
    # A speech-free window (noise, music) with an empty transcript replaces the mixture.
    nonspeech_prob: float = 0.0
    nonspeech_seconds: tuple[float, float] = (0.3, 26.0)
    nonspeech_level_dbfs: tuple[float, float] = (-60.0, -20.0)
    # Low-level noise floor under the mixture: white, pink or recorded (speech-free) noise.
    floor_prob: float = 0.0
    floor_dbfs: tuple[float, float] = (-65.0, -35.0)
    floor_kinds: tuple[float, float, float] = (0.15, 0.15, 0.70)
    # Enrollment through a narrowband channel, or recorded with a silent start.
    enrollment_lowpass_prob: float = 0.0
    enrollment_lowpass_hz: float = 4000.0
    enrollment_lead_silence_prob: float = 0.0
    enrollment_lead_silence_seconds: tuple[float, float] = (0.3, 2.0)
    # Enrollment spoken quietly in an unchanged room: the speech is turned down by
    # enrollment_quiet_db and a noise floor (floor_dbfs, floor_kinds) is laid under it. A gain
    # alone would be undone by the slot's level_dbfs normalisation; the floor keeps the drop.
    enrollment_quiet_prob: float = 0.0
    enrollment_quiet_db: tuple[float, float] = (-30.0, -10.0)
    # Mixture through a telephone channel (300-3400 Hz, 8-bit mu-law) or driven into clipping.
    telephony_prob: float = 0.0
    clip_prob: float = 0.0
    clip_gain_db: tuple[float, float] = (6.0, 18.0)
    # Scene records (metadata.scene) name every utterance; these are the silences between
    # one speaker's utterances, or between turns when the mode is alternate.
    alternate_gap_seconds: tuple[float, float] = (0.2, 1.0)
    burst_gap_seconds: tuple[float, float] = (0.0, 0.5)
    short_gap_seconds: tuple[float, float] = (0.0, 0.3)

    def __post_init__(self):
        for name in ("interferer_probs", "absent_interferer_probs"):
            probs = getattr(self, name)
            if not probs or any(p < 0 for p in probs) or not np.isclose(sum(probs), 1.0):
                raise ValueError(f"{name} must be non-negative and sum to 1")
        if len(self.floor_kinds) != 3 or any(p < 0 for p in self.floor_kinds) or not np.isclose(sum(self.floor_kinds), 1.0):
            raise ValueError("floor_kinds must be three non-negative weights that sum to 1")
        for name in ("entry_prob", "nonspeech_prob", "floor_prob", "enrollment_lowpass_prob",
                     "enrollment_lead_silence_prob", "enrollment_quiet_prob", "telephony_prob", "clip_prob"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("lead_seconds", "trail_seconds", "nonspeech_seconds", "nonspeech_level_dbfs",
                     "floor_dbfs", "enrollment_lead_silence_seconds", "enrollment_quiet_db", "clip_gain_db",
                     "alternate_gap_seconds", "burst_gap_seconds", "short_gap_seconds"):
            pair = getattr(self, name)
            if len(pair) != 2 or pair[0] > pair[1]:
                raise ValueError(f"{name} must be an ordered pair")
        if self.nonspeech_seconds[0] <= 0 or self.enrollment_lowpass_hz <= 0:
            raise ValueError("nonspeech_seconds and enrollment_lowpass_hz must be positive")
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
    # Start at the direct path so each speaker keeps its place in the window.
    impulse = impulse[int(np.argmax(np.abs(impulse))) :]
    energy = np.sqrt(np.sum(impulse**2))
    if energy <= 0:
        return audio
    return fftconvolve(audio, impulse / energy)[: len(audio)].astype(np.float32)


def timing(rng, config, sampling_rate):
    """Samples of padding before and after the target; (0, 0) keeps the target as the window."""
    if not config.entry_prob or rng.random() >= config.entry_prob:
        return 0, 0
    mode = rng.choice(("enter", "leave", "middle"))
    lead = int(rng.uniform(*config.lead_seconds) * sampling_rate) if mode != "leave" else 0
    trail = int(rng.uniform(*config.trail_seconds) * sampling_rate) if mode != "enter" else 0
    return lead, trail


def build_mixture(rng, target, config, *, absent, speech_loader, rir_loader, sampling_rate, info=None):
    """Return the mixture window and the number of interferers used.

    The window is the target plus any padding drawn by ``timing``, so the target can enter
    late or leave early. When the target is absent the window holds only interferers, or
    silence for the later noise step. ``info`` receives the padding in seconds.
    """
    target = np.asarray(target, dtype=np.float32)
    length = len(target)
    lead, trail = timing(rng, config, sampling_rate)
    window = lead + length + trail
    probs = config.absent_interferer_probs if absent else config.interferer_probs
    count = _draw_count(rng, probs)
    present = reverberate(rng, target, config, rir_loader, sampling_rate)
    reference = _rms(present[np.abs(present) > 0])
    mixture = np.zeros(window, dtype=np.float32)
    if not absent:
        mixture[lead : lead + length] = present
    for _ in range(count):
        other = reverberate(rng, speech_loader(rng, sampling_rate), config, rir_loader, sampling_rate)
        other = _place(rng, other, window)
        mixture += _scale_to(other, reference, rng.uniform(*config.sir_db))
    if info is not None:
        info.update(lead=lead / sampling_rate, trail=trail / sampling_rate, window=window / sampling_rate)
    return mixture, count


def _track(utterances, gap):
    """One speaker's utterances in a row, with a random silence between them."""
    pieces = []
    for number, audio in enumerate(utterances):
        if number:
            pieces.append(np.zeros(gap(), dtype=np.float32))
        pieces.append(np.asarray(audio, dtype=np.float32))
    return np.concatenate(pieces)


def _on_timeline(audio, start, length):
    out = np.zeros(length, dtype=np.float32)
    out[start : start + len(audio)] = audio
    return out


SCENE_MODES = ("alternate", "burst", "short", "full")


def build_scene(rng, mode, targets, others, config, *, rir_loader, sampling_rate, info=None, gaps=None):
    """Mix a scene whose utterances are all given.

    alternate: the target's utterances and the others' take turns, never overlapping.
    burst / short: each other speaker talks all its utterances in a row and the target
    speaks inside that stretch, so the target is overlapped for (nearly) all of its time.
    full: like burst, but every speaker has its own silence between utterances (``gaps``:
    ``{"target": (low, high), "others": [(low, high), ...]}`` in seconds), so the target can speak
    several times with other speakers talking in between and over it.
    Each speaker is reverberated as a whole, then the others are set against the target by SIR.
    """
    if mode not in SCENE_MODES:
        raise ValueError(f"unknown scene mode {mode!r}")
    if mode == "full":
        if not gaps or len(gaps.get("others", ())) != len(others):
            raise ValueError("a full scene needs a gap range for the target and for each other speaker")
        draw = lambda bounds: (lambda: int(rng.uniform(*bounds) * sampling_rate))
        gap, target_gap = None, draw(gaps["target"])
        other_gaps = [draw(bounds) for bounds in gaps["others"]]
    else:
        low, high = getattr(config, f"{mode}_gap_seconds")
        gap = lambda: int(rng.uniform(low, high) * sampling_rate)
        target_gap, other_gaps = gap, [gap] * len(others)
    if mode == "alternate":
        # Turns in random order; the target's utterances keep their order (the transcript's).
        turns = [None] * len(targets) + [(j, k) for j, utterances in enumerate(others) for k in range(len(utterances))]
        rng.shuffle(turns)
        placed, cursor, spoken = [], 0, 0
        for turn in turns:
            if turn is None:
                audio, speaker = targets[spoken], None
                spoken += 1
            else:
                audio, speaker = others[turn[0]][turn[1]], turn[0]
            placed.append((speaker, cursor, audio))
            cursor += len(audio) + gap()
        length = max(start + len(audio) for _, start, audio in placed)
        target_track = np.zeros(length, dtype=np.float32)
        tracks = [np.zeros(length, dtype=np.float32) for _ in others]
        for speaker, start, audio in placed:
            (target_track if speaker is None else tracks[speaker])[start : start + len(audio)] = audio
    else:
        packed = [_track(utterances, other_gap) for utterances, other_gap in zip(others, other_gaps)]
        target = _track(targets, target_gap) if targets else np.zeros(0, dtype=np.float32)
        length = max([len(target), *map(len, packed)])
        target_track = _on_timeline(target, rng.randrange(length - len(target) + 1), length)
        tracks = [_on_timeline(audio, rng.randrange(length - len(audio) + 1), length) for audio in packed]
    present = reverberate(rng, target_track, config, rir_loader, sampling_rate)
    reference = _rms(present[np.abs(present) > 0])
    mixture = present.copy()
    for number, track in enumerate(tracks):
        other = reverberate(rng, track, config, rir_loader, sampling_rate)
        if not targets and number == 0:
            # The target is absent: the first other speaker sets the level the rest are set against.
            reference = _rms(other[np.abs(other) > 0])
            mixture += other
        else:
            mixture += _scale_to(other, reference, rng.uniform(*config.sir_db))
    if info is not None:
        spoken = target_track != 0
        others_on = np.any([track != 0 for track in tracks], axis=0) if tracks else np.zeros(length, dtype=bool)
        info.update(lead=0.0, trail=0.0, window=length / sampling_rate,
                    overlap=float(np.mean(others_on[spoken])) if spoken.any() else 0.0)
    return mixture


def scene_seconds(scene, config):
    """Longest the scene's mixture can run: build_scene with every gap at its upper bound.

    The record's mixture slot only names the first target utterance, so this is what
    the sampler must budget for instead.
    """
    track = lambda utterances, gap: sum(u["duration"] for u in utterances) + gap * max(len(utterances) - 1, 0)
    if scene["mode"] == "full":
        return max([track(scene["targets"], scene["target_gap"][1]),
                    *(track(other["utts"], other["gap"][1]) for other in scene["interferers"])])
    gap = getattr(config, f"{scene['mode']}_gap_seconds")[1]
    if scene["mode"] == "alternate":
        utterances = [u for u in scene["targets"]] + [u for other in scene["interferers"] for u in other["utts"]]
        return track(utterances, gap)
    return max([track(scene["targets"], gap), *(track(other["utts"], gap) for other in scene["interferers"])])


def _synthetic_noise(rng, kind, length):
    generator = np.random.default_rng(rng.getrandbits(64))
    noise = generator.standard_normal(length).astype(np.float32)
    if kind == "pink" and length > 1:
        spectrum = np.fft.rfft(noise)
        frequency = np.maximum(np.arange(len(spectrum)), 1)
        noise = np.fft.irfft(spectrum / np.sqrt(frequency), n=length).astype(np.float32)
    return noise


def _fit_noise(rng, clip, length):
    """Tile a recorded clip to ``length`` and take a random window."""
    clip = np.asarray(clip, dtype=np.float32)
    if len(clip) < length:
        clip = np.tile(clip, (length + len(clip) - 1) // len(clip))
    start = rng.randrange(len(clip) - length + 1)
    return clip[start : start + length]


def noise_floor(rng, length, config, noise_loader, sampling_rate):
    """A white, pink or recorded noise floor at a random level within floor_dbfs."""
    kind = rng.choices(("white", "pink", "recorded"), weights=config.floor_kinds)[0]
    if kind == "recorded" and noise_loader is not None:
        noise = _fit_noise(rng, noise_loader(rng, sampling_rate), length)
    else:
        noise = _synthetic_noise(rng, "pink" if kind == "recorded" else kind, length)
    rms = _rms(noise)
    if rms < _SILENT_RMS:
        return np.zeros(length, dtype=np.float32)
    return (noise * (10 ** (rng.uniform(*config.floor_dbfs) / 20) / rms)).astype(np.float32)


def nonspeech_window(rng, config, noise_loader, sampling_rate, length=None):
    """A speech-free window: ``length`` samples, or a log-uniform duration if not given."""
    if length is None:
        low, high = config.nonspeech_seconds
        length = max(1, int(math.exp(rng.uniform(math.log(low), math.log(high))) * sampling_rate))
    return np.asarray(_fit_noise(rng, noise_loader(rng, sampling_rate), length), dtype=np.float32)


def slot_level(config, nonspeech, name):
    """Level range for a slot: a nonspeech mixture keeps its own, wider range."""
    return config.nonspeech_level_dbfs if nonspeech and name == "mixture" else config.level_dbfs


def lowpass(audio, hz, sampling_rate):
    from scipy.signal import butter, sosfiltfilt

    return sosfiltfilt(butter(8, hz, "low", fs=sampling_rate, output="sos"), audio).astype(np.float32)


def telephone(audio, sampling_rate):
    """300-3400 Hz band-pass and 8-bit mu-law, as in a narrowband call."""
    from scipy.signal import butter, sosfiltfilt

    band = sosfiltfilt(butter(6, (300, 3400), "bandpass", fs=sampling_rate, output="sos"), audio)
    peak = float(np.max(np.abs(band)))
    if peak < _SILENT_RMS:
        return audio
    mu = 255.0
    x = band / peak
    quantised = np.round(np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu) * 127) / 127
    return (np.sign(quantised) * np.expm1(np.abs(quantised) * np.log1p(mu)) / mu * peak).astype(np.float32)


def overdrive(rng, audio, config):
    """Scale the peak past full scale and clip, as a recorder with too much gain does."""
    peak = float(np.max(np.abs(audio)))
    if peak < _SILENT_RMS:
        return audio
    gain = 10 ** (rng.uniform(*config.clip_gain_db) / 20)
    return np.clip(audio / peak * gain, -1.0, 1.0).astype(np.float32)


def degrade_enrollment(rng, enrollment, config, noise_loader, sampling_rate):
    """Turn the enrollment down over a noise floor, band-limit it and/or give it a quiet start."""
    if config.enrollment_quiet_prob and rng.random() < config.enrollment_quiet_prob:
        gain = 10 ** (rng.uniform(*config.enrollment_quiet_db) / 20)
        enrollment = enrollment * gain + noise_floor(rng, len(enrollment), config, noise_loader, sampling_rate)
    if config.enrollment_lowpass_prob and rng.random() < config.enrollment_lowpass_prob:
        enrollment = lowpass(enrollment, config.enrollment_lowpass_hz, sampling_rate)
    if config.enrollment_lead_silence_prob and rng.random() < config.enrollment_lead_silence_prob:
        pad = int(rng.uniform(*config.enrollment_lead_silence_seconds) * sampling_rate)
        enrollment = np.concatenate([noise_floor(rng, pad, config, noise_loader, sampling_rate), enrollment])
    return enrollment


def degrade_mixture(rng, mixture, config, noise_loader, sampling_rate):
    """Noise floor, telephone channel and clipping, in the order a recording meets them."""
    if config.floor_prob and rng.random() < config.floor_prob:
        mixture = mixture + noise_floor(rng, len(mixture), config, noise_loader, sampling_rate)
    if config.telephony_prob and rng.random() < config.telephony_prob:
        mixture = telephone(mixture, sampling_rate)
    if config.clip_prob and rng.random() < config.clip_prob:
        mixture = overdrive(rng, mixture, config)
    return mixture


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
