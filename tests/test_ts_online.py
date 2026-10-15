import random

import numpy as np
import pytest

from open_audio_llm.data.ts_online import (
    OnlineMixConfig,
    build_mixture,
    contaminate_enrollment,
    crop_enrollment,
    set_level,
)

SR = 16000


def _tone(freq, seconds=1.0, amplitude=0.1):
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(x**2)))


def test_one_interferer_is_mixed_at_the_drawn_sir():
    target, other = _tone(200), _tone(500, amplitude=0.5)
    config = OnlineMixConfig(interferer_probs=(0, 1), sir_db=(6, 6), speaker_rir_prob=0)
    mixture, count = build_mixture(random.Random(0), target, config, absent=False,
                                   speech_loader=lambda r, sr: other, rir_loader=None, sampling_rate=SR)
    assert count == 1 and len(mixture) == len(target)
    leaked = mixture - target
    active = leaked[np.abs(leaked) > 0]
    assert 20 * np.log10(_rms(target) / _rms(active)) == pytest.approx(6, abs=0.1)


def test_absent_target_leaves_only_interferers_or_silence():
    target = _tone(200)
    config = OnlineMixConfig(absent_interferer_probs=(1.0,), speaker_rir_prob=0)
    mixture, count = build_mixture(random.Random(1), target, config, absent=True,
                                   speech_loader=lambda r, sr: pytest.fail("no interferer"),
                                   rir_loader=None, sampling_rate=SR)
    assert count == 0 and not mixture.any()


def test_each_speaker_draws_its_own_reverberation():
    calls = []

    def rir(r, sr, mode):
        calls.append(mode)
        impulse = np.zeros(10, dtype=np.float32)
        impulse[0] = 1
        return impulse

    config = OnlineMixConfig(interferer_probs=(0, 0, 1), speaker_rir_prob=1, speaker_far_rir_prob=1)
    build_mixture(random.Random(2), _tone(200), config, absent=False,
                  speech_loader=lambda r, sr: _tone(400), rir_loader=rir, sampling_rate=SR)
    assert calls == ["far", "far", "far"]  # target and both interferers


def test_level_hits_the_requested_rms_without_clipping():
    rng = random.Random(3)
    quiet = set_level(rng, _tone(200), (-45, -45))
    assert 20 * np.log10(_rms(quiet)) == pytest.approx(-45, abs=0.01)
    loud = set_level(rng, _tone(200), (0, 0))
    assert np.max(np.abs(loud)) <= 0.99 + 1e-6
    assert not set_level(rng, np.zeros(100, np.float32), (-20, -20)).any()


def test_enrollment_window_and_contamination():
    config = OnlineMixConfig(enrollment_seconds=(1, 2), enrollment_contam_prob=1,
                             enrollment_contam_sir_db=(10, 10))
    enrollment = _tone(200, seconds=5)
    for seed in range(20):
        cropped = crop_enrollment(random.Random(seed), enrollment, config, SR)
        assert SR <= len(cropped) <= 2 * SR
    short = crop_enrollment(random.Random(0), _tone(200, seconds=0.5), config, SR)
    assert len(short) == SR // 2  # never longer than what was recorded
    clean = _tone(200, seconds=2)
    mixed = contaminate_enrollment(random.Random(4), clean, config, lambda r, sr: _tone(700), SR)
    leaked = mixed - clean
    # SIR is measured where the other speaker actually talks.
    assert 20 * np.log10(_rms(clean) / _rms(leaked[np.abs(leaked) > 0])) == pytest.approx(10, abs=0.1)


@pytest.mark.parametrize("bad", [
    {"interferer_probs": (0.5, 0.4)},
    {"absent_prob": 1.5},
    {"sir_db": (5, -5)},
    {"enrollment_seconds": (0, 3)},
])
def test_config_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        OnlineMixConfig(**bad)
