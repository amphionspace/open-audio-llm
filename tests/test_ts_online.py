import random

import numpy as np
import pytest

from open_audio_llm.data.ts_online import (
    OnlineMixConfig,
    activity_codes,
    build_mixture,
    build_scene,
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


def test_reverberation_starts_at_the_direct_path():
    impulse = np.zeros(400, dtype=np.float32)
    impulse[300] = 1.0  # 300 samples of pre-delay before the direct path
    config = OnlineMixConfig(interferer_probs=(1.0,), speaker_rir_prob=1, speaker_far_rir_prob=0)
    target = _tone(200)
    mixture, _ = build_mixture(random.Random(5), target, config, absent=False,
                               speech_loader=None, rir_loader=lambda r, sr, mode: impulse, sampling_rate=SR)
    np.testing.assert_allclose(mixture / _rms(mixture), target / _rms(target), atol=1e-5)


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


def _impulse_free(config, **kwargs):
    return build_mixture(random.Random(kwargs.pop("seed", 0)), _tone(200, 2.0), config, absent=kwargs.pop("absent", False),
                         speech_loader=lambda r, sr: _tone(500, 1.0, 0.3), rir_loader=None, sampling_rate=SR, **kwargs)


def test_target_can_enter_late_and_leave_early():
    from open_audio_llm.data.ts_online import timing

    config = OnlineMixConfig(interferer_probs=(1.0,), entry_prob=1.0, lead_seconds=(1, 1), trail_seconds=(0.5, 0.5))
    seen = set()
    for seed in range(60):
        lead, trail = timing(random.Random(seed), config, SR)
        seen.add((lead > 0, trail > 0))
    assert seen == {(True, False), (False, True), (True, True)}  # enter, leave, middle

    info = {}
    config = OnlineMixConfig(interferer_probs=(1.0,), entry_prob=1.0, lead_seconds=(3, 3), trail_seconds=(0, 0))
    mixture, _ = _impulse_free(config, info=info, seed=3)
    assert len(mixture) == int((info["lead"] + 2.0 + info["trail"]) * SR)
    if info["lead"]:  # nothing of the target before it enters
        assert not mixture[: int(info["lead"] * SR)].any()


def test_timing_is_off_by_default_and_keeps_the_random_stream():
    config = OnlineMixConfig(interferer_probs=(0, 1), speaker_rir_prob=0)
    a, _ = _impulse_free(config, seed=5)
    b, _ = _impulse_free(OnlineMixConfig(interferer_probs=(0, 1), speaker_rir_prob=0, entry_prob=0.0), seed=5)
    assert len(a) == 2 * SR and np.array_equal(a, b)


def test_noise_floor_levels_and_kinds():
    from open_audio_llm.data.ts_online import noise_floor

    config = OnlineMixConfig(floor_dbfs=(-50, -50), floor_kinds=(0.0, 1.0, 0.0))
    pink = noise_floor(random.Random(0), SR, config, None, SR)
    assert 20 * np.log10(_rms(pink)) == pytest.approx(-50, abs=0.2)
    spectrum = np.abs(np.fft.rfft(pink)) ** 2
    assert spectrum[10:100].mean() > 10 * spectrum[-2000:].mean()  # 1/f: more low-frequency power
    recorded = OnlineMixConfig(floor_dbfs=(-40, -40), floor_kinds=(0.0, 0.0, 1.0))
    floor = noise_floor(random.Random(1), 3000, recorded, lambda r, sr: _tone(300, 0.1), SR)
    assert len(floor) == 3000 and 20 * np.log10(_rms(floor)) == pytest.approx(-40, abs=0.2)


def test_nonspeech_window_duration_is_log_uniform_within_bounds():
    from open_audio_llm.data.ts_online import nonspeech_window

    config = OnlineMixConfig(nonspeech_seconds=(0.3, 26.0))
    lengths = [len(nonspeech_window(random.Random(i), config, lambda r, sr: _tone(100, 2.0), SR)) / SR for i in range(300)]
    assert 0.29 <= min(lengths) and max(lengths) <= 26.01
    assert np.median(lengths) == pytest.approx(np.sqrt(0.3 * 26.0), rel=0.35)  # log-uniform median


def test_nonspeech_level_range_applies_only_to_the_mixture():
    from open_audio_llm.data.ts_online import slot_level

    config = OnlineMixConfig(level_dbfs=(-50, -15), nonspeech_level_dbfs=(-60, -20))
    assert slot_level(config, True, "mixture") == (-60, -20)
    assert slot_level(config, True, "enrollment") == (-50, -15)
    assert slot_level(config, False, "mixture") == (-50, -15)


def test_degradations_change_the_signal_as_documented():
    from open_audio_llm.data.ts_online import degrade_enrollment, degrade_mixture, overdrive, telephone

    low, high = _tone(1000, 1.0, 0.2), _tone(6000, 1.0, 0.2)
    config = OnlineMixConfig(enrollment_lowpass_prob=1.0, enrollment_lowpass_hz=4000)
    assert _rms(degrade_enrollment(random.Random(0), high, config, None, SR)) < 0.01 * _rms(high)
    assert _rms(degrade_enrollment(random.Random(0), low, config, None, SR)) > 0.9 * _rms(low)
    padded = degrade_enrollment(random.Random(0), low, OnlineMixConfig(enrollment_lead_silence_prob=1.0,
                                enrollment_lead_silence_seconds=(0.5, 0.5), floor_dbfs=(-60, -60)), None, SR)
    assert len(padded) == len(low) + SR // 2 and _rms(padded[: SR // 2]) < 0.01
    phone = telephone(_tone(100, 1.0, 0.3) + _tone(1000, 1.0, 0.3), SR)
    spectrum = np.abs(np.fft.rfft(phone))
    assert spectrum[1000] > 20 * spectrum[100]  # 100 Hz is outside 300-3400 Hz
    clipped = overdrive(random.Random(0), _tone(300, 1.0, 0.1), OnlineMixConfig(clip_gain_db=(12, 12)))
    assert np.max(np.abs(clipped)) == 1.0 and np.mean(np.abs(clipped) >= 1.0) > 0.2
    untouched = degrade_mixture(random.Random(0), low, OnlineMixConfig(), None, SR)
    assert np.array_equal(untouched, low)


def test_quiet_enrollment_is_turned_down_over_an_unchanged_floor():
    from open_audio_llm.data.ts_online import degrade_enrollment

    speech = _tone(1000, 1.0, 0.2)
    config = OnlineMixConfig(enrollment_quiet_prob=1.0, enrollment_quiet_db=(-20, -20),
                             floor_dbfs=(-50, -50), floor_kinds=(1.0, 0.0, 0.0))
    floor = degrade_enrollment(random.Random(0), speech, config, None, SR) - 0.1 * speech
    # The floor keeps its absolute level, so the speech-to-floor ratio drops by the full 20 dB.
    assert _rms(floor) == pytest.approx(10 ** (-50 / 20), rel=0.05)
    assert abs(np.corrcoef(floor, speech)[0, 1]) < 0.05
    rng = random.Random(0)
    assert np.array_equal(degrade_enrollment(rng, speech, OnlineMixConfig(), None, SR), speech)
    assert rng.random() == random.Random(0).random()  # off by default: the random stream is unchanged


@pytest.mark.parametrize("bad", [{"entry_prob": 1.5}, {"lead_seconds": (3, 1)}, {"floor_kinds": (0.5, 0.5, 0.5)},
                                 {"nonspeech_seconds": (0, 5)}, {"clip_prob": -0.1},
                                 {"enrollment_quiet_prob": 1.5}, {"enrollment_quiet_db": (-5, -20)}])
def test_new_config_values_are_validated(bad):
    with pytest.raises(ValueError):
        OnlineMixConfig(**bad)


def _scene(mode, targets, others, seed=0, **config):
    return build_scene(random.Random(seed), mode, targets, others, OnlineMixConfig(speaker_rir_prob=0, sir_db=(0, 0), **config),
                       rir_loader=None, sampling_rate=SR)


def _active(mixture, freq, frame=800):
    """Frames in which a tone of ``freq`` sounds (projection on that frequency)."""
    t = np.arange(len(mixture) // frame * frame) / SR
    carrier = np.exp(-2j * np.pi * freq * t)
    return np.abs((mixture[: len(t)] * carrier).reshape(-1, frame).mean(axis=1)) > 0.02


def test_alternate_scene_never_overlaps_and_keeps_the_targets_in_order():
    first, second = _tone(200, 1.0), _tone(300, 1.5)
    others = [[_tone(500, 1.0)], [_tone(700, 1.0), _tone(900, 1.0)]]
    for seed in range(20):
        mixture = _scene("alternate", [first, second], others, seed=seed, alternate_gap_seconds=(0.2, 0.4))
        tones = np.array([_active(mixture, f) for f in (200, 300, 500, 700, 900)])
        assert tones.sum(axis=0).max() == 1  # one speaker at a time
        assert np.argmax(tones[0]) < np.argmax(tones[1])  # the transcript's order


def test_burst_scene_overlaps_the_target_with_the_others_for_its_whole_length():
    target = _tone(200, 1.0)
    others = [[_tone(500, 1.0)] * 4]  # a 4+ s stretch
    for seed in range(10):
        mixture = _scene("burst", [target], others, seed=seed, burst_gap_seconds=(0, 0))
        assert len(mixture) == 4 * SR
        # wherever the 200 Hz target sounds, the 500 Hz speaker sounds too
        target_on, other_on = _active(mixture, 200), _active(mixture, 500)
        assert target_on.any() and other_on[target_on].all()


def test_short_scene_window_is_the_longest_speaker_and_gaps_come_from_the_config():
    target = _tone(200, 0.5)
    mixture = _scene("short", [target], [[_tone(500, 1.0), _tone(500, 1.0)]], short_gap_seconds=(0.5, 0.5))
    assert len(mixture) == int(2.5 * SR)
    with pytest.raises(ValueError):
        _scene("bogus", [target], [])


def test_scene_without_a_target_levels_the_others_against_the_first():
    first, second = _tone(500, 1.0, 0.4), _tone(700, 1.0, 0.4)
    mixture = _scene("burst", [], [[first], [second]], seed=1, burst_gap_seconds=(0, 0))
    assert len(mixture) == SR
    assert _active(mixture, 500).any() and _active(mixture, 700).any() and not _active(mixture, 200).any()


def _full(targets, others, gaps, seed=0):
    return build_scene(random.Random(seed), "full", targets, others, OnlineMixConfig(speaker_rir_prob=0, sir_db=(0, 0)),
                       rir_loader=None, sampling_rate=SR, gaps=gaps)


def test_full_scene_gives_each_speaker_its_own_silence_and_lets_the_target_speak_repeatedly():
    targets = [_tone(200, 1.0), _tone(300, 1.0), _tone(400, 1.0)]
    others = [[_tone(500, 1.0)] * 6, [_tone(700, 1.0)] * 2]
    gaps = {"target": (2.0, 2.0), "others": [(0.5, 0.5), (3.0, 3.0)]}
    for seed in range(10):
        mixture = _full(targets, others, gaps, seed)
        # the longest track (6 x 1 s + 5 x 0.5 s) sets the window
        assert len(mixture) == int(8.5 * SR)
        on = [np.flatnonzero(_active(mixture, f)) for f in (200, 300, 400)]
        assert all(len(frames) for frames in on)  # the target speaks all three times
        # target utterances keep their order, each separated by the 2 s silence (frames are 0.05 s)
        assert on[0].max() < on[1].min() < on[1].max() < on[2].min()
        assert on[1].min() - on[0].max() >= 2.0 * SR / 800 - 2


def test_full_scene_needs_a_gap_range_for_every_speaker():
    with pytest.raises(ValueError):
        _full([_tone(200, 1.0)], [[_tone(500, 1.0)]], None)
    with pytest.raises(ValueError):
        _full([_tone(200, 1.0)], [[_tone(500, 1.0)]], {"target": (0, 1), "others": []})


def test_full_scene_seconds_uses_the_gaps_the_record_names():
    from open_audio_llm.data.ts_online import scene_seconds

    config = OnlineMixConfig(speaker_rir_prob=0, sir_db=(0, 0))
    targets = [_tone(200, 1.0), _tone(300, 1.5)]
    others = [[_tone(500, 1.0)] * 3, [_tone(700, 1.0), _tone(900, 2.0)]]
    gaps = {"target": (0.2, 4.0), "others": [(0.0, 0.5), (1.0, 6.0)]}
    scene = {"mode": "full", "target_gap": gaps["target"],
             "targets": [{"duration": len(t) / SR} for t in targets],
             "interferers": [{"gap": g, "utts": [{"duration": len(u) / SR} for u in o]} for g, o in zip(gaps["others"], others)]}
    bound = scene_seconds(scene, config)
    lengths = [len(_full(targets, others, gaps, seed)) / SR for seed in range(40)]
    assert max(lengths) <= bound + 1e-3
    assert max(lengths) > bound - 1.0


@pytest.mark.parametrize("mode", ["alternate", "burst", "short"])
def test_scene_seconds_bounds_the_mixture_the_scene_builds(mode):
    from open_audio_llm.data.ts_online import scene_seconds

    config = OnlineMixConfig(speaker_rir_prob=0, sir_db=(0, 0))
    targets = [_tone(200, 1.0), _tone(300, 1.5)]
    others = [[_tone(500, 1.0)], [_tone(700, 1.0), _tone(900, 2.0)]]
    scene = {"mode": mode,
             "targets": [{"duration": len(t) / SR} for t in targets],
             "interferers": [{"utts": [{"duration": len(u) / SR} for u in o]} for o in others]}
    bound = scene_seconds(scene, config)
    lengths = [len(build_scene(random.Random(seed), mode, targets, others, config, rir_loader=None, sampling_rate=SR)) / SR
               for seed in range(30)]
    assert max(lengths) <= bound + 1e-3
    assert max(lengths) > bound - 1.0  # a bound, not a loose multiple


def test_activity_codes_mark_who_speaks_in_each_10ms_frame():
    # 1000 samples are 7 frames of 160; a span covers every frame it touches.
    codes = activity_codes(1000, [(0, 320)], [(300, 500), (900, 2000), (-50, -10)], SR)
    assert codes.tolist() == [1, 3, 2, 2, 0, 2, 2]
    assert activity_codes(1000, [], [], SR).tolist() == [0] * 7


def test_mixture_info_names_where_the_target_and_each_interferer_speak():
    config = OnlineMixConfig(interferer_probs=(0, 0, 1), speaker_rir_prob=0, entry_prob=1.0,
                             lead_seconds=(0.5, 0.5), trail_seconds=(0.5, 0.5))
    for seed in range(10):
        info = {}
        mixture, count = _impulse_free(config, info=info, seed=seed)
        lead = int(info["lead"] * SR)
        assert info["target_spans"] == [(lead, lead + 2 * SR)]
        others = mixture.copy()
        others[lead : lead + 2 * SR] -= _tone(200, 2.0)
        sounding = np.zeros(len(mixture), dtype=bool)
        for begin, end in info["other_spans"]:
            sounding[begin:end] = True
        assert count == 2 and np.array_equal(np.abs(others) > 1e-6, sounding & (np.abs(others) > 1e-6))
        assert not np.abs(others[~sounding]).max(initial=0) > 1e-6  # nothing else sounds
        # Recording the spans draws nothing from the random stream.
        again, _ = _impulse_free(config, seed=seed)
        assert np.array_equal(mixture, again)
    info = {}
    _impulse_free(config, info=info, absent=True)
    assert info["target_spans"] == []


def test_scene_info_spans_follow_each_track():
    target = _tone(200, 1.0)
    for mode, others in (("alternate", [[_tone(500, 1.0)], [_tone(700, 0.5)]]), ("burst", [[_tone(500, 1.0)] * 3])):
        info = {}
        mixture = build_scene(random.Random(1), mode, [target], others, OnlineMixConfig(speaker_rir_prob=0, sir_db=(0, 0)),
                              rir_loader=None, sampling_rate=SR, info=info)
        codes = activity_codes(len(mixture), info["target_spans"], info["other_spans"], SR)
        frames = len(mixture) // 160
        tone_on = lambda f: _active(mixture, f, frame=160)
        for bit, heard, spans in ((1, tone_on(200), info["target_spans"]),
                                  (2, np.any([tone_on(f) for f in (500, 700)], axis=0), info["other_spans"])):
            marked = codes[:frames] & bit > 0
            assert marked[heard].all()  # every frame where the speaker sounds is marked
            assert (marked & ~heard).sum() <= 2 * len(spans)  # extra marks only where a span starts or ends mid-frame
        assert (3 in codes) == (mode == "burst")  # the burst target always has company

