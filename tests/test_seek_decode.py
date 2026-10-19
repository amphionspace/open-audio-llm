import numpy as np
import pytest

av = pytest.importorskip("av")

from lhotse import AudioSource, Recording  # noqa: E402

from open_audio_llm.data.catalog_dataset import _seek_decode  # noqa: E402

RATE = 44100


def write_m4a(path, channels, seconds):
    # Tones only: AAC codes noise-like bands with a pseudo-random generator (PNS),
    # whose state after a seek differs from a decode that started at the file head.
    t = np.arange(int(seconds * RATE)) / RATE
    tones = sum(np.sin(2 * np.pi * f * t + p) for f, p in ((220, 0.0), (440, 1.0), (1330, 2.0)))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)
    audio = np.stack([0.1 * envelope * tones * (1 + 0.2 * c) for c in range(channels)]).astype(np.float32)
    layout = "stereo" if channels == 2 else "mono"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("aac", rate=RATE, layout=layout)
        for start in range(0, t.size, 1024):
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(audio[:, start:start + 1024]), format="fltp", layout=layout)
            frame.sample_rate = RATE
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return t.size


def decode_all(path):
    """Reference: decode from the file start, place samples by their timestamps."""
    chunks, first = [], None
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            if first is None:
                first = int(round(frame.pts * stream.time_base * RATE))
            chunks.append(frame.to_ndarray())
    return np.concatenate(chunks, axis=1), first


@pytest.mark.parametrize("channels", [1, 2])
def test_seek_decode_matches_decoding_from_start(tmp_path, channels):
    path = tmp_path / "speech.m4a"
    samples = write_m4a(path, channels, 20)
    recording = Recording(id="speech", sources=[AudioSource("file", list(range(channels)), str(path))],
                          sampling_rate=RATE, num_samples=samples, duration=samples / RATE)
    full, origin = decode_all(path)
    for offset, duration in ((0.0, 3.0), (7.31, 5.2), (14.0, 5.9)):
        cut = recording.to_cut().truncate(offset=offset, duration=duration)
        got = _seek_decode(cut)
        start = int(round(offset * RATE)) - origin
        np.testing.assert_allclose(got, full[:, start:start + cut.num_samples], atol=1e-6)


def test_seek_decode_leaves_wav_to_lhotse(tmp_path):
    recording = Recording(id="x", sources=[AudioSource("file", [0], str(tmp_path / "x.wav"))],
                          sampling_rate=16000, num_samples=16000, duration=1.0)
    assert _seek_decode(recording.to_cut().truncate(offset=0.2, duration=0.5)) is None


def test_load_mono_prefers_the_transcoded_flac_and_falls_back_to_the_m4a(tmp_path):
    import soundfile as sf

    from open_audio_llm.data.catalog_dataset import load_mono

    path = tmp_path / "dialog.m4a"
    samples = write_m4a(path, 2, 6)
    recording = Recording(id="dialog", sources=[AudioSource("file", [0, 1], str(path))],
                          sampling_rate=RATE, num_samples=samples, duration=samples / RATE)
    # A different tone in the FLAC shows which file was read.
    flac = 0.25 * np.sin(2 * np.pi * 1000 * np.arange(6 * 16000) / 16000).astype(np.float32)
    sf.write(tmp_path / "dialog.flac", flac, 16000, subtype="PCM_16")
    cut = recording.to_cut().truncate(offset=2.0, duration=1.5)

    np.testing.assert_allclose(load_mono(cut, 16000), flac[32000:56000], atol=1e-4)
    # A channel view needs the original microphone, not the downmix.
    m4a = load_mono(cut.with_channels(0), 16000)
    assert np.abs(m4a - flac[32000:56000][:len(m4a)]).max() > 0.1
    # A FLAC shorter than the cut (still being written) is not trusted.
    sf.write(tmp_path / "dialog.flac", flac[:40000], 16000, subtype="PCM_16")
    fallback = load_mono(cut, 16000)
    assert np.abs(fallback - flac[32000:56000][:len(fallback)]).max() > 0.1


def test_a_cut_ending_just_past_the_flac_is_padded_without_the_m4a(tmp_path):
    import soundfile as sf
    from lhotse.audio.utils import get_audio_duration_mismatch_tolerance

    from open_audio_llm.data.catalog_dataset import transcoded_flac

    # The m4a is gone; the index declares 6 s, the FLAC decoded to 12 frames less.
    flac = 0.25 * np.sin(2 * np.pi * 1000 * np.arange(6 * 16000 - 12) / 16000).astype(np.float32)
    sf.write(tmp_path / "dialog.flac", flac, 16000, subtype="PCM_16")
    recording = Recording(id="dialog", sources=[AudioSource("file", [0, 1], str(tmp_path / "dialog.m4a"))],
                          sampling_rate=RATE, num_samples=6 * RATE, duration=6.0)
    cut = recording.to_cut().truncate(offset=4.0, duration=2.0)

    audio = transcoded_flac(cut, 16000)
    assert audio.shape == (32000,)
    np.testing.assert_allclose(audio[:-12], flac[64000:], atol=1e-4)
    np.testing.assert_allclose(audio[-12:], flac[-2:-14:-1], atol=1e-4)
    # Beyond lhotse's mismatch tolerance the FLAC is not trusted.
    short = 6 * 16000 - int(get_audio_duration_mismatch_tolerance() * 16000) - 1
    sf.write(tmp_path / "dialog.flac", flac[:short], 16000, subtype="PCM_16")
    assert transcoded_flac(cut, 16000) is None
