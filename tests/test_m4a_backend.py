import shutil
import subprocess
import time

import numpy as np
import pytest
from lhotse import AudioSource, Recording
from lhotse.audio.backend import CompositeAudioBackend, get_current_audio_backend
from lhotse.audio.utils import AudioLoadingError

from open_audio_llm.data import m4a_backend
from open_audio_llm.data.m4a_backend import TimeoutFfmpegM4aBackend, install_m4a_backend


def _recording(path, sampling_rate, seconds, channels):
    return Recording(id="clip", sources=[AudioSource(type="file", channels=list(range(channels)), source=str(path))],
                     sampling_rate=sampling_rate, num_samples=sampling_rate * seconds, duration=seconds)


def test_hung_ffmpeg_becomes_a_retryable_loading_error(tmp_path, monkeypatch):
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\nexec sleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setattr(m4a_backend.shutil, "which", lambda name: str(fake))
    monkeypatch.setattr(m4a_backend, "FFMPEG_TIMEOUT_S", 0.5)
    monkeypatch.setattr(m4a_backend, "FFMPEG_REAP_S", 0.5)
    install_m4a_backend()
    started = time.monotonic()
    with pytest.raises(AudioLoadingError, match="within 0.5 s"):
        _recording(tmp_path / "stuck.m4a", 44100, 10, 1).load_audio(offset=2.0, duration=1.0)
    assert time.monotonic() - started < 5


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg with an AAC encoder")
def test_lhotse_reads_m4a_through_the_seeking_backend(tmp_path):
    path = tmp_path / "tone.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
                    "-f", "lavfi", "-i", "sine=frequency=660:duration=6", "-filter_complex", "amerge=inputs=2",
                    "-ar", "44100", "-c:a", "aac", str(path)], check=True)
    install_m4a_backend()
    install_m4a_backend()
    backends = get_current_audio_backend().backends
    assert isinstance(get_current_audio_backend(), CompositeAudioBackend)
    assert sum(isinstance(backend, TimeoutFfmpegM4aBackend) for backend in backends) == 1
    audio = _recording(path, 44100, 6, 2).load_audio(offset=2.5, duration=1.5)
    assert audio.shape == (2, int(1.5 * 44100))
    expected = 0.125 * np.sin(2 * np.pi * 440 * (2.5 + np.arange(audio.shape[1]) / 44100))
    # AAC is lossy and needs a few frames after the seek; compare the settled part.
    np.testing.assert_allclose(audio[0, 4096:], expected[4096:], atol=0.02)
