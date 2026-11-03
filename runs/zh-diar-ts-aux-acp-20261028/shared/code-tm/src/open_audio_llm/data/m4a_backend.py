"""Decode m4a with an ffmpeg subprocess that cannot hang a DataLoader worker."""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
from lhotse.audio.backend import (
    AudioBackend,
    CompositeAudioBackend,
    get_current_audio_backend,
    set_current_audio_backend,
)

# A seeked training segment decodes in well under a second. Reads from /workspace
# have stalled for 10 s and more (ACP 003); audioread then waits without a limit
# while probing the stream or reaping ffmpeg, which hung one rank in ACP 011/014.
FFMPEG_TIMEOUT_S = 60
# A killed ffmpeg is reaped briefly; one stuck in uninterruptible I/O is abandoned.
FFMPEG_REAP_S = 5


class TimeoutFfmpegM4aBackend(AudioBackend):
    """Seek with ffmpeg before decoding and give up after ``FFMPEG_TIMEOUT_S``.

    lhotse would otherwise fall back to audioread, which decodes from the file
    start and has no timeout on its stream probe or process reaping. A timeout
    surfaces as ``AudioLoadingError``, which ``load_mono`` retries.
    """

    def read_audio(self, path_or_fd, offset=0.0, duration=None, force_opus_sampling_rate=None):
        import soundfile as sf

        command = [shutil.which("ffmpeg") or "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                   "-threads", "1", "-ss", str(offset)]
        if duration is not None:
            command += ["-t", str(duration)]
        command += ["-i", str(path_or_fd), "-map_metadata", "-1", "-f", "wav", "-acodec", "pcm_s16le", "-"]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        try:
            stdout, stderr = process.communicate(timeout=FFMPEG_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.communicate(timeout=FFMPEG_REAP_S)
            except subprocess.TimeoutExpired:
                pass
            raise TimeoutError(f"ffmpeg did not decode {path_or_fd} within {FFMPEG_TIMEOUT_S} s")
        if process.returncode:
            raise RuntimeError(f"ffmpeg failed on {path_or_fd}: {stderr.decode(errors='replace')[-500:]}")
        # Same int16 samples and channel-first layout as lhotse's audioread path.
        audio, sampling_rate = sf.read(io.BytesIO(stdout), dtype="float32", always_2d=True)
        return np.ascontiguousarray(audio.T), sampling_rate

    def handles_special_case(self, path_or_fd):
        return isinstance(path_or_fd, (str, Path)) and str(path_or_fd).lower().endswith(".m4a")

    def is_applicable(self, path_or_fd):
        return self.handles_special_case(path_or_fd)


def install_m4a_backend():
    """Route lhotse's m4a reads through ``TimeoutFfmpegM4aBackend`` in this process."""
    current = get_current_audio_backend()
    backends = list(current.backends) if isinstance(current, CompositeAudioBackend) else [current]
    if not any(isinstance(backend, TimeoutFfmpegM4aBackend) for backend in backends):
        set_current_audio_backend(CompositeAudioBackend([TimeoutFfmpegM4aBackend(), *backends]))
