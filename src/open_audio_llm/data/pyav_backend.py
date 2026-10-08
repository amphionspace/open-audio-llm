"""Decode AAC/M4A with PyAV for the Lhotse file loader.

soundfile and the torchaudio build in the training env cannot open these
files. The audio index stores the container sample count. This backend
returns that many samples, counted from the start of the decode, so a
record duration of ``num_frames / sample_rate`` matches the 16 kHz wav
the trainer writes within one output sample.
"""

from __future__ import annotations

_INSTALLED = False


def install_pyav_backend() -> bool:
    """Prefer PyAV for m4a/mp4 paths. Other formats keep Lhotse's backends."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    from lhotse.audio.backend import (
        CompositeAudioBackend,
        get_default_audio_backend,
        set_current_audio_backend,
    )

    default = get_default_audio_backend()
    backends = default.backends if isinstance(default, CompositeAudioBackend) else [default]
    if any(isinstance(backend, PyavM4aBackend) for backend in backends):
        _INSTALLED = True
        return True
    set_current_audio_backend(CompositeAudioBackend([PyavM4aBackend(), *backends]))
    _INSTALLED = True
    return True


class PyavM4aBackend:
    """Registered as a Lhotse backend via install; subclassing happens below."""


def _is_m4a(path) -> bool:
    return isinstance(path, str) and path.lower().endswith((".m4a", ".m4b", ".mp4"))


def _header(stream, sample_rate: int) -> int:
    if stream.duration is None or stream.time_base is None or sample_rate <= 0:
        raise RuntimeError("m4a container has no duration")
    if stream.time_base.numerator == 1 and stream.time_base.denominator == sample_rate:
        return int(stream.duration)
    return int(round(float(stream.duration * stream.time_base) * sample_rate))


def _frame_channels(frame) -> int:
    try:
        return int(frame.layout.nb_channels)
    except Exception:
        return 0


def _as_channels_first(frame, channels: int):
    import numpy as np

    array = np.asarray(frame.to_ndarray())
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / np.float32(np.iinfo(array.dtype).max)
    else:
        array = array.astype(np.float32, copy=False)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.shape[0] == channels:
        return array
    if array.shape[-1] == channels:
        return np.swapaxes(array, 0, -1)
    if array.shape[0] == frame.samples:
        return array.T
    raise RuntimeError(f"unexpected PyAV frame shape {array.shape} channels={channels}")


def read_m4a(path: str, start: int, end: int):
    """Return float32 audio shaped (channels, end-start)."""
    import av
    import numpy as np

    container = av.open(path)
    try:
        stream = container.streams.audio[0]
        pieces = []
        seen = 0
        for frame in container.decode(stream):
            channels = _frame_channels(frame) or int(stream.channels or 1)
            array = np.asarray(_as_channels_first(frame, channels), dtype=np.float32)
            count = array.shape[1]
            if seen + count <= start:
                seen += count
                continue
            left = max(0, start - seen)
            right = min(count, end - seen)
            if right > left:
                pieces.append(array[:, left:right])
            seen += count
            if seen >= end:
                break
    finally:
        container.close()
    if not pieces:
        raise RuntimeError(f"decoded no samples from {path} [{start}:{end}]")
    audio = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)
    need = end - start
    if audio.shape[1] > need:
        audio = audio[:, :need]
    return np.ascontiguousarray(audio, dtype=np.float32)


def probe_m4a(path: str):
    import av

    container = av.open(path)
    try:
        stream = container.streams.audio[0]
        sample_rate = int(stream.rate or stream.codec_context.sample_rate)
        channels = int(stream.channels or stream.codec_context.channels or 1)
        frames = _header(stream, sample_rate)
    finally:
        container.close()
    if frames <= 0 or sample_rate <= 0 or channels <= 0:
        raise RuntimeError(f"unusable m4a header: {path}")
    return sample_rate, channels, frames


# Lhotse's AudioBackend is imported lazily so this module can be imported
# without lhotse during the metadata scan.
def _backend_class():
    from lhotse.audio.backend import AudioBackend
    from lhotse.utils import compute_num_samples

    class _Pyav(AudioBackend):
        def handles_special_case(self, path_or_fd) -> bool:
            return _is_m4a(path_or_fd)

        def is_applicable(self, path_or_fd) -> bool:
            return _is_m4a(path_or_fd)

        def supports_info(self) -> bool:
            return True

        def info(self, path_or_fd, force_opus_sampling_rate=None):
            from lhotse.audio.backend import LibsndfileCompatibleAudioInfo

            sample_rate, channels, frames = probe_m4a(path_or_fd)
            return LibsndfileCompatibleAudioInfo(
                channels, frames, sample_rate, frames / sample_rate
            )

        def read_audio(self, path_or_fd, offset=0.0, duration=None, force_opus_sampling_rate=None):
            sample_rate, _channels, frames = probe_m4a(path_or_fd)
            start = 0 if not offset else compute_num_samples(float(offset), sample_rate)
            if duration is None:
                end = frames
            else:
                end = start + compute_num_samples(float(duration), sample_rate)
            start = min(max(start, 0), frames)
            end = min(max(end, start), frames)
            if end <= start:
                raise RuntimeError(f"empty m4a slice {path_or_fd} offset={offset} duration={duration}")
            return read_m4a(path_or_fd, start, end), sample_rate

    _Pyav.__name__ = "PyavM4aBackend"
    return _Pyav


# Rebuild the placeholder as the real backend once lhotse is importable.
def _bind() -> None:
    global PyavM4aBackend
    try:
        PyavM4aBackend = _backend_class()
    except Exception:
        return


_bind()
