"""TS-ASR waveforms: enroll 3s and mix are independent until the transformer.

Mel and conv run on each clip separately. A learned [SEP] is concatenated
only when packing tokens into the audio transformer.
`max_samples=0` means do not truncate. Qwen3-ASR still passes 480000 (30s).
"""
from __future__ import annotations

import base64
import io
import os
import warnings
from typing import Optional

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
ENROLL_SEC = 3.0
# v3: no waveform silence; keep the name so old call sites still import.
SILENCE_SEC = 0.0
# Whisper feature extractor on Qwen3-ASR-0.6B: n_samples = 480000 (30s).
MAX_SAMPLES = 480000
_ENV_ENROLL_SEC = "AMPHION_TSASR_ENROLL_SEC"
_ENV_NOPAD = "AMPHION_TSASR_ENROLL_NOPAD"


def configured_enroll_sec() -> float:
    """Eval-only override. Unset → 3s, same as training."""
    raw = os.environ.get(_ENV_ENROLL_SEC, "").strip()
    if not raw:
        return ENROLL_SEC
    return float(raw)


def enroll_nopad_enabled() -> bool:
    return os.environ.get(_ENV_NOPAD, "").strip().lower() in {"1", "true", "yes", "on"}


def _read_mono(
    path: str,
    *,
    sr: int = SAMPLE_RATE,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
) -> np.ndarray:
    info = sf.info(path)
    start = max(0, int(round(start_sec * info.samplerate)))
    stop = None
    if duration_sec is not None:
        stop = start + int(round(duration_sec * info.samplerate))
        stop = min(stop, info.frames)
    wav, file_sr = sf.read(path, start=start, stop=stop, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if file_sr != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr).astype(np.float32)
    return wav


def _crop_or_pad(wav: np.ndarray, n: int) -> np.ndarray:
    if wav.shape[0] >= n:
        return wav[:n]
    return np.pad(wav, (0, n - wav.shape[0]))


def enroll_offset_sec(enroll_path: str, enroll_sec: float, rng: Optional[np.random.Generator]) -> float:
    if rng is None:
        return 0.0
    try:
        dur = float(sf.info(enroll_path).duration)
    except Exception:
        return 0.0
    slack = dur - enroll_sec
    if slack <= 0:
        return 0.0
    return float(rng.uniform(0.0, slack))


def enroll_n_samples(enroll_sec: float = ENROLL_SEC, sr: int = SAMPLE_RATE) -> int:
    return int(round(enroll_sec * sr))


def enroll_mel_frames(
    hop_length: int = 160,
    *,
    enroll_sec: float = ENROLL_SEC,
    sr: int = SAMPLE_RATE,
    max_length: int = 3000,
) -> int:
    n_frames = enroll_n_samples(enroll_sec, sr) // int(hop_length)
    return min(n_frames, int(max_length))


def load_enroll_wav(
    enroll_path: str,
    *,
    enroll_sec: Optional[float] = None,
    sr: int = SAMPLE_RATE,
    enroll_start_sec: float = 0.0,
) -> np.ndarray:
    if enroll_sec is None:
        enroll_sec = configured_enroll_sec()
    wav = _read_mono(
        enroll_path,
        sr=sr,
        start_sec=enroll_start_sec,
        duration_sec=enroll_sec,
    )
    if enroll_nopad_enabled():
        return wav
    n_enroll = int(round(enroll_sec * sr))
    return _crop_or_pad(wav, n_enroll)


def load_mix_wav(
    mix_path: str,
    *,
    sr: int = SAMPLE_RATE,
    max_samples: int = MAX_SAMPLES,
) -> np.ndarray:
    mix = _read_mono(mix_path, sr=sr)
    if max_samples and mix.shape[0] > max_samples:
        warnings.warn(
            f"load_mix_wav truncated {mix.shape[0]} -> {max_samples} samples "
            f"({max_samples / sr:.1f}s cap)",
            stacklevel=2,
        )
        mix = mix[:max_samples]
    return mix


def split_enroll_mix_wav(
    wav: np.ndarray,
    *,
    enroll_sec: Optional[float] = None,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a concat wav (enroll prefix + mix) back into two clips."""
    if enroll_sec is None:
        enroll_sec = configured_enroll_sec()
    n_enroll = enroll_n_samples(enroll_sec, sr)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.shape[0] <= n_enroll:
        return _crop_or_pad(wav, n_enroll), np.zeros(0, dtype=np.float32)
    return wav[:n_enroll], wav[n_enroll:]


def concat_enroll_mix(
    enroll_path: str,
    mix_path: str,
    *,
    enroll_sec: Optional[float] = None,
    silence_sec: float = SILENCE_SEC,
    sr: int = SAMPLE_RATE,
    max_samples: int = 0,
    enroll_start_sec: float = 0.0,
) -> np.ndarray:
    """Eval/HTTP transport only. Training extracts Mel from each clip separately."""
    if enroll_sec is None:
        enroll_sec = configured_enroll_sec()
    enroll = load_enroll_wav(
        enroll_path,
        enroll_sec=enroll_sec,
        sr=sr,
        enroll_start_sec=enroll_start_sec,
    )
    mix = load_mix_wav(mix_path, sr=sr, max_samples=0)
    n_sil = int(round(silence_sec * sr))
    parts = [enroll]
    if n_sil > 0:
        parts.append(np.zeros(n_sil, dtype=np.float32))
    parts.append(mix)
    wav = np.concatenate(parts, axis=0)
    if max_samples and wav.shape[0] > max_samples:
        warnings.warn(
            f"concat_enroll_mix truncated {wav.shape[0]} -> {max_samples} samples "
            f"({max_samples / sr:.1f}s cap)",
            stacklevel=2,
        )
        wav = wav[:max_samples]
    return wav


def random_enroll_enabled() -> bool:
    return os.environ.get("COT_TSASR_ENROLL_RANDOM", "0").strip() not in ("", "0", "false", "False")


def pack_ts_transport_b64(enroll_b64: str, mix_b64: str) -> str:
    """Pack the served TS-ASR request audio: 3s enrollment immediately followed by the mixture.

    The TS-ASR vLLM service accepts one audio per request, so callers send this
    waveform together with ``ts_prompt.TS_CONCAT_SYSTEM``.
    """
    enroll, enroll_sr = sf.read(io.BytesIO(base64.b64decode(enroll_b64)), dtype="float32")
    mix, mix_sr = sf.read(io.BytesIO(base64.b64decode(mix_b64)), dtype="float32")
    if int(enroll_sr) != SAMPLE_RATE or int(mix_sr) != SAMPLE_RATE:
        raise ValueError("TS-ASR request audio must already be 16 kHz")
    enroll = np.asarray(enroll, dtype=np.float32).reshape(-1)
    mix = np.asarray(mix, dtype=np.float32).reshape(-1)
    count = int(ENROLL_SEC * SAMPLE_RATE)
    enroll = enroll[:count] if enroll.shape[0] >= count else np.pad(enroll, (0, count - enroll.shape[0]))
    buf = io.BytesIO()
    sf.write(buf, np.concatenate([enroll, mix]), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
