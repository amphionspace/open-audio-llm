"""Audio decoder copied from the established synthesis pipeline (main ad4e9db)."""
import math
from pathlib import Path
import numpy as np
import soundfile as sf
FRAME_SECONDS = 0.02
ACTIVITY_RELATIVE_DB = -35
MP3_FRAME_SAMPLES = 1152


def activity(audio, rate):
    width = round(rate * FRAME_SECONDS)
    padded = np.pad(audio, (0, (-len(audio)) % width))
    rms = np.sqrt(np.mean(padded.reshape(-1, width).astype(np.float64) ** 2, axis=1))
    threshold = max(float(rms.max()) * 10 ** (ACTIVITY_RELATIVE_DB / 20), 1e-5)
    return rms > threshold


def load_audio(row, roots, rate):
    path = row.get('resolved_audio') or Path(roots[row['audio']['root_alias']]) / row['audio']['relative_path']
    with sf.SoundFile(path) as stream:
        if stream.samplerate != row['sample_rate']:
            raise ValueError(f"Source rate changed: {row['source_id']}")
        stream.seek(round(row['start'] * stream.samplerate))
        count = round(row['duration'] * stream.samplerate)
        waveform = stream.read(count, dtype='float32', always_2d=True)
        # Some MP3 headers overstate the complete decoded length by < one frame.
        # Accept the actual EOF only for a whole-file read, never a truncated segment.
        whole_mp3 = stream.format == 'MP3' and row['start'] == 0 and abs(count - stream.frames) <= 1
        allowance = MP3_FRAME_SAMPLES if whole_mp3 else 1
        if len(waveform) < count - allowance:
            raise ValueError(f"Short audio: {row['source_id']}")
        waveform = waveform[:, row['channel']]
    if row['sample_rate'] != rate:
        from scipy.signal import resample_poly

        divisor = math.gcd(row['sample_rate'], rate)
        waveform = resample_poly(waveform, rate // divisor, row['sample_rate'] // divisor)
    if not np.isfinite(waveform).all() or not activity(waveform, rate).any():
        raise ValueError(f"Invalid or silent audio: {row['source_id']}")
    return waveform
