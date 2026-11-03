"""Render Catalog ASR facts in the native Qwen3-ASR conversation protocol."""

from audio_data_contract import AudioRecord

from open_audio_llm.tsasr.ts_prompt import TS_CONCAT_SYSTEM

from .sot import TIMESTAMP_FORMAT, TIMESTAMP_SYSTEM
from .target_sot import TARGET_FORMAT, target_prompt

# Whisper Mel hop (160 samples at 16 kHz).
MEL_HOP_SECONDS = 0.01
SOT_SYSTEM = (
    "Transcribe every speaker in the mixture. Output one line per speaker as "
    "[S1] text, [S2] text, and so on, ordered by when each speaker first speaks. "
    "Keep all utterances from the same speaker on the same line. "
    "Include every speaker, including overlapping speech."
)


def native_language(language: str) -> str:
    """Map Catalog language aliases to native Qwen3-ASR language labels."""
    return {
        "en": "English", "english": "English", "zh": "Chinese",
        "zh-cn": "Chinese", "chinese": "Chinese", "mandarin": "Chinese",
        # KeSpeech stores Chinese regional accents in supervision.language.
        "northeastern": "Chinese", "jiang-huai": "Chinese",
        "southwestern": "Chinese", "jiao-liao": "Chinese", "beijing": "Chinese",
        "zhongyuan": "Chinese", "ji-lu": "Chinese", "lan-yin": "Chinese",
    }.get(language.lower(), language)


def _wav_bytes(waveform, sampling_rate):
    from io import BytesIO

    import soundfile as sf

    buffer = BytesIO()
    sf.write(buffer, waveform, sampling_rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def _clip_samples(value, sampling_rate):
    """Return one mono waveform. Arrays are already at the training rate."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value, dtype=np.float32).reshape(-1), True
    from io import BytesIO

    import soundfile as sf

    audio, rate = sf.read(BytesIO(value), dtype="float32", always_2d=False)
    if int(rate) != int(sampling_rate):
        raise ValueError("TS-ASR audio must be resampled before the encoder")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return np.ascontiguousarray(audio.reshape(-1)), False


def prepare_ts_clips(audios, sampling_rate):
    """Keep enrollment and mixture as separate clips at the lengths passed in.

    The template extracts Mel and runs conv on each clip, then inserts SEP.
    Enrollment is not cropped or zero-padded. Arrays skip the WAV round-trip
    that the training path still uses for ms-swift's bytes loader.
    """
    import numpy as np

    enrollment, enroll_array = _clip_samples(audios[0], sampling_rate)
    mixture, mix_array = _clip_samples(audios[1], sampling_rate)
    # Serving pads each clip to whole Mel hops before extraction (vLLM pads the
    # clip, the TS service requires a hop-aligned enrollment); without the same
    # padding training would see one frame fewer at each partial hop.
    hop = int(sampling_rate * MEL_HOP_SECONDS)
    enrollment = np.pad(enrollment, (0, -len(enrollment) % hop))
    mixture = np.pad(mixture, (0, -len(mixture) % hop))
    if enroll_array and mix_array:
        return enrollment, mixture
    return _wav_bytes(enrollment, sampling_rate), _wav_bytes(mixture, sampling_rate)


def native_messages(record: AudioRecord, hotwords: str) -> list[dict[str, str]]:
    if record.task == "ts_asr":
        if [s.name for s in record.audio_slots] != ["enrollment", "mixture"]:
            raise ValueError("Native TS-ASR requires enrollment then mixture slots")
    elif record.task == "speaker_attributed_asr":
        count = len(record.audio_slots) - 1
        expected = ([f"enrollment_{i + 1}" for i in range(count)] + ["mixture"]
                    if record.metadata.get("sot_output_format") == TARGET_FORMAT else ["mixture"])
        if [slot.name for slot in record.audio_slots] != expected:
            raise ValueError("Speaker-attributed ASR requires one mixture slot")
    elif record.task not in {"asr", "asr_hotwords"} or len(record.audio_slots) != 1:
        raise ValueError("Native Qwen3-ASR training supports single-audio ASR/hotword tasks")
    mixed = record.task == 'speaker_attributed_asr' and record.language == 'zh-en'
    if mixed:
        # Keep the native single-language header; the record retains both languages.
        primary = record.metadata.get('primary_language')
        if primary not in ('zh', 'en'):
            raise ValueError('Mixed SOT needs a zh/en primary_language in metadata')
        language = native_language(primary)
    else:
        language = native_language(record.language)
    # Reject unknown language codes rather than teaching them as output tokens.
    from qwen_asr.inference.utils import validate_language

    if record.target.strip():
        validate_language(language)
    else:
        language = "None"
    context = TS_CONCAT_SYSTEM if record.task == "ts_asr" else ""
    if record.task == "speaker_attributed_asr":
        context = (TIMESTAMP_SYSTEM if record.metadata.get("sot_output_format") == TIMESTAMP_FORMAT
                   else SOT_SYSTEM)
        if record.metadata.get("sot_output_format") == TARGET_FORMAT:
            context = target_prompt(len(record.audio_slots) - 1, record.metadata["enrollment_view"]["mode"])
        elif mixed:
            context += ' Keep each speaker\'s original language, including Chinese and English. Do not translate.'
    if record.task == "asr_hotwords" and hotwords != "N/A":
        context = f"Hotwords: {hotwords}"
    return [
        {"role": "system", "content": context},
        {"role": "user", "content": "<audio>"},
        {"role": "assistant", "content": f"language {language}<asr_text>{record.target}"},
    ]
