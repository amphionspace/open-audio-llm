"""Render Catalog ASR facts in the native Qwen3-ASR conversation protocol."""

from audio_data_contract import AudioRecord

from open_audio_llm.tsasr.ts_prompt import TS_CONCAT_SYSTEM

from .sot import TIMESTAMP_FORMAT, TIMESTAMP_SYSTEM
from .target_sot import TARGET_FORMAT, target_prompt

# v3: enroll is 3s; mixture stays a separate clip. No waveform silence.
ENROLL_SECONDS = 3
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


def prepare_ts_clips(audios, sampling_rate):
    """Crop enrollment to 3s and keep mixture as its own clip.

    The template extracts Mel and runs conv on each clip, then inserts SEP.
    """
    from io import BytesIO

    import numpy as np
    import soundfile as sf

    enrollment, rate = sf.read(BytesIO(audios[0]), dtype="float32")
    mixture, mix_rate = sf.read(BytesIO(audios[1]), dtype="float32")
    if rate != sampling_rate or mix_rate != sampling_rate:
        raise ValueError("TS-ASR audio must be resampled before the encoder")
    count = ENROLL_SECONDS * sampling_rate
    enrollment = np.asarray(enrollment, dtype=np.float32).reshape(-1)
    mixture = np.asarray(mixture, dtype=np.float32).reshape(-1)
    enrollment = np.pad(enrollment[:count], (0, max(0, count - len(enrollment))))
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
