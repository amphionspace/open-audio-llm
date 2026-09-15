"""Render Catalog ASR facts in the native Qwen3-ASR conversation protocol."""

from audio_data_contract import AudioRecord

# Matches the reference TS-ASR recipe; training and evaluation share these.
ENROLL_SECONDS = 3
SILENCE_SECONDS = 3
TS_CONCAT_SYSTEM = (
    "The clip contains a 3-second enrollment, 3 seconds of silence, "
    "and then a mixture. Transcribe only the enrolled speaker from the mixture. "
    "If the enrolled speaker is not present, output nothing."
)
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


def concat_ts_audio(audios, sampling_rate):
    """Convert enrollment/mixture WAV bytes to one native encoder input."""
    from io import BytesIO

    import numpy as np
    import soundfile as sf

    enrollment, rate = sf.read(BytesIO(audios[0]), dtype="float32")
    mixture, mix_rate = sf.read(BytesIO(audios[1]), dtype="float32")
    if rate != sampling_rate or mix_rate != sampling_rate:
        raise ValueError("TS-ASR audio must be resampled before concatenation")
    count = ENROLL_SECONDS * sampling_rate
    enrollment = np.pad(enrollment[:count], (0, max(0, count - len(enrollment))))
    waveform = np.concatenate(
        [enrollment, np.zeros(SILENCE_SECONDS * sampling_rate, dtype=np.float32), mixture]
    )
    buffer = BytesIO()
    sf.write(buffer, waveform, sampling_rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def native_messages(record: AudioRecord, hotwords: str) -> list[dict[str, str]]:
    if record.task == "ts_asr":
        if [s.name for s in record.audio_slots] != ["enrollment", "mixture"]:
            raise ValueError("Native TS-ASR requires enrollment then mixture slots")
    elif record.task == "speaker_attributed_asr":
        if [slot.name for slot in record.audio_slots] != ["mixture"]:
            raise ValueError("Speaker-attributed ASR requires one mixture slot")
    elif record.task not in {"asr", "asr_hotwords"} or len(record.audio_slots) != 1:
        raise ValueError("Native Qwen3-ASR training supports single-audio ASR/hotword tasks")
    language = native_language(record.language)
    # Reject unknown language codes rather than teaching them as output tokens.
    from qwen_asr.inference.utils import validate_language

    if record.target.strip():
        validate_language(language)
    else:
        language = "None"
    context = TS_CONCAT_SYSTEM if record.task == "ts_asr" else ""
    if record.task == "speaker_attributed_asr":
        context = SOT_SYSTEM
    if record.task == "asr_hotwords" and hotwords != "N/A":
        context = f"Hotwords: {hotwords}"
    return [
        {"role": "system", "content": context},
        {"role": "user", "content": "<audio>"},
        {"role": "assistant", "content": f"language {language}<asr_text>{record.target}"},
    ]
