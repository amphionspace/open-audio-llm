from dataclasses import replace

import pytest
from audio_data_contract import AudioRecord, AudioRef, AudioSlot

from open_audio_llm.data.qwen3_asr import native_messages

pytest.importorskip("qwen_asr")


def record(language="en", target="Hello Alice."):
    return AudioRecord(
        id="test", task="asr_hotwords", language=language, target=target,
        audio_slots=(AudioSlot("primary", AudioRef("test", "1", "train", "1")),),
        hotwords=("Alice",),
    )


def test_native_context_and_labels():
    messages = native_messages(record(), "Alice,Bob")
    assert messages == [
        {"role": "system", "content": "Hotwords: Alice,Bob"},
        {"role": "user", "content": "<audio>"},
        {"role": "assistant", "content": "language English<asr_text>Hello Alice."},
    ]
    assert native_messages(record("zh", "你好。"), "N/A")[-1]["content"] == "language Chinese<asr_text>你好。"
    assert native_messages(record(), "N/A")[0]["content"] == ""
    assert native_messages(replace(record(), task="asr"), "Alice")[0]["content"] == ""


def test_all_speaker_prompt_keeps_one_mixture_and_complete_speaker_labels():
    from open_audio_llm.data.qwen3_asr import SOT_SYSTEM

    sot = replace(record('zh', '[S1] 你好\n[S2] 再见\n[S3] 谢谢'), task='speaker_attributed_asr',
                  audio_slots=(AudioSlot('mixture', AudioRef('sot', '1', 'train', 'audio')),))
    messages = native_messages(sot, 'ignored')
    assert messages[0]['content'] == SOT_SYSTEM
    assert messages[1]['content'] == '<audio>'
    assert messages[2]['content'] == 'language Chinese<asr_text>' + sot.target
    with pytest.raises(ValueError, match='one mixture'):
        native_messages(replace(sot, audio_slots=record().audio_slots), 'N/A')


@pytest.mark.parametrize("language", ["Mandarin", "Southwestern"])
def test_native_kespeech_catalog_language(language):
    messages = native_messages(record(language, "你好世界"), "N/A")
    assert messages[-1]["content"] == "language Chinese<asr_text>你好世界"


def test_silence_and_unsupported_task():
    assert native_messages(record(target=""), "N/A")[-1]["content"] == "language None<asr_text>"
    with pytest.raises(ValueError, match="single-audio"):
        native_messages(replace(record(), task="ser"), "N/A")


def test_ts_prompt_and_negative_keep_native_task_boundary():
    from open_audio_llm.data.qwen3_asr import TS_CONCAT_SYSTEM

    ts = replace(record("zh", "你好"), task="ts_asr", audio_slots=tuple(
        AudioSlot(name, AudioRef("test", "1", "train", name))
        for name in ("enrollment", "mixture")
    ))
    messages = native_messages(ts, "ignored")
    assert messages[0]["content"] == TS_CONCAT_SYSTEM
    assert messages[1]["content"] == "<audio>"
    assert messages[2]["content"] == "language Chinese<asr_text>你好"
    assert native_messages(replace(ts, target=""), "N/A")[-1]["content"] == "language None<asr_text>"
    with pytest.raises(ValueError, match="enrollment then mixture"):
        native_messages(replace(ts, audio_slots=ts.audio_slots[::-1]), "N/A")


@pytest.mark.parametrize("enroll_seconds", [1, 4])
def test_ts_concat_crops_or_pads_enrollment_without_truncating_mixture(enroll_seconds):
    from io import BytesIO

    import numpy as np
    import soundfile as sf

    from open_audio_llm.data.qwen3_asr import concat_ts_audio

    audios = []
    for seconds, value in [(enroll_seconds, 0.25), (2, 0.5)]:
        buffer = BytesIO()
        sf.write(buffer, np.full(16000 * seconds, value), 16000, format="WAV", subtype="FLOAT")
        audios.append(buffer.getvalue())
    audio, rate = sf.read(BytesIO(concat_ts_audio(audios, 16000)))
    assert rate == 16000 and len(audio) == 8 * rate
    assert np.all(audio[:min(enroll_seconds, 3) * rate] == 0.25)
    assert np.all(audio[min(enroll_seconds, 3) * rate:6 * rate] == 0)
    assert np.all(audio[6 * rate:] == 0.5)
