import pytest

pytest.importorskip("rapidfuzz")

from open_audio_llm.scripts.grpo_mining import score


def item(system, solution):
    return {"system": system, "solution": solution}


def test_keeps_only_presence_or_hotword_variation():
    empty = item("", "language None<asr_text>")
    assert score(empty, ["language None<asr_text>", "language Chinese<asr_text>嗯"])[1]
    assert not score(empty, ["language None<asr_text>"] * 2)[1]
    speech = item("", "language Chinese<asr_text>你好世界")
    assert not score(speech, ["language Chinese<asr_text>你好世界", "language Chinese<asr_text>你好视界"])[1]
    listed = item("Hotwords: 张伟,章玮", "language Chinese<asr_text>找张伟")
    assert score(listed, ["language Chinese<asr_text>找张伟", "language Chinese<asr_text>找章玮"])[1]


def test_ts_prompts_go_to_the_ts_server_with_their_enrollment_split():
    from types import SimpleNamespace

    from open_audio_llm.scripts.grpo_mining import payload

    args = SimpleNamespace(samples=8, temperature=1.0, max_tokens=512, server="http://asr", model="asr",
                           ts_server="http://ts", ts_model="ts")
    plain = {"source": "a", "system": "", "audio": "x", "enroll_seconds": None}
    server, body = payload(args, plain)
    assert (server, body["model"]) == ("http://asr", "asr") and "mm_processor_kwargs" not in body
    ts = {**plain, "enroll_seconds": 4.8100000001}
    server, body = payload(args, ts)
    assert (server, body["model"]) == ("http://ts", "ts")
    assert body["mm_processor_kwargs"] == {"enroll_seconds": 4.81}
    with pytest.raises(ValueError, match="ts_server"):
        payload(SimpleNamespace(**{**vars(args), "ts_server": None}), ts)


def test_ts_candidate_sends_enrollment_then_mixture_as_one_clip():
    import base64
    import io

    import numpy as np
    import soundfile as sf

    from open_audio_llm.data.qwen3_asr import _wav_bytes
    from open_audio_llm.scripts.grpo_mining import Candidates

    rate = 16000
    enrollment, mixture = np.full(480, 0.25, np.float32), np.full(320, -0.5, np.float32)

    class Dataset:
        sampling_rate = rate

        def __getitem__(self, key):
            return {"audios": [_wav_bytes(enrollment, rate)], "mix_wav": _wav_bytes(mixture, rate),
                    "messages": [{"role": "system", "content": "TS"}], "solution": "language None<asr_text>"}

    item = Candidates(Dataset(), [{"source": "s", "index": [0, 1, 2], "record_id": "r"}])[0]
    assert item["enroll_seconds"] == pytest.approx(0.03) and item["system"] == "TS"
    audio, sr = sf.read(io.BytesIO(base64.b64decode(item["audio"])), dtype="float32")
    assert sr == rate and np.array_equal(audio, np.concatenate([enrollment, mixture]))
