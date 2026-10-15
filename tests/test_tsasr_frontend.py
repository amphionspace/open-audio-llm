import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from open_audio_llm.tsasr.sep_token import install_independent_mel_fe


@pytest.fixture
def ts_extractor(monkeypatch):
    monkeypatch.setenv("AMPHION_TSASR_INSERT_SEP", "1")
    monkeypatch.setenv("AMPHION_TSASR_ENROLL_SEC", "3")
    # Restore the class method after exercising the process-wide inference patch.
    monkeypatch.setattr(
        WhisperFeatureExtractor, "__call__", WhisperFeatureExtractor.__call__
    )
    install_independent_mel_fe()
    return WhisperFeatureExtractor(feature_size=128)


def extract(extractor, audio):
    return extractor(
        audio,
        sampling_rate=16000,
        return_attention_mask=True,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )


@pytest.mark.parametrize("mix_samples", [1281, 16000, 16001, 431999])
def test_native_and_vllm_prepadding_produce_identical_mel(ts_extractor, mix_samples):
    wav = np.random.default_rng(7).normal(0, 0.1, 48000 + mix_samples).astype(np.float32)
    # vLLM 0.18's Qwen3-Omni processor pads the entire concat waveform first.
    padded = np.pad(wav, (0, -wav.size % ts_extractor.hop_length))
    native = extract(ts_extractor, [wav])
    served = extract(ts_extractor, [padded])
    torch.testing.assert_close(
        native["input_features"], served["input_features"], rtol=0, atol=0
    )
    assert torch.equal(native["attention_mask"], served["attention_mask"])
    frames = int(native["attention_mask"].sum())
    assert frames == 300 + (mix_samples + 159) // 160


def test_batch_companions_do_not_change_each_clips_features(ts_extractor):
    rng = np.random.default_rng(9)
    wavs = [rng.normal(0, 0.1, 48000 + n).astype(np.float32) for n in (1281, 16001)]
    batch = extract(ts_extractor, wavs)
    for i, wav in enumerate(wavs):
        single = extract(ts_extractor, [wav])
        frames = int(single["attention_mask"].sum())
        torch.testing.assert_close(
            batch["input_features"][i, :, :frames],
            single["input_features"][0],
            rtol=0,
            atol=0,
        )
        assert int(batch["attention_mask"][i].sum()) == frames
        assert not torch.any(batch["attention_mask"][i, frames:])


def test_plain_asr_feature_extraction_is_unchanged(monkeypatch):
    wav = np.random.default_rng(11).normal(0, 0.1, 16001).astype(np.float32)
    extractor = WhisperFeatureExtractor(feature_size=128)
    monkeypatch.setenv("AMPHION_TSASR_INSERT_SEP", "0")
    expected = extract(extractor, [wav])
    monkeypatch.setattr(
        WhisperFeatureExtractor, "__call__", WhisperFeatureExtractor.__call__
    )
    install_independent_mel_fe()
    actual = extract(extractor, [wav])
    for key in ("input_features", "attention_mask"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_request_enrollment_length_moves_the_split(ts_extractor, monkeypatch):
    from open_audio_llm.tsasr.sep_token import REQUEST_ENROLL_SEC

    rng = np.random.default_rng(13)
    enroll = rng.normal(0, 0.1, 116800).astype(np.float32)  # 7.3 s
    mix = rng.normal(0, 0.1, 160000).astype(np.float32)
    token = REQUEST_ENROLL_SEC.set(7.3)
    try:
        served = extract(ts_extractor, [np.concatenate([enroll, mix])])
    finally:
        REQUEST_ENROLL_SEC.reset(token)
    monkeypatch.setenv("AMPHION_TSASR_INSERT_SEP", "0")  # plain extraction of each clip
    parts = [extract(ts_extractor, [clip]) for clip in (enroll, mix)]
    expected = torch.cat([p["input_features"] for p in parts], dim=-1)
    assert served["enroll_feature_lengths"].tolist() == [730]
    torch.testing.assert_close(served["input_features"], expected, rtol=0, atol=0)


def test_default_split_keeps_the_native_outputs(ts_extractor):
    wav = np.random.default_rng(17).normal(0, 0.1, 64000).astype(np.float32)
    assert "enroll_feature_lengths" not in extract(ts_extractor, [wav])
