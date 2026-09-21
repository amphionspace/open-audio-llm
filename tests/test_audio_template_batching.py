from types import SimpleNamespace

import numpy as np
import pytest
import torch

from open_audio_llm.integrations.ms_swift import template as generic


@pytest.fixture(params=["generic", "qwen3"])
def audio_template(request, monkeypatch):
    if generic.Template is object:
        pytest.skip("ms-swift is unavailable")
    if request.param == "qwen3":
        from open_audio_llm.integrations.ms_swift import register_qwen3_asr as module

        cls = module.Qwen3ASRTemplate
    else:
        module, cls = generic, generic.AudioLLMTemplate
    template = object.__new__(cls)
    template.sampling_rate = 16000
    template._hop_length = 160
    monkeypatch.setattr(module.Template, "_data_collator", lambda *a, **kw: {})
    monkeypatch.setattr(module.Template, "_encode", lambda *a, **kw: {})
    return module, template


@pytest.mark.parametrize("slots", [1, 2])
def test_variable_length_batch_preserves_frames_lengths_and_slots(
    audio_template, slots
):
    module, template = audio_template
    is_generic = module is generic
    batch = []
    for frames in (3, 5):
        features = torch.arange(slots * frames * 2).reshape(slots, frames, 2).float()
        if is_generic:
            batch.append(
                {
                    "input_features": features,
                    "feature_lens": torch.full((slots,), frames),
                }
            )
        else:
            batch.append(
                {
                    "input_features": features.transpose(1, 2),
                    "feature_attention_mask": torch.ones(
                        slots, frames, dtype=torch.long
                    ),
                }
            )
    result = template._data_collator(batch)
    if is_generic:
        outputs = result["input_features"] if slots > 1 else [result["input_features"]]
        lengths = result["feature_lens"] if slots > 1 else [result["feature_lens"]]
        for slot, (output, lens) in enumerate(zip(outputs, lengths)):
            assert output.shape == (2, 5, 2)
            torch.testing.assert_close(output[0, :3], batch[0]["input_features"][slot])
            assert torch.count_nonzero(output[0, 3:]) == 0
            assert lens.tolist() == [3, 5]
    else:
        output = result["input_features"]
        assert output.shape == (slots * 2, 2, 5)
        torch.testing.assert_close(output[:slots, :, :3], batch[0]["input_features"])
        assert torch.count_nonzero(output[:slots, :, 3:]) == 0
        assert (
            result["feature_attention_mask"].sum(-1).tolist()
            == [3] * slots + [5] * slots
        )
    assert batch[0]["input_features"].shape == (
        (slots, 3, 2) if is_generic else (slots, 2, 3)
    )


@pytest.mark.parametrize("n_samples", [3200, 1281, 16001, 16000 * 31 + 1])
def test_encode_preserves_extractor_values_and_decodes_once(
    audio_template, monkeypatch, n_samples
):
    from transformers import WhisperFeatureExtractor

    module, template = audio_template
    extractor = WhisperFeatureExtractor(feature_size=128, chunk_length=1)
    template._feature_extractor = extractor
    wav = np.random.default_rng(42).normal(size=n_samples).astype(np.float32)
    decoded = []

    def load(path, **kwargs):
        decoded.append(path)
        return wav

    monkeypatch.setattr(module, "load_audio", load)
    monkeypatch.setattr(
        module, "load_batch", lambda paths, load_func: [load_func(p) for p in paths]
    )

    def encode(self, inputs):
        tags = self.replace_tag("audio", 0, inputs)
        return {"tags": tags}

    monkeypatch.setattr(module.Template, "_encode", encode)
    inputs = SimpleNamespace(audios=["sample.wav"])
    result = template._encode(inputs)
    extraction = {}
    if module is not generic:
        extraction = dict(truncation=False, padding='max_length',
                          max_length=max(extractor.n_samples, ((n_samples + 159) // 160) * 160))
    reference = extractor(
        [wav], sampling_rate=16000, return_attention_mask=True, return_tensors="pt",
        **extraction,
    )
    frames = int(reference["attention_mask"].sum())
    expected = reference["input_features"][..., :frames]
    if module is generic:
        expected = expected.transpose(1, 2)
        assert result["feature_lens"].tolist() == [frames]
    else:
        assert frames == (n_samples + 159) // 160
        assert result["tags"].count(
            "<|audio_pad|>"
        ) == module._get_feat_extract_output_lengths(frames)
        assert result["feature_attention_mask"].sum().item() == frames
    torch.testing.assert_close(result["input_features"], expected, rtol=0, atol=0)
    assert decoded == ["sample.wav"]
    assert not hasattr(inputs, "_audio_llm_waveforms")


def test_encode_releases_waveform_on_failure(audio_template, monkeypatch):
    module, template = audio_template
    if module is generic:
        pytest.skip("generic template does not retain waveforms on inputs")
    monkeypatch.setattr(module, "load_batch", lambda *a, **kw: [np.zeros(3200)])

    def fail(*args):
        raise ValueError("invalid prompt")

    monkeypatch.setattr(module.Template, "_encode", fail)
    inputs = SimpleNamespace(audios=["sample.wav"])
    with pytest.raises(ValueError, match="invalid prompt"):
        template._encode(inputs)
    assert not hasattr(inputs, "_audio_llm_waveforms")
