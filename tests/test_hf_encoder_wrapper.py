"""AudioEncoderWrapper checks: no weights or GPU needed.

Tests:
  1. Output-length prediction consistency for each encoder type.
  2. State-dict key migration (old -> new format).
  3. Config round-trip serialization for ZipformerAudioEncoderConfig.
  4. Zipformer SwooshR numerical equivalence vs. the reference formula.
"""

import json
import tempfile
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# 1. Output-length prediction tests
# ---------------------------------------------------------------------------


def test_qwen3asr_output_lengths():
    """Verify Qwen3AudioEncoderWrapper.get_output_lengths formula."""
    from open_audio_llm.integrations.hf.modeling_amphion_asr import Qwen3AudioEncoderWrapper

    for input_len in [100, 200, 500, 1000, 1600]:
        out = Qwen3AudioEncoderWrapper.get_output_lengths(input_len)
        assert isinstance(out, int), f"Expected int, got {type(out)}"
        assert out > 0, f"Expected positive output length, got {out} for input {input_len}"

    t = torch.tensor([100, 200, 500, 1000])
    out_t = Qwen3AudioEncoderWrapper.get_output_lengths(t)
    assert out_t.shape == (4,), f"Expected shape (4,), got {out_t.shape}"
    assert (out_t > 0).all(), f"Expected all positive, got {out_t}"


def test_omnimoe_output_lengths():
    """Verify OmniMoeAudioEncoderWrapper.get_output_lengths formula."""
    from open_audio_llm.integrations.hf.modeling_amphion_asr import OmniMoeAudioEncoderWrapper

    for input_len in [100, 200, 500, 1000, 1600]:
        out = OmniMoeAudioEncoderWrapper.get_output_lengths(input_len)
        assert isinstance(out, int), f"Expected int, got {type(out)}"
        assert out > 0, f"Expected positive, got {out} for input {input_len}"


def test_zipformer_output_lengths():
    """Verify ZipformerAudioEncoderWrapper.get_output_lengths formula."""
    from open_audio_llm.integrations.hf.modeling_amphion_asr import ZipformerAudioEncoderWrapper

    for input_len in [20, 50, 100, 200, 500]:
        out = ZipformerAudioEncoderWrapper.get_output_lengths(input_len)
        expected = ((input_len - 7) // 2) // 2
        assert out == expected, f"input={input_len}: expected {expected}, got {out}"

    t = torch.tensor([20, 50, 100, 200])
    out_t = ZipformerAudioEncoderWrapper.get_output_lengths(t)
    expected_t = ((t - 7) // 2) // 2
    assert (out_t == expected_t).all(), f"Tensor mismatch: {out_t} vs {expected_t}"


# ---------------------------------------------------------------------------
# 2. State-dict key migration
# ---------------------------------------------------------------------------


def test_state_dict_migration():
    """Verify _migrate_old_state_dict remaps old keys correctly."""
    from open_audio_llm.integrations.hf.modeling_amphion_asr import _migrate_old_state_dict

    old_sd = {
        "audio_encoder.conv1.weight": torch.randn(3, 3),
        "audio_encoder.norm.bias": torch.randn(10),
        "multi_modal_projector.proj.0.weight": torch.randn(5, 5),
        "language_model.model.layers.0.weight": torch.randn(4, 4),
        "prompt_embedding.weight": torch.randn(4, 128),
        "audio_encoder.encoder.already_nested.weight": torch.randn(2, 2),
    }

    new_sd = _migrate_old_state_dict(old_sd)

    assert "audio_encoder.encoder.conv1.weight" in new_sd
    assert "audio_encoder.encoder.norm.bias" in new_sd
    assert "multi_modal_projector.proj.0.weight" in new_sd
    assert "language_model.model.layers.0.weight" in new_sd
    assert "prompt_embedding.weight" in new_sd
    # Already nested keys should not be double-nested
    assert "audio_encoder.encoder.already_nested.weight" in new_sd
    assert "audio_encoder.encoder.encoder.already_nested.weight" not in new_sd

    assert "audio_encoder.conv1.weight" not in new_sd
    assert "audio_encoder.norm.bias" not in new_sd


# ---------------------------------------------------------------------------
# 3. Config round-trip
# ---------------------------------------------------------------------------


def test_zipformer_config_roundtrip():
    """Verify ZipformerAudioEncoderConfig survives save/load."""
    from open_audio_llm.integrations.hf.configuration_amphion_asr import ZipformerAudioEncoderConfig, AmphionASRConfig

    cfg = ZipformerAudioEncoderConfig(
        feature_dim=80,
        num_encoder_layers="2,2,4,5,4,2",
        encoder_dim="256,384,512,768,512,384",
        causal=True,
        chunk_size="32",
        left_context_frames="256",
    )

    d = cfg.to_dict()
    assert d["feature_dim"] == 80
    assert d["encoder_dim"] == "256,384,512,768,512,384"
    assert d["causal"] is True

    cfg2 = ZipformerAudioEncoderConfig(**{
        k: v for k, v in d.items()
        if k not in ("model_type", "transformers_version")
    })
    assert cfg2.feature_dim == 80
    assert cfg2.encoder_dim == "256,384,512,768,512,384"
    assert cfg2.causal is True

    full_cfg = AmphionASRConfig(
        audio_encoder_config=d,
        encoder_type="zipformer",
    )
    assert isinstance(full_cfg.audio_encoder_config, ZipformerAudioEncoderConfig)
    assert full_cfg.audio_encoder_config.feature_dim == 80

    with tempfile.TemporaryDirectory() as tmpdir:
        full_cfg.save_pretrained(tmpdir)
        config_path = Path(tmpdir) / "config.json"
        assert config_path.exists()
        with open(config_path) as f:
            raw = json.load(f)
        assert raw["encoder_type"] == "zipformer"
        assert raw["audio_encoder_config"]["feature_dim"] == 80



def test_qwen3asr_config_backward_compat():
    """Verify AmphionASRConfig still works for qwen3asr encoder type."""
    from open_audio_llm.integrations.hf.configuration_amphion_asr import AmphionASRAudioEncoderConfig, AmphionASRConfig

    cfg = AmphionASRConfig(
        audio_encoder_config={"d_model": 1280, "num_mel_bins": 128},
        encoder_type="qwen3asr",
    )
    assert isinstance(cfg.audio_encoder_config, AmphionASRAudioEncoderConfig)
    assert cfg.audio_encoder_config.d_model == 1280


# ---------------------------------------------------------------------------
# 4. SwooshR numerical equivalence
# ---------------------------------------------------------------------------


def test_swoosh_r_equivalence():
    """Verify pure-PyTorch SwooshR matches the mathematical definition."""
    from open_audio_llm.integrations.hf.modeling_amphion_asr import SwooshR

    swoosh = SwooshR()
    x = torch.randn(100, dtype=torch.float32)

    result = swoosh(x)
    zero = torch.tensor(0.0)
    expected = torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687

    assert torch.allclose(result, expected, atol=1e-6), (
        f"Max diff: {(result - expected).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 5. Encoder registry
# ---------------------------------------------------------------------------


def test_encoder_registry():
    """Verify all expected encoder types are registered."""
    from open_audio_llm.integrations.hf.modeling_amphion_asr import _ENCODER_REGISTRY

    expected_types = {"qwen3asr", "qwen3omni_captioner", "qwen3omni", "zipformer"}
    assert set(_ENCODER_REGISTRY.keys()) == expected_types, (
        f"Expected {expected_types}, got {set(_ENCODER_REGISTRY.keys())}"
    )
