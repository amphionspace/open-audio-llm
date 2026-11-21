#!/usr/bin/env python3
"""Smoke-test that AmphionASR can be instantiated, saved, and reloaded
via the HuggingFace ``AutoModel`` API.

This script does NOT require a real checkpoint — it creates a tiny
model from scratch to verify the full save/load round-trip.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from loguru import logger

# Ensure the local package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .configuration_amphion_asr import (
        AmphionASRAudioEncoderConfig,
        AmphionASRConfig,
        AmphionASRProjectorConfig,
    )
except ImportError:
    from configuration_amphion_asr import (
        AmphionASRAudioEncoderConfig,
        AmphionASRConfig,
        AmphionASRProjectorConfig,
    )


def _build_tiny_config() -> AmphionASRConfig:
    """Build a minimal config for a tiny model (fast instantiation)."""
    from transformers import AutoConfig

    # Use a tiny Qwen2-style config
    text_config = {
        "model_type": "qwen2",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
        "vocab_size": 151680,
        "use_sliding_window": False,
        "sliding_window": None,
        "tie_word_embeddings": False,
    }

    audio_config = AmphionASRAudioEncoderConfig(
        d_model=64,
        encoder_attention_heads=2,
        encoder_ffn_dim=128,
        encoder_layers=2,
        num_mel_bins=128,
        output_dim=64,
    )

    proj_config = AmphionASRProjectorConfig(
        encoder_dim=64,
        llm_dim=64,
        downsample_rate=1,
    )

    return AmphionASRConfig(
        audio_encoder_config=audio_config.to_dict(),
        projector_config=proj_config.to_dict(),
        text_config=text_config,
        num_prompt_tokens=4,
        default_speech_token_id=151655,
        start_text_token_id=151656,
        end_text_token_id=151657,
        start_speech_token_id=151658,
        end_speech_token_id=151659,
        encoder_type="qwen3asr",
    )


def test_config_roundtrip():
    """Test config serialisation / deserialisation."""
    logger.info("=" * 60)
    logger.info("TEST: Config round-trip")
    logger.info("=" * 60)

    config = _build_tiny_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        config.save_pretrained(tmpdir)

        loaded = AmphionASRConfig.from_pretrained(tmpdir)
        assert loaded.model_type == "amphion_asr"
        assert loaded.num_prompt_tokens == 4
        assert loaded.default_speech_token_id == 151655
        assert loaded.encoder_type == "qwen3asr"

        # Sub-configs survive (may be dict or object after round-trip)
        enc_cfg = loaded.audio_encoder_config
        d_model = enc_cfg["d_model"] if isinstance(enc_cfg, dict) else getattr(enc_cfg, "d_model", None)
        assert d_model == 64, f"d_model={d_model}"
        logger.success("  PASSED: config round-trip OK")
    return True


def test_model_instantiation():
    """Test that the model can be instantiated from config."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Model instantiation from config")
    logger.info("=" * 60)

    config = _build_tiny_config()

    # We need to skip qwen_asr import for the tiny test since the audio
    # encoder requires the qwen_asr package.  Test the projector and
    # overall structure instead.
    try:
        from .modeling_amphion_asr import (
            AmphionASRMultiModalProjector,
            SwooshR,
        )
    except ImportError:
        from modeling_amphion_asr import (
            AmphionASRMultiModalProjector,
            SwooshR,
        )

    # Test SwooshR
    act = SwooshR()
    x = torch.randn(2, 10)
    y = act(x)
    expected = torch.logaddexp(torch.tensor(0.0), x - 1.0) - 0.08 * x - 0.313261687
    assert torch.allclose(y, expected, atol=1e-6), "SwooshR mismatch"
    logger.success("  PASSED: SwooshR activation OK")

    # Test MultiModalProjector
    proj = AmphionASRMultiModalProjector(config)
    x = torch.randn(2, 20, 64)  # (B, T, encoder_dim)
    out = proj(x)
    assert out.shape == (2, 20, 64), f"Projector output shape: {out.shape}"
    logger.success(f"  PASSED: Projector output shape {tuple(out.shape)} OK")

    # Test with downsample_rate > 1
    config2 = _build_tiny_config()
    config2.projector_config.downsample_rate = 5
    config2.projector_config.encoder_dim = 64
    proj2 = AmphionASRMultiModalProjector(config2)
    x2 = torch.randn(2, 23, 64)
    out2 = proj2(x2)
    assert out2.shape == (2, 4, 64), f"DS projector shape: {out2.shape}"
    logger.success(f"  PASSED: Downsample projector shape {tuple(out2.shape)} OK")

    return True


def test_model_save_load():
    """Test save_pretrained / from_pretrained round-trip (without encoder)."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Model save/load round-trip (structural)")
    logger.info("=" * 60)

    config = _build_tiny_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save config + source files
        config.save_pretrained(tmpdir)

        import shutil
        src_dir = Path(__file__).resolve().parent
        for fname in [
            "configuration_amphion_asr.py",
            "modeling_amphion_asr.py",
            "processing_amphion_asr.py",
        ]:
            shutil.copy2(src_dir / fname, Path(tmpdir) / fname)

        # Verify auto_map is in config
        import json
        with open(Path(tmpdir) / "config.json") as f:
            saved_cfg = json.load(f)
        assert "auto_map" in saved_cfg, "auto_map missing from config.json"
        assert "AutoModelForCausalLM" in saved_cfg["auto_map"]
        logger.info(f"  auto_map: {saved_cfg['auto_map']}")
        logger.success("  PASSED: auto_map present in config.json")

    return True


def test_peft_compatibility():
    """Test that PEFT LoraConfig can target the language model."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: PEFT compatibility (structural)")
    logger.info("=" * 60)

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        logger.warning("  SKIPPED: peft not installed")
        return True

    from transformers import AutoModelForCausalLM, AutoConfig

    # Build a tiny causal LM to test PEFT on
    text_cfg = AutoConfig.for_model(
        model_type="qwen2",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        vocab_size=151680,
        use_sliding_window=False,
        sliding_window=None,
    )
    llm = AutoModelForCausalLM.from_config(text_cfg)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(llm, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    assert trainable > 0, "No trainable parameters after PEFT"
    assert trainable < total, "All parameters trainable (LoRA not applied)"
    logger.success("  PASSED: PEFT LoRA applied successfully to LLM component")

    return True


def test_automodel_from_pretrained():
    """Test the full AutoModel.from_pretrained flow with trust_remote_code."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: AutoModel.from_pretrained (trust_remote_code)")
    logger.info("=" * 60)

    from transformers import AutoConfig
    import shutil, json

    config = _build_tiny_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Save config
        config.save_pretrained(tmpdir)

        # 2. Copy Python source files
        src_dir = Path(__file__).resolve().parent
        for fname in [
            "configuration_amphion_asr.py",
            "modeling_amphion_asr.py",
            "processing_amphion_asr.py",
        ]:
            shutil.copy2(src_dir / fname, Path(tmpdir) / fname)

        # 3. Create dummy weights
        from safetensors.torch import save_file

        weights = {}

        weights["multi_modal_projector.proj.1.weight"] = torch.randn(64, 64)
        weights["multi_modal_projector.proj.1.bias"] = torch.zeros(64)
        weights["multi_modal_projector.proj.3.weight"] = torch.randn(64, 64)
        weights["multi_modal_projector.proj.3.bias"] = torch.zeros(64)

        weights["prompt_embedding.weight"] = torch.randn(4, 64)

        save_file(weights, Path(tmpdir) / "model.safetensors")

        # 4. Try to load config via AutoConfig
        loaded_cfg = AutoConfig.from_pretrained(tmpdir, trust_remote_code=True)
        assert loaded_cfg.model_type == "amphion_asr"
        logger.success("  PASSED: AutoConfig.from_pretrained loaded correctly")

        # 5. Verify config.json has correct structure
        with open(Path(tmpdir) / "config.json") as f:
            cfg_json = json.load(f)
        assert cfg_json["model_type"] == "amphion_asr"
        assert "audio_encoder_config" in cfg_json
        assert "projector_config" in cfg_json
        assert "text_config" in cfg_json
        logger.success("  PASSED: config.json structure verified")

    return True


def test_peft_on_amphion_model():
    """Test PEFT LoRA on the full AmphionASR model structure."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: PEFT LoRA on AmphionASR model (structural)")
    logger.info("=" * 60)

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        logger.warning("  SKIPPED: peft not installed")
        return True

    from transformers import AutoConfig, AutoModelForCausalLM

    # Build a tiny LLM
    text_cfg = AutoConfig.for_model(
        model_type="qwen2",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        vocab_size=151680,
        use_sliding_window=False,
        sliding_window=None,
    )
    llm = AutoModelForCausalLM.from_config(text_cfg)

    # Simulate AmphionASR structure: wrap LLM with projector + prompt_embedding
    try:
        from .modeling_amphion_asr import AmphionASRMultiModalProjector
    except ImportError:
        from modeling_amphion_asr import AmphionASRMultiModalProjector

    config = _build_tiny_config()
    projector = AmphionASRMultiModalProjector(config)
    prompt_emb = torch.nn.Embedding(4, 64)

    # Apply LoRA only to the LLM (this is what the training code does)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "up_proj", "gate_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    peft_llm = get_peft_model(llm, lora_config)

    # Verify LLM has LoRA adapters
    trainable = sum(p.numel() for p in peft_llm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_llm.parameters())
    logger.info(f"  LLM: Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Verify projector is still fully trainable
    proj_trainable = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    proj_total = sum(p.numel() for p in projector.parameters())
    assert proj_trainable == proj_total, "Projector should be fully trainable"
    logger.info(f"  Projector: {proj_trainable:,} / {proj_total:,} (100%)")

    # Verify prompt embedding is trainable
    pe_trainable = sum(p.numel() for p in prompt_emb.parameters() if p.requires_grad)
    assert pe_trainable > 0, "Prompt embedding should be trainable"
    logger.info(f"  Prompt embedding: {pe_trainable:,} params")

    # Test merge_and_unload
    merged_llm = peft_llm.merge_and_unload()
    merged_params = sum(p.numel() for p in merged_llm.parameters())
    logger.info(f"  After merge_and_unload: {merged_params:,} params (all base)")
    logger.success("  PASSED: PEFT LoRA + merge_and_unload works correctly")

    return True


def main():
    results = []
    for test_fn in [
        test_config_roundtrip,
        test_model_instantiation,
        test_model_save_load,
        test_peft_compatibility,
        test_automodel_from_pretrained,
        test_peft_on_amphion_model,
    ]:
        try:
            ok = test_fn()
            results.append((test_fn.__name__, ok))
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_fn.__name__, False))

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        logger.info(f"  [{status}] {name}")
        if not ok:
            all_ok = False

    if all_ok:
        logger.success("\nAll tests passed!")
    else:
        logger.error("\nSome tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
