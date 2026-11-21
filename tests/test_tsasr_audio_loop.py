from types import SimpleNamespace

import pytest
import torch


def _tower(layers=3):
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
    from open_audio_llm.tsasr.sep_token import AudioSepToken, _patch_hf_encoder_forward

    config = Qwen3ASRAudioEncoderConfig(
        d_model=32, encoder_layers=layers, encoder_attention_heads=4,
        encoder_ffn_dim=64, output_dim=16, downsample_hidden_size=8, n_window=50,
        conv_chunksize=3,
    )
    config._attn_implementation = "sdpa"
    encoder = Qwen3ASRAudioEncoder(config).train()
    encoder.add_module("sep_token", AudioSepToken(config.d_model))
    _patch_hf_encoder_forward(encoder)
    return encoder


def _save_loop(tmp_path, encoder, start, end):
    from safetensors.torch import save_file
    from open_audio_llm.tsasr.audio_loop import init_loop_state

    state = init_loop_state({k: v.detach() for k, v in encoder.state_dict().items()}, start, end)
    save_file({k: v.contiguous() for k, v in state.items()}, str(tmp_path / "model.safetensors"))
    encoder.config.amphion_audio_loop = {"start": start, "end": end}
    return state


def _serial(encoder, features, lengths, enrolls):
    return torch.cat([
        encoder(feature[:, :length], feature_lens=torch.tensor([length]), enroll_n_frames=enroll).last_hidden_state
        for feature, length, enroll in zip(features, lengths, enrolls)
    ])


def test_identity_loop_keeps_outputs_and_runs_in_both_training_paths(tmp_path):
    pytest.importorskip("qwen_asr")
    from open_audio_llm.integrations.ms_swift.audio_batching import enable_batched_audio
    from open_audio_llm.tsasr.audio_loop import attach_audio_loop, layer_sequence

    torch.manual_seed(3)
    encoder = _tower()
    lengths, enrolls = [80, 120, 250], [0, 50, 100]
    features = torch.randn(len(lengths), 128, 260)
    with torch.no_grad():
        expected = _serial(encoder, features, lengths, enrolls)
    state = _save_loop(tmp_path, encoder, 1, 3)
    assert len(state) == 2 * len([k for k in encoder.state_dict() if k.startswith("layers.0.")])
    model = SimpleNamespace(thinker=SimpleNamespace(audio_tower=encoder, get_audio_features=None))
    attach_audio_loop(model, str(tmp_path))
    order = layer_sequence(encoder)
    assert order == [encoder.layers[0], encoder.layers[1], encoder.layers[2], *encoder.loop_layers]
    # The copies start from the source layers, with zero output projections.
    torch.testing.assert_close(encoder.loop_layers[0].fc1.weight, encoder.layers[1].fc1.weight, rtol=0, atol=0)
    assert torch.count_nonzero(encoder.loop_layers[1].self_attn.out_proj.weight) == 0
    with torch.no_grad():
        torch.testing.assert_close(_serial(encoder, features, lengths, enrolls), expected, rtol=0, atol=0)

    restore = enable_batched_audio(model)
    try:
        actual = model.thinker.get_audio_features(
            features, audio_feature_lengths=torch.tensor(lengths), enroll_n_frames=torch.tensor(enrolls),
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        (actual * torch.randn_like(actual)).sum().backward()
        # A zero output projection still receives gradient, so the copies leave the identity.
        for layer in encoder.loop_layers:
            assert torch.count_nonzero(layer.fc2.weight.grad) > 0
            assert torch.count_nonzero(layer.self_attn.out_proj.weight.grad) > 0
        with torch.no_grad():
            encoder.loop_layers[0].fc2.weight.normal_()
            batched = model.thinker.get_audio_features(
                features, audio_feature_lengths=torch.tensor(lengths), enroll_n_frames=torch.tensor(enrolls),
            )
    finally:
        restore()
    with torch.no_grad():
        serial = _serial(encoder, features, lengths, enrolls)
    assert not torch.allclose(serial, expected)
    torch.testing.assert_close(batched, serial, atol=2e-5, rtol=2e-4)


def test_configured_loop_without_weights_fails(tmp_path):
    pytest.importorskip("qwen_asr")
    from safetensors.torch import save_file
    from open_audio_llm.tsasr.audio_loop import attach_audio_loop

    encoder = _tower()
    encoder.config.amphion_audio_loop = {"start": 0, "end": 2}
    save_file({"thinker.audio_tower.sep_token.weight": torch.zeros(32)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError, match="do not match"):
        attach_audio_loop(SimpleNamespace(thinker=SimpleNamespace(audio_tower=encoder)), str(tmp_path))


@pytest.mark.parametrize("spec", [{"start": 2, "end": 2}, {"start": -1, "end": 2}, {"start": 0, "end": 4}])
def test_loop_span_rejects_bad_ranges(spec):
    from open_audio_llm.tsasr.audio_loop import loop_span

    with pytest.raises(ValueError):
        loop_span(SimpleNamespace(encoder_layers=3, amphion_audio_loop=spec))
    assert loop_span(SimpleNamespace(encoder_layers=3)) is None
