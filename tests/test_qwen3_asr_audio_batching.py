from types import MethodType, SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('checkpointing', [None, False, True])
def test_batched_encoder_preserves_outputs_gradients_and_audio_isolation(checkpointing):
    pytest.importorskip('qwen_asr')
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
    from open_audio_llm.integrations.ms_swift.audio_batching import enable_batched_audio

    torch.manual_seed(7)
    config = Qwen3ASRAudioEncoderConfig(
        d_model=32, encoder_layers=2, encoder_attention_heads=4,
        encoder_ffn_dim=64, output_dim=16, downsample_hidden_size=8, n_window=50,
        conv_chunksize=3,
    )
    config._attn_implementation = 'sdpa'
    encoder = Qwen3ASRAudioEncoder(config).train()
    if checkpointing is not None:
        encoder.gradient_checkpointing_enable({'use_reentrant': checkpointing})
    lengths = [1, 7, 8, 15, 49, 99, 100, 101, 199, 200, 817]
    features = torch.randn(len(lengths), 128, 821, requires_grad=True)
    mask = torch.arange(821)[None, :] < torch.tensor(lengths)[:, None]
    thinker = SimpleNamespace(audio_tower=encoder)

    def native(self, features, feature_attention_mask=None, audio_feature_lengths=None):
        lens = feature_attention_mask.sum(-1) if feature_attention_mask is not None else audio_feature_lengths
        return torch.cat([self.audio_tower(feature[:, :length], feature_lens=length.unsqueeze(0)).last_hidden_state
                          for feature, length in zip(features, lens)])

    thinker.get_audio_features = MethodType(native, thinker)
    expected = thinker.get_audio_features(features, mask)
    weight = torch.randn_like(expected)
    (expected * weight).sum().backward()
    expected_grads = {name: p.grad.clone() for name, p in encoder.named_parameters()}
    expected_input_grad = features.grad.clone()
    encoder.zero_grad(set_to_none=True)
    features.grad = None
    restore = enable_batched_audio(SimpleNamespace(thinker=thinker))
    try:
        actual = thinker.get_audio_features(features, audio_feature_lengths=torch.tensor(lengths))
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        (actual * weight).sum().backward()
        for name, p in encoder.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            # Key bias has an analytically zero gradient (softmax shift
            # invariance); relative error alone magnifies roundoff there.
            torch.testing.assert_close(p.grad, expected_grads[name], atol=2e-7, rtol=2e-4)
        torch.testing.assert_close(features.grad, expected_input_grad, atol=2e-7, rtol=2e-4)
        assert torch.count_nonzero(features.grad.masked_select(~mask[:, None, :])) == 0
        with torch.no_grad():
            altered = features.detach().clone()
            altered[1:] *= 100
            altered[0, :, lengths[0]:] = -1000
            independent = thinker.get_audio_features(altered, mask)
            torch.testing.assert_close(independent[0], actual[0], rtol=0, atol=0)
            # Long TS audio must retain attention across the old 800-frame window.
            changed = features.detach().clone()
            changed[-1, :, 800:817] *= 100
            mixed = thinker.get_audio_features(changed, mask)
            first_long_token = len(actual) - (817 // 100 * 13 + (17 + 7) // 8)
            assert not torch.equal(mixed[first_long_token], actual[first_long_token])
    finally:
        restore()
    assert thinker.get_audio_features.__func__ is native
    with torch.no_grad():
        torch.testing.assert_close(thinker.get_audio_features(features, mask), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA streams require a GPU")
def test_batched_bf16_encoder_backward_has_no_cross_audio_or_padding_gradients():
    pytest.importorskip('qwen_asr')
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder
    from open_audio_llm.integrations.ms_swift.audio_batching import enable_batched_audio

    torch.manual_seed(11)
    config = Qwen3ASRAudioEncoderConfig(
        d_model=32, encoder_layers=2, encoder_attention_heads=4,
        encoder_ffn_dim=64, output_dim=16, downsample_hidden_size=8, n_window=50,
    )
    config._attn_implementation = 'sdpa'
    encoder = Qwen3ASRAudioEncoder(config).cuda().train()
    encoder.gradient_checkpointing_enable({'use_reentrant': False})
    lengths = [817, 99, 101]
    features = torch.randn(3, 128, 830, device='cuda', requires_grad=True)
    mask = torch.arange(830, device='cuda')[None, :] < torch.tensor(lengths, device='cuda')[:, None]
    thinker = SimpleNamespace(audio_tower=encoder, get_audio_features=None)
    restore = enable_batched_audio(SimpleNamespace(thinker=thinker))
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = thinker.get_audio_features(features, mask)
        assert actual.dtype == torch.bfloat16
        assert len(actual) == sum(n // 100 * 13 + (n % 100 + 7) // 8 for n in lengths)
        actual[0].float().square().sum().backward()
        assert torch.count_nonzero(features.grad[1:]) == 0
        assert torch.count_nonzero(features.grad[0, :, 817:]) == 0
        assert torch.count_nonzero(features.grad[0, :, 800:817]) > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            altered = features.detach().clone()
            altered[1:] *= 100
            altered[0, :, 817:] = 1000
            independent = thinker.get_audio_features(altered, mask)
        torch.testing.assert_close(independent[0], actual[0], rtol=0, atol=0)
    finally:
        restore()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA streams require a GPU")
def test_parallel_encoder_preserves_native_precision_and_caller_context():
    pytest.importorskip("qwen_asr")
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRAudioEncoderConfig,
    )
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
        Qwen3ASRAudioEncoder,
    )
    from open_audio_llm.integrations.ms_swift.audio_batching import (
        enable_parallel_audio,
    )

    torch.manual_seed(3)
    config = Qwen3ASRAudioEncoderConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        output_dim=16,
        downsample_hidden_size=8,
        n_window=50,
    )
    config._attn_implementation = "sdpa"
    encoder = (
        Qwen3ASRAudioEncoder(config)
        .to("cuda", torch.bfloat16)
        .eval()
        .requires_grad_(False)
    )
    lengths = [1, 7, 8, 15, 49, 99, 100, 101, 199, 200, 217]
    features = torch.randn(len(lengths), 128, 230, device="cuda")
    mask = (
        torch.arange(230, device="cuda")[None, :]
        < torch.tensor(lengths, device="cuda")[:, None]
    )
    thinker = SimpleNamespace(audio_tower=encoder)

    def native(self, features, mask):
        return torch.cat(
            [
                self.audio_tower(
                    feature[:, :length], feature_lens=length.unsqueeze(0)
                ).last_hidden_state
                for feature, length in zip(features, mask.sum(-1))
            ]
        )

    thinker.get_audio_features = MethodType(native, thinker)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = thinker.get_audio_features(features, mask)
    restore = enable_parallel_audio(SimpleNamespace(thinker=thinker))
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = thinker.get_audio_features(features, mask)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            decoder = torch.nn.Linear(16, 1).cuda()
            decoder(actual).float().square().mean().backward()
            assert torch.isfinite(decoder.weight.grad).all()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            altered = features.clone()
            altered[1:] *= 100
            altered[0, :, lengths[0] :] = -1000
            independent = thinker.get_audio_features(altered, mask)
            torch.testing.assert_close(independent[:1], actual[:1], rtol=0, atol=0)
    finally:
        restore()
    assert thinker.get_audio_features.__func__ is native
