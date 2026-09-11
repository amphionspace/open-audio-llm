from types import MethodType, SimpleNamespace

import pytest
import torch


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
