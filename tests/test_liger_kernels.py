import copy

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger kernels are Triton GPU kernels")
def test_liger_text_decoder_matches_native_outputs_and_gradients():
    pytest.importorskip("liger_kernel")
    pytest.importorskip("qwen_asr")
    from qwen_asr.core.transformers_backend import modeling_qwen3_asr as qwen3_asr
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRTextConfig
    from open_audio_llm.integrations.ms_swift.liger import enable_liger_kernels

    torch.manual_seed(7)
    config = Qwen3ASRTextConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]},
    )
    config._attn_implementation = "sdpa"
    native = qwen3_asr.Qwen3ASRThinkerTextModel(config).cuda().train()
    model = torch.nn.Module()
    model.thinker = torch.nn.Module()
    model.thinker.model = copy.deepcopy(native)
    embeds = torch.randn(3, 37, 64, device="cuda", requires_grad=True)
    mask = torch.ones(3, 37, dtype=torch.long, device="cuda")
    mask[1, 30:] = 0
    mask[2, 11:] = 0
    weight = torch.randn(3, 37, 64, device="cuda")

    def run(text):
        embeds.grad = None
        output = text(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        (output * weight).sum().backward()
        return output.detach(), embeds.grad.clone(), {n: p.grad for n, p in text.named_parameters()}

    expected = run(native)
    native_rope = qwen3_asr.apply_rotary_pos_emb
    patched, restore = enable_liger_kernels(model)
    try:
        assert qwen3_asr.apply_rotary_pos_emb is not native_rope
        actual = run(model.thinker.model)
    finally:
        restore()

    # Per layer: input, post-attention, q and k norms; plus the final norm.
    assert patched == {"rms_norm": 2 * 4 + 1, "swiglu": 2}
    assert qwen3_asr.apply_rotary_pos_emb is native_rope
    tolerance = {"rtol": 1e-4, "atol": 1e-4}
    torch.testing.assert_close(actual[0], expected[0], **tolerance)
    torch.testing.assert_close(actual[1], expected[1], **tolerance)
    assert actual[2].keys() == expected[2].keys()
    for name, grad in expected[2].items():
        torch.testing.assert_close(actual[2][name], grad, **tolerance, msg=name)
