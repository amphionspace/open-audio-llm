import copy

import pytest
import torch

from open_audio_llm.integrations.ms_swift.packed_text import pack_text


@pytest.mark.parametrize("left", [False, True])
def test_pack_text_concatenates_rows_and_keeps_audio_inputs(left):
    rows = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    labels = [[-100, 6, 7], [-100, 9], [4, 11, -100, 13]]
    width = 4
    pad = lambda row, value: [value] * (width - len(row)) + row if left else row + [value] * (width - len(row))
    features = torch.randn(3, 128, 9)
    batch = {
        "input_ids": torch.tensor([pad(r, 0) for r in rows]),
        "labels": torch.tensor([pad(r, -100) for r in labels]),
        "attention_mask": torch.tensor([pad([1] * len(r), 0) for r in rows]),
        "input_features": features,
        "feature_attention_mask": torch.ones(3, 4, dtype=torch.long),
    }

    packed = pack_text(batch)

    assert "attention_mask" not in packed
    assert packed["input_ids"].tolist() == [[5, 6, 7, 8, 9, 10, 11, 12, 13]]
    # Each sample's first label would be predicted across the boundary; it is dropped as in a padded row.
    assert packed["labels"].tolist() == [[-100, 6, 7, -100, 9, -100, 11, -100, 13]]
    assert packed["position_ids"].tolist() == [[0, 1, 2, 0, 1, 0, 1, 2, 3]]
    assert packed["cu_seq_lens_q"].tolist() == packed["cu_seq_lens_k"].tolist() == [0, 3, 5, 9]
    assert packed["cu_seq_lens_q"].dtype == torch.int32
    assert packed["max_length_q"] == packed["max_length_k"] == 4
    assert packed["input_features"] is features
    assert packed["feature_attention_mask"].shape == (3, 4)


def test_pack_text_rejects_per_token_inputs_it_cannot_align():
    batch = {"input_ids": torch.ones(2, 3, dtype=torch.long), "labels": torch.ones(2, 3, dtype=torch.long),
             "attention_mask": torch.ones(2, 3, dtype=torch.long), "loss_scale": torch.ones(2, 3)}
    with pytest.raises(ValueError, match="loss_scale"):
        pack_text(batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashAttention-2 needs a GPU")
def test_packed_flash_text_decoder_matches_padded_rows():
    pytest.importorskip("flash_attn")
    pytest.importorskip("qwen_asr")
    from qwen_asr.core.transformers_backend import modeling_qwen3_asr as qwen3_asr
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRTextConfig

    torch.manual_seed(7)
    config = Qwen3ASRTextConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]},
    )
    config._attn_implementation = "flash_attention_2"
    padded_model = qwen3_asr.Qwen3ASRThinkerTextModel(config).to("cuda", torch.bfloat16).train()
    packed_model = copy.deepcopy(padded_model)
    lengths = [37, 11, 30]
    width = max(lengths)
    mask = torch.arange(width)[None] < torch.tensor(lengths)[:, None]
    embeds = torch.randn(3, width, 64)
    weight = torch.randn(3, width, 64)
    batch = {"input_ids": torch.zeros(3, width, dtype=torch.long), "labels": torch.zeros(3, width, dtype=torch.long),
             "attention_mask": mask.long()}
    packed = pack_text(dict(batch))

    padded_out = padded_model(inputs_embeds=embeds.to("cuda", torch.bfloat16),
                              attention_mask=mask.long().cuda()).last_hidden_state
    (padded_out[mask.cuda()].float() * weight[mask].cuda()).sum().backward()
    packed_out = packed_model(
        inputs_embeds=embeds[mask][None].to("cuda", torch.bfloat16),
        position_ids=packed["position_ids"].cuda(),
        cu_seq_lens_q=packed["cu_seq_lens_q"].cuda(), cu_seq_lens_k=packed["cu_seq_lens_k"].cuda(),
        max_length_q=packed["max_length_q"], max_length_k=packed["max_length_k"],
    ).last_hidden_state
    (packed_out[0].float() * weight[mask].cuda()).sum().backward()

    torch.testing.assert_close(packed_out[0], padded_out[mask.cuda()], rtol=2e-2, atol=2e-2)
    for (name, expected), actual in zip(padded_model.named_parameters(), packed_model.parameters()):
        if expected.grad is None:  # embed_tokens: the inputs are embeddings
            assert actual.grad is None, name
            continue
        torch.testing.assert_close(actual.grad.float(), expected.grad.float(), rtol=5e-2, atol=5e-2, msg=name)
