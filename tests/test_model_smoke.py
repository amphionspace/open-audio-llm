import pytest
import torch

from open_audio_llm import AudioLLMConfig, AudioLLMForConditionalGeneration
from open_audio_llm.configuration_audio_llm import AudioTowerConfig, ConnectorConfig


@pytest.fixture
def model():
    cfg = AudioLLMConfig(
        audio_tower_config=AudioTowerConfig(type="identity", input_dim=4, output_dim=4),
        connector_config=ConnectorConfig(
            type="mlp_downsample",
            input_dim=4,
            output_dim=8,
            downsample_rate=1,
        ),
        text_config={
            "model_type": "gpt2",
            "n_embd": 8,
            "n_layer": 1,
            "n_head": 1,
            "vocab_size": 32,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        default_speech_token_id=9,
        start_text_token_id=6,
        end_text_token_id=7,
        start_speech_token_id=8,
        end_speech_token_id=10,
    )
    return AudioLLMForConditionalGeneration(cfg)


def test_audio_llm_forward_smoke(model):
    input_ids = torch.tensor([[6, 9, 7, 1]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    labels = input_ids.clone()
    features = torch.randn(1, 3, 4)
    lengths = torch.tensor([3])
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=features,
        feature_lens=lengths,
        labels=labels,
    )
    assert out.logits.shape[0] == 1


@pytest.mark.parametrize("audio", [True, False])
@pytest.mark.parametrize("return_dict", [True, False])
@pytest.mark.parametrize("num_return_sequences", [1, 2])
def test_swift_generation_preserves_completions(
    model, audio, return_dict, num_return_sequences
):
    from open_audio_llm.integrations.ms_swift.template import AudioLLMTemplate, Template

    if Template is object:
        pytest.skip("ms-swift is unavailable")
    template = object.__new__(AudioLLMTemplate)
    input_ids = torch.tensor([[6, 9 if audio else 1, 7, 1]])
    audio_inputs = (
        {"input_features": torch.randn(1, 3, 4), "feature_lens": torch.tensor([3])}
        if audio else {}
    )
    output = template.generate(
        model.eval(),
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        min_new_tokens=3,
        max_new_tokens=3,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        return_dict_in_generate=return_dict,
        audio_slot_count=[1 if audio else 0],
        **audio_inputs,
    )
    sequences = output.sequences if return_dict else output
    completions = template.get_generate_ids(sequences, input_ids.shape[-1])
    assert completions.shape == (num_return_sequences, 3)
    torch.testing.assert_close(
        sequences[:, :input_ids.shape[-1]],
        input_ids.expand(num_return_sequences, -1),
    )
