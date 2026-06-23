import torch

from open_audio_llm import AudioLLMConfig, AudioLLMForConditionalGeneration
from open_audio_llm.configuration_audio_llm import AudioTowerConfig, ConnectorConfig


def test_audio_llm_forward_smoke():
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
        },
        default_speech_token_id=9,
        start_text_token_id=6,
        end_text_token_id=7,
        start_speech_token_id=8,
        end_speech_token_id=10,
    )
    model = AudioLLMForConditionalGeneration(cfg)
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
