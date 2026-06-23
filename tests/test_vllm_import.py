import torch

from open_audio_llm.audio.base import IdentityAudioTower
from open_audio_llm.configuration_audio_llm import AudioTowerConfig
from open_audio_llm.integrations.vllm.processor import estimate_audio_tokens


def test_vllm_token_estimate_uses_tower_and_connector_rate():
    tower = IdentityAudioTower(AudioTowerConfig(input_dim=4, output_dim=4))
    out = estimate_audio_tokens(tower, torch.tensor([10, 6]), connector_downsample=2)
    assert out.tolist() == [5, 3]
