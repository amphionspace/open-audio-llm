import torch

from open_audio_llm.configuration_audio_llm import AudioTowerConfig
from open_audio_llm.audio.base import IdentityAudioTower


def test_identity_audio_tower_contract():
    tower = IdentityAudioTower(AudioTowerConfig(input_dim=4, output_dim=4))
    features = torch.randn(2, 5, 4)
    lengths = torch.tensor([5, 3])
    hidden, hidden_lens = tower(features, lengths)
    assert hidden.shape == features.shape
    assert hidden_lens.tolist() == [5, 3]
    assert IdentityAudioTower.get_output_lengths(torch.tensor([5])).item() == 5
