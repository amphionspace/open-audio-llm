import torch

from open_audio_llm.configuration_audio_llm import ConnectorConfig
from open_audio_llm.connectors.mlp_downsample import MLPDownsampleConnector


def test_mlp_downsample_connector_shapes():
    connector = MLPDownsampleConnector(
        ConnectorConfig(input_dim=4, output_dim=8, downsample_rate=2)
    )
    hidden = torch.randn(2, 6, 4)
    lengths = torch.tensor([6, 4])
    out, out_lens = connector(hidden, lengths)
    assert out.shape == (2, 3, 8)
    assert out_lens.tolist() == [3, 2]
