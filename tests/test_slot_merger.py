import torch

from open_audio_llm.configuration_audio_llm import AudioLLMConfig
from open_audio_llm.merge.slot_merger import SlotMerger


def test_slot_merger_supports_multiple_speech_slots():
    cfg = AudioLLMConfig(
        default_speech_token_id=9,
        start_text_token_id=6,
        end_text_token_id=7,
        start_speech_token_id=8,
        end_speech_token_id=10,
    )
    merger = SlotMerger(cfg, hidden_size=4)
    input_ids = torch.tensor([[6, 1, 9, 2, 9, 7]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    inputs_embeds = torch.randn(1, 6, 4)
    first = torch.randn(1, 3, 4)
    second = torch.randn(1, 2, 4)
    merged, mask, labels = merger(
        [first, second],
        [torch.tensor([3]), torch.tensor([2])],
        inputs_embeds,
        input_ids,
        attention_mask,
    )
    assert labels is None
    assert merged.shape == (1, 9, 4)
    assert mask.sum().item() == 9
