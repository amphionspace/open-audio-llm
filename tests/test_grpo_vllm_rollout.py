import pytest
import torch

from open_audio_llm.integrations.ms_swift.grpo_vllm_rollout import (
    collapse_audio_pads,
    group_prompts,
    policy_weights,
)

PAD = 9


def test_the_expanded_audio_run_becomes_one_placeholder():
    assert collapse_audio_pads([1, 2, PAD, PAD, PAD, 3, 4], PAD, 3) == [1, 2, PAD, 3, 4]


def test_a_text_only_prompt_is_unchanged():
    assert collapse_audio_pads([1, 2, 3], PAD, 0) == [1, 2, 3]


def test_pad_count_must_match_the_encoder_frames():
    with pytest.raises(ValueError, match="encoder produced 2"):
        collapse_audio_pads([1, PAD, PAD, PAD, 2], PAD, 2)


def test_split_audio_runs_are_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        collapse_audio_pads([PAD, 1, PAD], PAD, 2)


def encoded(ids, features, enroll=0):
    return {"input_ids": ids, "input_features": torch.tensor(features, dtype=torch.float32), "enroll_n_frames": enroll}


def test_generations_of_one_prompt_share_a_group_in_first_seen_order():
    items = [
        encoded([1, PAD], [[0.5]]),
        encoded([1, PAD], [[0.25]]),  # same text, different audio
        encoded([1, PAD], [[0.5]]),
        encoded([1, PAD], [[0.5]], enroll=3),  # same audio, a TS split
    ]
    assert list(group_prompts(items).values()) == [[0, 2], [1], [3]]


def test_policy_weights_skip_the_audio_tower_and_unwrap_checkpoint_names():
    model = torch.nn.Module()
    model.thinker = torch.nn.Module()
    model.thinker.audio_tower = torch.nn.Linear(1, 1)
    model.thinker.model = torch.nn.Module()
    model.thinker.model._checkpoint_wrapped_module = torch.nn.Linear(1, 1, bias=False)
    assert [name for name, _ in policy_weights(model)] == ["thinker.model.weight"]
