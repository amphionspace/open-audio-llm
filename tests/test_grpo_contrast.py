from types import SimpleNamespace

import pytest

from open_audio_llm.integrations.ms_swift.grpo_contrast import contrast_indices, replace_with_reference

EMPTY = "language None<asr_text>"
SPEECH = "language Chinese<asr_text>你好"


def test_only_groups_that_all_decide_wrongly_get_a_contrast():
    outputs = [
        SPEECH, SPEECH,  # target absent, both write: wrong
        EMPTY, SPEECH,  # target absent, one right: on-policy
        EMPTY, EMPTY,  # speech, both empty: wrong
        SPEECH, EMPTY,  # speech, one right: on-policy
        SPEECH, SPEECH,  # speech, both write, even with different errors: on-policy
        EMPTY, EMPTY,  # target absent, both right: on-policy
    ]
    solutions = [EMPTY] * 4 + [SPEECH] * 6 + [EMPTY] * 2
    assert contrast_indices(outputs, solutions, 2) == [1, 5]


def test_a_mixed_group_means_the_rollout_order_is_not_what_grpo_assumes():
    with pytest.raises(ValueError, match="mixes prompts"):
        contrast_indices([EMPTY, EMPTY], [EMPTY, SPEECH], 2)


def sample(content, ids, finish="stop"):
    return SimpleNamespace(
        messages=[{"role": "user", "content": "<audio>"}, {"role": "assistant", "content": content}],
        response_token_ids=ids, rollout_logprobs=[[-0.1]], finish_reason=finish, extra={"solution": EMPTY},
    )


def test_the_reference_ends_like_a_finished_generation_of_its_group():
    group = [sample(SPEECH, [5, 6, 99]), sample(SPEECH, [7, 8], "length")]
    assert replace_with_reference(group, group[1], lambda text: [1, 2], {99})
    assert group[1].response_token_ids == [1, 2, 99]
    assert group[1].messages[-1] == {"role": "assistant", "content": EMPTY}
    assert group[1].finish_reason == "stop"
    assert group[0].response_token_ids == [5, 6, 99]


def test_no_replacement_when_no_generation_finished():
    group = [sample(SPEECH, [5], "length"), sample(SPEECH, [6], "length")]
    assert not replace_with_reference(group, group[1], lambda text: [1], {99})
    assert group[1].response_token_ids == [6]
