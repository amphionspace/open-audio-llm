import pytest
import torch

pytest.importorskip("vllm.v1.sample.logits_processor")

from vllm.v1.sample.logits_processor import BatchUpdate

from open_audio_llm.integrations.vllm.presence_bias import (
    PresenceBias,
    bias_from_env,
    decision_rows,
)

LANGUAGE, NONE = 7, 3


def _processor(bias):
    processor = PresenceBias.__new__(PresenceBias)
    processor.bias, processor.language_id, processor.none_id, processor.outputs = bias, LANGUAGE, NONE, {}
    return processor


def test_only_rows_that_generated_just_language_are_at_the_decision():
    assert decision_rows({0: [], 1: [LANGUAGE], 2: [LANGUAGE, 5], 3: [9]}, LANGUAGE) == [1]


def test_bias_reaches_the_none_logit_at_the_decision_and_nowhere_else():
    processor = _processor(-1.5)
    outputs = [[], [LANGUAGE], [LANGUAGE, 5]]
    processor.update_state(BatchUpdate(batch_size=3, removed=[], moved=[],
                                       added=[(i, None, None, o) for i, o in enumerate(outputs)]))
    logits = processor.apply(torch.zeros(3, 10))
    assert logits[1, NONE] == -1.5
    assert logits.abs().sum() == 1.5
    outputs[0].append(LANGUAGE)  # vLLM appends in place: the processor sees the new token without a batch update
    processor.update_state(None)
    assert processor.apply(torch.zeros(3, 10))[0, NONE] == -1.5


def test_zero_bias_is_inactive_and_argmax_invariant():
    processor = _processor(0.0)
    processor.update_state(BatchUpdate(batch_size=1, removed=[], moved=[], added=[(0, None, None, [LANGUAGE])]))
    assert processor.is_argmax_invariant()
    assert processor.apply(torch.zeros(1, 10)).abs().sum() == 0


def test_bias_comes_from_the_environment_and_must_be_a_finite_number():
    assert bias_from_env({}) == 0.0
    assert bias_from_env({"AMPHION_PRESENCE_BIAS": "-1.25"}) == -1.25
    for bad in ("abc", "inf", "nan"):
        with pytest.raises(ValueError):
            bias_from_env({"AMPHION_PRESENCE_BIAS": bad})
