from collections import deque
from types import SimpleNamespace

from open_audio_llm.integrations.ms_swift.patches import GRPO_NON_MODEL_KEYS, log_rollout_with_meta


def _trainer(log_completions=True):
    calls = []
    trainer = SimpleNamespace(log_completions=log_completions, _logs={"prompt": deque(maxlen=4)},
                              args=SimpleNamespace(generation_batch_size=4),
                              _gather_and_flatten=lambda values, flatten_level: list(values))
    return trainer, calls


def test_grpo_meta_becomes_a_rollout_log_column():
    trainer, calls = _trainer()
    wrapped = log_rollout_with_meta(lambda self, samples: calls.append(len(samples)))
    samples = [SimpleNamespace(extra={"grpo_meta": f'{{"n": {i}}}'}) for i in range(3)]
    wrapped(trainer, samples)
    wrapped(trainer, samples)
    assert calls == [3, 3]
    # bounded like ms-swift's own columns: the newest generation_batch_size entries are kept
    assert list(trainer._logs["grpo_meta"]) == ['{"n": 1}', '{"n": 2}', '{"n": 0}', '{"n": 1}', '{"n": 2}'][-4:]
    assert trainer._logs["grpo_meta"].maxlen == 4


def test_no_column_without_meta_on_every_sample_or_without_completion_logging():
    trainer, _ = _trainer()
    wrapped = log_rollout_with_meta(lambda self, samples: None)
    wrapped(trainer, [SimpleNamespace(extra={"grpo_meta": "{}"}), SimpleNamespace(extra={})])
    assert "grpo_meta" not in trainer._logs
    trainer, _ = _trainer(log_completions=False)
    wrapped(trainer, [SimpleNamespace(extra={"grpo_meta": "{}"})])
    assert "grpo_meta" not in trainer._logs


def test_grpo_meta_never_reaches_generation():
    assert "grpo_meta" in GRPO_NON_MODEL_KEYS
