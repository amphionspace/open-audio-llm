"""Give GRPO groups that all make the wrong presence decision something to learn from.

GRPO z-scores rewards inside each group of generations. When every sample of a target-free prompt
writes text, or every sample of a speech prompt writes nothing, the rewards tie, the advantages are
all zero and the worst behaviour gets no policy gradient. In such a group the last sample is replaced
by the reference before scoring, so the group holds one right decision to reinforce and the rest to
push down. Groups with any correct decision stay on-policy.
"""

from __future__ import annotations

from typing import Callable

from .rewards import qwen3_asr


def contrast_indices(outputs: list[str], solutions: list[str], group_size: int) -> list[int]:
    """Index of the sample to replace in each group whose outputs all decide presence wrongly."""
    if group_size < 2 or len(outputs) != len(solutions) or len(outputs) % group_size:
        raise ValueError(f"{len(outputs)} outputs and {len(solutions)} solutions in groups of {group_size}")
    indices = []
    for start in range(0, len(outputs), group_size):
        group = range(start, start + group_size)
        if len({solutions[i] for i in group}) != 1:
            raise ValueError(f"Group at {start} mixes prompts: GRPO rollouts must keep generations together")
        speech = bool(qwen3_asr.units(qwen3_asr.transcript(solutions[start])))
        if all(bool(qwen3_asr.units(qwen3_asr.transcript(outputs[i]))) != speech for i in group):
            indices.append(group[-1])
    return indices


def replace_with_reference(group, sample, encode: Callable[[str], list[int]], special_ids) -> bool:
    """Rewrite ``sample`` as the reference, ending like a finished generation of its group."""
    finished = next((s for s in group if s.finish_reason == "stop" and s.response_token_ids), None)
    if finished is None:
        return False  # no finished generation shows how a completion ends, and no sample stops anyway
    solution = sample.extra["solution"]
    end = finished.response_token_ids[-1]
    sample.response_token_ids = encode(solution) + ([end] if end in special_ids else [])
    sample.rollout_logprobs = []
    sample.finish_reason = "stop"
    sample.messages = [*sample.messages[:-1], {"role": "assistant", "content": solution}]
    return True


def install_presence_contrast(trainer) -> None:
    rollout = trainer._rollout_samples
    tokenizer = trainer.template.tokenizer
    special_ids = set(tokenizer.all_special_ids)

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def rollout_with_contrast(inputs):
        samples = rollout(inputs)
        if not trainer.model.training:
            return samples
        size = trainer.num_generations
        outputs = [s.messages[-1]["content"] for s in samples]
        solutions = [s.extra["solution"] for s in samples]
        replaced = 0
        for index in contrast_indices(outputs, solutions, size):
            group = samples[index - size + 1: index + 1]
            replaced += replace_with_reference(group, samples[index], encode, special_ids)
        trainer._metrics["train"]["presence_contrast/groups"].append(replaced)
        return samples

    trainer._rollout_samples = rollout_with_contrast
