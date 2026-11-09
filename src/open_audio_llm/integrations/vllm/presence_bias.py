"""A fixed logit bias on `` None`` at the Qwen3-ASR presence decision, for ``vllm serve --logits-processors``.

Qwen3-ASR answers ``language X<asr_text>text``; the token after ``language`` decides whether anything is written
(`` None``: no speech). GRPO for target-speaker ASR moved this decision of the shared checkpoint toward ``None``,
and plain ASR inherited it: on gigaspeech, sum-full drops 194 utterances with speech where SFT 073 dropped 17 and
the official model 20 (runs/whitebox-20261107 analysis, 2026-11-09). Serving plain ASR and TS-ASR from separate
servers lets each pick its own operating point without retraining (white-box task 08: a ``None`` bias moves along
the FA/miss curve).

The bias is added to the `` None`` logit only while a request has generated exactly ``language``; every other
step, and every request whose first token is not ``language``, is untouched. With bias 0 the processor is
inactive. Configure it on the server:

    AMPHION_PRESENCE_BIAS=-1.0 vllm serve ... \
        --logits-processors open_audio_llm.integrations.vllm.presence_bias:PresenceBias

A negative bias makes the server write more (fewer dropped utterances, more text on silence); a positive one
writes less.
"""

from __future__ import annotations

import math
import os

import torch
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

BIAS_ENV = "AMPHION_PRESENCE_BIAS"
DECISION_PREFIX = "language"
NONE_TOKEN = " None"


def bias_from_env(environ=os.environ) -> float:
    value = environ.get(BIAS_ENV, "0")
    try:
        bias = float(value)
    except ValueError as error:
        raise ValueError(f"{BIAS_ENV} must be a number, got {value!r}") from error
    if not math.isfinite(bias):
        raise ValueError(f"{BIAS_ENV} must be finite")
    return bias


def decision_rows(outputs: dict[int, list[int]], language_id: int) -> list[int]:
    """Batch rows whose only generated token so far is ``language``: their next token is the presence decision."""
    return [row for row, tokens in outputs.items() if len(tokens) == 1 and tokens[0] == language_id]


class PresenceBias(LogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        from transformers import AutoTokenizer

        self.bias = bias_from_env()
        model = vllm_config.model_config
        tokenizer = AutoTokenizer.from_pretrained(model.tokenizer, trust_remote_code=model.trust_remote_code)
        ids = [tokenizer.encode(t, add_special_tokens=False) for t in (DECISION_PREFIX, NONE_TOKEN)]
        if any(len(i) != 1 for i in ids):
            raise ValueError(f"presence bias needs single-token {DECISION_PREFIX!r} and {NONE_TOKEN!r}, got {ids}")
        self.language_id, self.none_id = ids[0][0], ids[1][0]
        self.outputs: dict[int, list[int]] = {}

    def is_argmax_invariant(self) -> bool:
        return self.bias == 0

    def update_state(self, batch_update: BatchUpdate | None):
        if self.bias:
            # Keep a reference to every request's running output tokens (vLLM updates the list in place).
            process_dict_updates(self.outputs, batch_update, lambda params, prompt, output: output)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.bias:
            rows = decision_rows(self.outputs, self.language_id)
            if rows:
                logits[rows, self.none_id] += self.bias
        return logits
