"""Patch vLLM registry inspection for rollout smoke environments.

In the current amphionft environment, ``python -m
vllm.model_executor.models.registry`` segfaults during model inspection while a
plain ``import vllm`` succeeds. ms-swift rollout relies on that subprocess
inspection before the server starts. This patch intercepts only that registry
subprocess call and executes the cloudpickled inspection function in-process.
"""

from __future__ import annotations

import pickle
import subprocess
from pathlib import Path
from typing import Any


_ORIGINAL_RUN = subprocess.run


def _is_vllm_registry_inspection(cmd: Any) -> bool:
    if not isinstance(cmd, (list, tuple)):
        return False
    return "-m" in cmd and "vllm.model_executor.models.registry" in cmd


def _patched_run(cmd, *args, **kwargs):
    if not _is_vllm_registry_inspection(cmd):
        return _ORIGINAL_RUN(cmd, *args, **kwargs)

    input_bytes = kwargs.get("input")
    if input_bytes is None:
        return _ORIGINAL_RUN(cmd, *args, **kwargs)

    fn, output_file = pickle.loads(input_bytes)
    result = fn()
    Path(output_file).write_bytes(pickle.dumps(result))
    return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")


if getattr(subprocess.run, "_open_audio_llm_vllm_registry_patch", False) is False:
    _patched_run._open_audio_llm_vllm_registry_patch = True
    subprocess.run = _patched_run
