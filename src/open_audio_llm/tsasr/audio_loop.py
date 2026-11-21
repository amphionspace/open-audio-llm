"""Looped audio-tower block: a trainable copy of layers ``[start, end)`` runs once more after layer ``end - 1``.

The audio config's ``amphion_audio_loop = {"start": s, "end": e}`` (kept in config.json) makes the tower run
``layers[:e] + loop_layers + layers[e:]``. The copies live at ``audio_tower.loop_layers.{i}``, so the HF tower
(training, scoring) and the vLLM tower build the same order and load the same checkpoint names.

``init_loop_state`` creates the copies from the tower's own layers with zero attention-output and fc2 weights and
biases. Each copy then starts as the identity on its residual stream (LLaMA-Pro block expansion), so a looped
model starts with exactly the source model's outputs and the original layers can stay frozen.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

LOOP_KEY = "amphion_audio_loop"
LOOP_PREFIX = "thinker.audio_tower.loop_layers."
# Output projections whose weights and biases start at zero: attention output (HF out_proj, vLLM out_proj) and fc2.
ZERO_INIT = ("self_attn.out_proj.", "fc2.")


def loop_span(config) -> tuple[int, int] | None:
    spec = getattr(config, LOOP_KEY, None)
    if not spec:
        return None
    start, end = int(spec["start"]), int(spec["end"])
    if not 0 <= start < end <= int(config.encoder_layers):
        raise ValueError(f"{LOOP_KEY} {spec} must satisfy 0 <= start < end <= {config.encoder_layers}")
    return start, end


def layer_sequence(tower: nn.Module) -> list[nn.Module]:
    """The tower's transformer layers in run order, loop copies included."""
    layers = list(tower.layers)
    loop = getattr(tower, "loop_layers", None)
    if loop is None:
        return layers
    end = tower.loop_end
    return layers[:end] + list(loop) + layers[end:]


def attach_audio_loop(model: nn.Module, model_dir: str | None = None) -> nn.Module:
    """Add ``audio_tower.loop_layers`` when the audio config asks for it and load them from ``model_dir``.

    ``from_pretrained`` drops the checkpoint's loop keys (the module does not exist yet), so they are read here;
    a configured loop whose weights are missing is an error, never a silent random init.
    """
    tower = model.thinker.audio_tower
    span = loop_span(tower.config)
    if span is None or getattr(tower, "loop_layers", None) is not None:
        return model
    start, end = span
    tower.add_module("loop_layers", nn.ModuleList(copy.deepcopy(tower.layers[i]) for i in range(start, end)))
    tower.loop_end = end
    if model_dir is None:
        raise ValueError(f"{LOOP_KEY} is set but no checkpoint directory was given for the loop weights")
    state = _read_prefixed(Path(model_dir), LOOP_PREFIX)
    missing, unexpected = tower.loop_layers.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Loop weights in {model_dir} do not match: missing {missing[:4]}, unexpected {unexpected[:4]}")
    return model


def init_loop_state(tower_state: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    """Checkpoint entries (``LOOP_PREFIX`` names) of identity-initialised copies of layers ``[start, end)``.

    ``tower_state`` holds the tower's own entries without the ``thinker.audio_tower.`` prefix.
    """
    out = {}
    for i, source in enumerate(range(start, end)):
        prefix = f"layers.{source}."
        entries = {k[len(prefix):]: v for k, v in tower_state.items() if k.startswith(prefix)}
        if not entries:
            raise KeyError(f"tower state has no layer {source}")
        for name, value in entries.items():
            value = torch.zeros_like(value) if name.startswith(ZERO_INIT) else value.clone()
            out[f"{LOOP_PREFIX}{i}.{name}"] = value
    return out


def _read_prefixed(root: Path, prefix: str) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    out = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    out[key[len(prefix):]] = f.get_tensor(key)
    return out
