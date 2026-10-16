"""Inference frontend of LongEnrollTSASR: request waveforms -> conditioned audio embeddings.

The LLM is not here; vLLM (integrations/vllm/longenroll.py) or the Transformers thinker
(infer.py, reference backend) decodes from the embeddings. The frontend holds the Qwen3-ASR
audio tower, the enrollment encoder with its frozen layer copy, GCA and the ECAPA path, built by
the training model's ``build_conditioning`` and run by its own encode methods, so an exported
checkpoint conditions audio exactly as in validation: enrollment in 30 s windows, mixture with
global attention.

Inference directory (written by export.py):
    <model_dir>/                  Qwen3-ASR checkpoint (bf16) + processor files; vLLM loads it
    <model_dir>/longenroll/       weights.safetensors (enrollment encoder and its layer copy, GCA,
                                  gates, ECAPA projection, ECAPA) and config.json (``model:`` section)
vLLM reads only the top-level weights, so the subfolder is invisible to it.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .backbone import AUDIO_SAMPLE_RATE
from .config import Config
from .ecapa import ECAPATDNN
from .model import LongEnrollTSASR, build_conditioning

EXTRA_DIR = "longenroll"
EXTRA_WEIGHTS = "weights.safetensors"
EXTRA_CONFIG = "config.json"
# Module names of the training model that are not part of the Qwen3-ASR checkpoint.
EXTRA_MODULES = ("enroll_encoder", "enroll_tower", "gca", "gca_gates", "speaker", "ecapa")


def read_extra(model_dir) -> tuple[dict, dict[str, Tensor]]:
    """(config.json, tensors) of the long-enrollment modules in an inference directory."""
    from safetensors.torch import load_file

    folder = Path(model_dir) / EXTRA_DIR
    return json.loads((folder / EXTRA_CONFIG).read_text()), load_file(str(folder / EXTRA_WEIGHTS))


def cast_for_inference(model: nn.Module, device) -> nn.Module:
    """bf16 weights as in DeepSpeed training; the frozen ECAPA stays fp32 (its BatchNorm
    statistics are not cast)."""
    for name, child in model.named_children():
        child.to(device=device, dtype=torch.float32 if name == "ecapa" else torch.bfloat16)
    return model.eval().requires_grad_(False)


class LongEnrollFrontend(nn.Module):
    def __init__(self, tower: nn.Module, ecapa: nn.Module, mcfg):
        super().__init__()
        self.tower = tower
        self.ecapa = ecapa
        build_conditioning(self, tower, mcfg)

    # The training model's encoder code, unchanged.
    device = LongEnrollTSASR.device
    enrollment_features = LongEnrollTSASR.enrollment_features
    encode_enrollment = LongEnrollTSASR.encode_enrollment
    encode_mix = LongEnrollTSASR.encode_mix

    @classmethod
    def from_pretrained(cls, model_dir, device="cuda") -> "LongEnrollFrontend":
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

        from ..audio.target_sot import checkpoint_tensors

        audio_config = Qwen3ASRConfig.from_pretrained(str(model_dir)).thinker_config.audio_config
        audio_config._attn_implementation = "sdpa"
        extra_config, state = read_extra(model_dir)
        frontend = cls(Qwen3ASRAudioEncoder(audio_config), ECAPATDNN(), Config(extra_config["model"]))
        state.update((f"tower.{key}", value) for key, value in checkpoint_tensors(model_dir, "thinker.audio_tower."))
        frontend.load_state_dict(state, strict=True)
        return cast_for_inference(frontend, device)


@torch.no_grad()
def embed_items(model: nn.Module, items: Sequence[dict]) -> list[Tensor]:
    """Audio embeddings [n_audio, 2048] per item (features.LongEnrollFeatures.request output).
    ``model`` is a LongEnrollFrontend or a LongEnrollTSASR."""
    device = model.device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        enroll = model.encode_enrollment([i["enroll_mel"] for i in items], [i["ecapa_fbank"] for i in items])
        modes = torch.tensor([i["mode"] for i in items], device=device, dtype=torch.long)
        audio = model.encode_mix([i["mix_mel"] for i in items], enroll, modes)
    return list(audio.split([i["n_audio"] for i in items]))


def embed_requests(model: nn.Module, features, requests: Sequence[dict], batch_size: int, workers: int = 1):
    """Features and embeddings for every request, in request order. Requests are grouped by
    mixture length so a batch pads little; the grouping only depends on the requests, so two
    backends given the same requests embed them identically. ``workers`` threads build the
    features (mel and fbank run in torch and release the GIL; serial they took 62% of the
    vLLM worker's time on 500 validation requests).
    Returns (items, embeddings, seconds spent on features, seconds spent on the encoders)."""
    started = time.perf_counter()

    def build(r):
        return features.request(r["task"], r["mixture"], r.get("enrollment"), r.get("hotwords", ""))

    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            items = list(pool.map(build, requests))
    else:
        items = [build(r) for r in requests]
    featured = time.perf_counter()
    order = sorted(range(len(items)), key=lambda i: (items[i]["n_audio"], i))
    embeds: list[Optional[Tensor]] = [None] * len(items)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        for index, embed in zip(chunk, embed_items(model, [items[i] for i in chunk])):
            embeds[index] = embed
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    return items, embeds, featured - started, time.perf_counter() - featured


def load_waveform(spec) -> np.ndarray:
    """16 kHz mono float32 from a path or {"path", "start", "duration"} (seconds)."""
    from fractions import Fraction

    import soundfile as sf
    from scipy.signal import resample_poly

    if isinstance(spec, str):
        spec = {"path": spec}
    with sf.SoundFile(spec["path"]) as handle:
        rate = handle.samplerate
        handle.seek(int(round(float(spec.get("start", 0.0)) * rate)))
        frames = int(round(float(spec["duration"]) * rate)) if spec.get("duration") is not None else -1
        audio = handle.read(frames, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != AUDIO_SAMPLE_RATE:
        ratio = Fraction(AUDIO_SAMPLE_RATE, rate)
        audio = resample_poly(audio, ratio.numerator, ratio.denominator)
    if audio.size == 0:
        raise ValueError(f"no audio in {spec}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def parse_answer(text: str) -> tuple[str, str]:
    """'language Chinese<asr_text>...' -> (language, transcript)."""
    head, sep, body = text.partition("<asr_text>")
    if not sep:
        return "", text.strip()
    return head.removeprefix("language ").strip(), body.strip()
