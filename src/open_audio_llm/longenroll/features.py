"""Model inputs of LongEnrollTSASR: mel, ECAPA fbank and prompt ids.

Training (data.py, Catalog samples) and inference (request waveforms) build their inputs here,
so a deployed model sees exactly what validation saw. Nothing here imports the Catalog: the
vLLM worker runs in the serving environment, which does not carry the data stack.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import torch

from .backbone import AUDIO_SAMPLE_RATE, MelExtractor, PromptBuilder, feat_extract_output_lengths
from .ecapa import compute_fbank
from .model import BOTH, NONE

TS_SYSTEM = (
    "The target speaker's enrollment is provided separately. Transcribe only the enrolled "
    "speaker from the clip. If the enrolled speaker is not present, output nothing."
)
# Plain / hotword ASR: the switched-off enrollment path reads this much of the mixture.
DUMMY_ENROLL_SECONDS = 3.0
# ECAPA input: an enrollment up to ECAPA_WHOLE_SECONDS goes in whole; a longer one gives its
# head, a random length in ECAPA_SECONDS when training and the upper bound in validation.
ECAPA_WHOLE_SECONDS = 10.0
ECAPA_SECONDS = (5.0, 10.0)
# Shorter clips are padded before fbank so ECAPA gets enough frames.
ECAPA_MIN_SECONDS = 0.5
TASK_NAMES = ("ts", "ts_absent", "asr", "hotword")
# Inference request tasks: target speaker, plain ASR, hotword ASR.
REQUEST_TASKS = ("ts", "asr", "hotword")
# Inference builds requests in threads; the Rust tokenizer must not be entered concurrently.
# A module-level lock is not pickled with the features into dataloader workers.
_TOKENIZER_LOCK = threading.Lock()


def task_name(task: str, answer: str) -> str:
    if task == "ts_asr":
        return "ts" if answer.split("<asr_text>", 1)[-1].strip() else "ts_absent"
    return "hotword" if task == "asr_hotwords" else "asr"


def request_system(task: str, hotwords: str = "") -> str:
    """System turn of an inference request, as the Catalog writes it for training records."""
    if task not in REQUEST_TASKS:
        raise ValueError(f"task must be one of {REQUEST_TASKS}, got {task!r}")
    if task == "ts":
        return TS_SYSTEM
    # A hotword record without hotwords is trained with an empty system turn.
    return f"Hotwords: {hotwords}" if task == "hotword" and hotwords else ""


class LongEnrollFeatures:
    """Mel, ECAPA fbank and prompt ids for one Catalog sample or inference request."""

    def __init__(self, processor, ts_system: str = TS_SYSTEM):
        self.mel = MelExtractor(processor.feature_extractor)
        self.prompt = PromptBuilder(processor.tokenizer)
        self.ts_system = ts_system

    def inputs(self, mix: np.ndarray, enroll: np.ndarray, system: str,
               rng: Optional[np.random.Generator] = None, answer: Optional[str] = None) -> dict:
        """``answer`` None gives the generation prompt (unlabeled, ends at the assistant turn)."""
        head = enroll
        if enroll.size > ECAPA_WHOLE_SECONDS * AUDIO_SAMPLE_RATE:
            lo, hi = ECAPA_SECONDS
            head = enroll[: int(round((hi if rng is None else rng.uniform(lo, hi)) * AUDIO_SAMPLE_RATE))]
        minimum = int(ECAPA_MIN_SECONDS * AUDIO_SAMPLE_RATE)
        if head.size < minimum:
            head = np.pad(head, (0, minimum - head.size))
        mix_mel = self.mel(mix).to(torch.bfloat16)
        n_audio = int(feat_extract_output_lengths(torch.tensor(mix_mel.shape[-1])))
        with _TOKENIZER_LOCK:
            input_ids, labels = self.prompt.build(n_audio, answer, system=system)
        return {
            "mix_mel": mix_mel,
            "enroll_mel": self.mel(enroll).to(torch.bfloat16),
            "ecapa_fbank": compute_fbank(torch.from_numpy(np.ascontiguousarray(head))),
            "input_ids": input_ids,
            "labels": labels,
            "n_audio": n_audio,
        }

    def __call__(self, sample: dict, rng: Optional[np.random.Generator]) -> dict:
        messages = sample["messages"]
        answer = messages[-1]["content"]
        if sample["task"] == "ts_asr":
            mix = np.asarray(sample["mix_wav"], dtype=np.float32)
            enroll = np.asarray(sample["audios"][0], dtype=np.float32)
            system, mode = self.ts_system, BOTH
        else:
            mix = np.asarray(sample["audios"][0], dtype=np.float32)
            enroll = mix[: int(DUMMY_ENROLL_SECONDS * AUDIO_SAMPLE_RATE)]
            system, mode = messages[0]["content"], NONE
        example = self.inputs(mix, enroll, system, rng, answer)
        example.update({
            "mode": mode,
            "task": task_name(sample["task"], answer),
            "answer": answer,
            "system": system,
            "language": answer.split("<asr_text>", 1)[0].removeprefix("language ").strip(),
            "dataset_id": sample.get("dataset_id"),
            "mix_sec": mix.size / AUDIO_SAMPLE_RATE,
            "enroll_sec": enroll.size / AUDIO_SAMPLE_RATE if mode == BOTH else 0.0,
            "_performance": sample.get("_performance"),
        })
        return example

    def request(self, task: str, mixture: np.ndarray, enrollment: Optional[np.ndarray] = None,
                hotwords: str = "") -> dict:
        """Inference inputs, built like a validation example of the same task."""
        system = request_system(task, hotwords)
        mix = np.asarray(mixture, dtype=np.float32).reshape(-1)
        if task == "ts":
            if enrollment is None:
                raise ValueError("task 'ts' needs an enrollment")
            enroll, mode = np.asarray(enrollment, dtype=np.float32).reshape(-1), BOTH
        else:
            if enrollment is not None:
                raise ValueError(f"task {task!r} takes no enrollment")
            enroll, mode = mix[: int(DUMMY_ENROLL_SECONDS * AUDIO_SAMPLE_RATE)], NONE
        example = self.inputs(mix, enroll, system)
        example.update({
            "mode": mode,
            "system": system,
            "mix_sec": mix.size / AUDIO_SAMPLE_RATE,
            "enroll_sec": enroll.size / AUDIO_SAMPLE_RATE if mode == BOTH else 0.0,
        })
        return example
