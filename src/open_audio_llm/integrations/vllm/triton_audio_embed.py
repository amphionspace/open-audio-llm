"""Triton audio-embedding client for vLLM evaluation.

The RAG-ASR Triton model exposes two independent surfaces on the same model:

* ``ACTION=infer`` returns ``PROJECTOR_OUT`` / ``PROJECTOR_LEN`` for ASR.
* ``ACTION=list/add/delete/reload`` manages the online hotword pool.

This module intentionally uses only the infer surface.  It keeps the vLLM
evaluation code free from Triton tensor boilerplate and makes the bypass path
easy to disable by leaving the default encoder source as ``vllm``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf


DEFAULT_TRITON_MODEL = "rag_asr_retrieve"


@dataclass(frozen=True)
class TritonAudioEmbedding:
    """One audio item's post-projector frames returned by RAG-ASR Triton."""

    projector_out: np.ndarray
    projector_len: int
    word_list: list[str]

    @property
    def frames(self) -> np.ndarray:
        return np.asarray(self.projector_out[: self.projector_len], dtype=np.float32)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_base64_wav(audio_b64: str) -> tuple[np.ndarray, int]:
    """Decode an OpenAI ``input_audio`` base64 WAV payload."""

    data = base64.b64decode(audio_b64)
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


class TritonAudioEmbedClient:
    """Small HTTP client for ``rag_asr_retrieve`` projector outputs."""

    def __init__(self, url: str, model_name: str = DEFAULT_TRITON_MODEL):
        import tritonclient.http as httpclient

        self.url = url
        self.model_name = model_name
        self._httpclient = httpclient
        self._client = httpclient.InferenceServerClient(url=url)

    def _string_input(self, name: str, value: str):
        tensor = self._httpclient.InferInput(name, [1], "BYTES")
        tensor.set_data_from_numpy(np.array([value], dtype=object))
        return tensor

    def _int_input(self, name: str, value: int):
        tensor = self._httpclient.InferInput(name, [1], "INT32")
        tensor.set_data_from_numpy(np.array([int(value)], dtype=np.int32))
        return tensor

    def infer_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        top_k: int = 0,
    ) -> TritonAudioEmbedding:
        """Return post-projector frames for one waveform."""

        wav = np.asarray(audio, dtype=np.float32)
        inputs = [
            self._string_input("ACTION", "infer"),
            self._httpclient.InferInput("WAV", wav.shape, "FP32"),
            self._int_input("SAMPLE_RATE", int(sample_rate)),
            self._int_input("TOP_K", int(top_k)),
        ]
        inputs[1].set_data_from_numpy(wav)

        outputs = [
            self._httpclient.InferRequestedOutput("PROJECTOR_OUT"),
            self._httpclient.InferRequestedOutput("PROJECTOR_LEN"),
            self._httpclient.InferRequestedOutput("WORD_LIST"),
        ]
        result = self._client.infer(self.model_name, inputs, outputs=outputs)
        projector_out = result.as_numpy("PROJECTOR_OUT").astype(np.float32, copy=False)
        projector_len = int(result.as_numpy("PROJECTOR_LEN")[0])
        word_list = json.loads(_decode(result.as_numpy("WORD_LIST")[0]))
        return TritonAudioEmbedding(projector_out, projector_len, word_list)

    def infer_base64_wav(
        self,
        audio_b64: str,
        *,
        top_k: int = 0,
    ) -> TritonAudioEmbedding:
        audio, sr = decode_base64_wav(audio_b64)
        return self.infer_audio(audio, sr, top_k=top_k)

    def list_hotwords(self, *, limit: int | None = None, offset: int = 0) -> dict:
        """Return a management snapshot for experiment metadata."""

        inputs = [self._string_input("ACTION", "list")]
        if limit is not None:
            inputs.append(self._int_input("LIMIT", int(limit)))
        inputs.append(self._int_input("OFFSET", int(offset)))
        outputs = [
            self._httpclient.InferRequestedOutput("STATUS"),
            self._httpclient.InferRequestedOutput("MESSAGE"),
            self._httpclient.InferRequestedOutput("HOTWORD_COUNT"),
            self._httpclient.InferRequestedOutput("HOTWORD_LIST"),
        ]
        result = self._client.infer(self.model_name, inputs, outputs=outputs)
        message = json.loads(_decode(result.as_numpy("MESSAGE")[0]))
        hotwords = json.loads(_decode(result.as_numpy("HOTWORD_LIST")[0]))
        message["status"] = _decode(result.as_numpy("STATUS")[0])
        message["hotword_count"] = int(result.as_numpy("HOTWORD_COUNT")[0])
        message["hotwords"] = hotwords
        return message


def tensor_to_vllm_audio_embeds_block(frames: np.ndarray, *, uuid: str | None = None) -> dict:
    """Build a Chat Completions content block for vLLM ``--enable-mm-embeds``."""

    import torch
    from vllm.utils.serial_utils import tensor2base64

    # Each Chat Completions content block represents one audio item. vLLM will
    # batch multiple 2D tensors from multiple blocks internally.
    tensor = torch.from_numpy(np.asarray(frames, dtype=np.float32))
    block = {
        "type": "audio_embeds",
        "audio_embeds": tensor2base64(tensor),
    }
    if uuid:
        block["uuid"] = uuid
    return block


def stable_audio_embed_uuid(audio_b64: str, *, prefix: str = "triton-audio") -> str:
    digest = hashlib.sha1(audio_b64.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"
