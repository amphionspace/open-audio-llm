#!/usr/bin/env python3
"""Validate raw-audio vs Triton-embedding vLLM requests for Qwen3-ASR."""

from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.request
from copy import deepcopy
from pathlib import Path

import soundfile as sf

from open_audio_llm.integrations.vllm.triton_audio_embed import (
    DEFAULT_TRITON_MODEL,
    TritonAudioEmbedClient,
    stable_audio_embed_uuid,
    tensor_to_vllm_audio_embeds_block,
)


def _wav_base64(path: str) -> str:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=-1)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_qwen3_asr_messages(
    *,
    audio_b64: str,
    enrollment_b64: str | None,
    hotwords: list[str],
) -> list[dict]:
    sys_lines: list[str] = []
    user_content: list[dict] = []
    if enrollment_b64:
        sys_lines.append("Given the speaker's voice in the first audio.")
        user_content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": enrollment_b64, "format": "wav"},
            }
        )
    if hotwords:
        sys_lines.append(f"Hotwords: {','.join(hotwords)}")
    user_content.append(
        {
            "type": "input_audio",
            "input_audio": {"data": audio_b64, "format": "wav"},
        }
    )
    return [
        {"role": "system", "content": "\n".join(sys_lines)},
        {"role": "user", "content": user_content},
    ]


def _messages_with_triton_embeds(
    messages: list[dict],
    *,
    client: TritonAudioEmbedClient,
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    converted = deepcopy(messages)
    embeds: list[dict] = []
    for message in converted:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "input_audio":
                new_content.append(block)
                continue
            audio_b64 = (block.get("input_audio") or {}).get("data")
            if not audio_b64:
                new_content.append(block)
                continue
            embedding = client.infer_base64_wav(audio_b64, top_k=top_k)
            embeds.append(
                {
                    "projector_len": embedding.projector_len,
                    "frames_shape": list(embedding.frames.shape),
                    "word_list": embedding.word_list,
                }
            )
            new_content.append(
                tensor_to_vllm_audio_embeds_block(
                    embedding.frames,
                    uuid=stable_audio_embed_uuid(audio_b64),
                )
            )
        message["content"] = new_content
    return converted, embeds


def _post_chat(base_url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return str(content).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8009")
    parser.add_argument("--model", default="amphionasr-1.7b")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--enrollment")
    parser.add_argument("--hotwords", default="")
    parser.add_argument("--triton-url", default="localhost:8000")
    parser.add_argument("--triton-model", default=DEFAULT_TRITON_MODEL)
    parser.add_argument("--triton-top-k", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    audio_b64 = _wav_base64(args.audio)
    enrollment_b64 = _wav_base64(args.enrollment) if args.enrollment else None
    hotwords = [w.strip() for w in args.hotwords.split(",") if w.strip()]
    raw_messages = _build_qwen3_asr_messages(
        audio_b64=audio_b64,
        enrollment_b64=enrollment_b64,
        hotwords=hotwords,
    )

    client = TritonAudioEmbedClient(args.triton_url, args.triton_model)
    triton_messages, embeds = _messages_with_triton_embeds(
        raw_messages,
        client=client,
        top_k=args.triton_top_k,
    )

    common = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    raw_response = _post_chat(
        args.base_url,
        {**common, "messages": raw_messages},
    )
    triton_response = _post_chat(
        args.base_url,
        {**common, "messages": triton_messages},
    )
    raw_text = _extract_text(raw_response)
    triton_text = _extract_text(triton_response)

    print(
        json.dumps(
            {
                "audio": str(Path(args.audio)),
                "enrollment": str(Path(args.enrollment)) if args.enrollment else None,
                "raw_text": raw_text,
                "triton_text": triton_text,
                "exact_match": raw_text == triton_text,
                "triton_embeds": embeds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
