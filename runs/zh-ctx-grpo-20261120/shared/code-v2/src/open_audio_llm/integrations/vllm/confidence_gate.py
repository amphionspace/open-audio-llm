"""Confidence rejection for Qwen3-ASR chat completions served by vLLM.

An ASGI middleware for ``vllm serve --middleware``. A recognition whose first
transcript tokens are unconfident, or that runs into the token limit as a
repetition loop, is answered as the model's own no-speech output, so clients
that already accept an empty result need no change. It targets distant
background talk that the model would otherwise transcribe as unstable text
(issue 91; calibration in
docs/experiments/2026-10-30-issue91-background-loops in this repository).

Enable it on the plain ASR server only, with the threshold calibrated for the
served checkpoint:

    AMPHION_CONFIDENCE_THRESHOLD=-0.831 vllm serve ... \
        --middleware open_audio_llm.integrations.vllm.confidence_gate.ConfidenceGate

The module uses only the standard library, so a delivery pack can ship this
file as a top-level module (``--middleware confidence_gate.ConfidenceGate``).

Contract with clients:

- Every audio chat completion is decoded upstream without streaming, with
  token logprobs (and prompt logprobs when the request continues an assistant
  prefix), then answered in the requested form. A streaming client receives
  the same chunks it would otherwise get, all at once after decoding.
- A rejected free decode returns ``language None<asr_text>`` with
  ``finish_reason`` ``stop``. A rejected continuation of a transcript prefix
  returns empty content; the prefix itself was sent by the client, so a client
  that sends one must drop it when ``amphion_confidence.rejected`` is true.
  The prefix is scored by its prompt logprobs, so it must be token-aligned
  with what the model generated (cut back by tokens, as audiollm-go does): a
  prefix cut mid-token re-tokenizes into a merge the model never emits and
  scores far too low.
- Each response carries ``amphion_confidence`` (score, threshold, rule).
  Logprobs appear only when the client asked for them.
"""

from __future__ import annotations

import json
import logging
import os
import re

ASR_TEXT = "<asr_text>"
STOP_TOKENS = frozenset({"<|im_end|>", "<|endoftext|>"})
NO_SPEECH = "language None" + ASR_TEXT
# Issue-91 probe: a background recognition is unconfident from its first
# tokens, even when it later settles into a confident-looking repetition loop.
HEAD_TOKENS = 16
# One unit of 1-4 characters repeated at least 8 times (word characters only).
LOOP = re.compile(r"(.{1,4}?)\1{7,}")
THRESHOLD_ENV = "AMPHION_CONFIDENCE_THRESHOLD"
PATH = "/v1/chat/completions"

logger = logging.getLogger("vllm.confidence_gate")


def has_audio(messages):
    for message in messages if isinstance(messages, list) else ():
        content = message.get("content") if isinstance(message, dict) else None
        for part in content if isinstance(content, list) else ():
            if isinstance(part, dict) and part.get("type") in ("input_audio", "audio_url"):
                return True
    return False


def transcript_prefix(request):
    """Transcript text the client forces as the start of the answer, or None."""
    messages = request.get("messages")
    if not request.get("continue_final_message") or not messages:
        return None
    last = messages[-1]
    if last.get("role") != "assistant" or not isinstance(last.get("content"), str):
        return None
    head, marker, text = last["content"].partition(ASR_TEXT)
    return text if marker else None


def prompt_transcript_logprobs(prompt_logprobs):
    """Logprobs of the prompt tokens after the last ``<asr_text>`` (the forced prefix)."""
    values = []
    for entry in prompt_logprobs or ():
        if not entry:
            continue
        # The prompt token itself comes first; any further entries are top-k alternatives.
        first = next(iter(entry.values()))
        if first.get("decoded_token") == ASR_TEXT:
            values = []
        else:
            values.append(first["logprob"])
    return values


def decide(request, response, threshold):
    """Return (rejected, report) for one non-streaming chat completion response."""
    choice = response["choices"][0]
    content = choice["message"].get("content") or ""
    prefix = transcript_prefix(request)
    tokens = (choice.get("logprobs") or {}).get("content") or []
    if prefix is None:
        marker = next((i for i, t in enumerate(tokens) if t["token"] == ASR_TEXT), None)
        if marker is None:  # not an ASR-shaped answer: leave it alone
            return False, {"threshold": threshold, "head_tokens": 0, "prefix": False, "score": None, "rule": None}
        tokens = tokens[marker + 1:]
        head_lp = []
        text = content.partition(ASR_TEXT)[2]
    else:
        head_lp = prompt_transcript_logprobs(response.get("prompt_logprobs"))
        text = prefix + content
    head_lp += [t["logprob"] for t in tokens if t["token"] not in STOP_TOKENS]
    head_lp = head_lp[:HEAD_TOKENS]
    report = {"threshold": threshold, "head_tokens": len(head_lp), "prefix": prefix is not None}
    if not head_lp or not re.sub(r"\W", "", text):
        return False, {**report, "score": None, "rule": None}
    score = sum(head_lp) / len(head_lp)
    report["score"] = round(score, 4)
    if score < threshold:
        return True, {**report, "rule": "head"}
    if choice.get("finish_reason") == "length" and LOOP.search(re.sub(r"\W", "", text)):
        return True, {**report, "rule": "loop"}
    return False, {**report, "rule": None}


def sse(objects):
    return b"".join(b"data: " + json.dumps(o, ensure_ascii=False).encode() + b"\n\n" for o in objects) + b"data: [DONE]\n\n"


def as_stream(response, report, include_usage):
    """The chunks a streaming client receives, rebuilt from the full response."""
    choice = response["choices"][0]
    base = {"id": response["id"], "object": "chat.completion.chunk", "created": response["created"], "model": response["model"]}
    chunks = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}]}]
    content = choice["message"].get("content") or ""
    if content:
        chunks.append({**base, "choices": [{"index": 0, "delta": {"content": content}, "logprobs": choice.get("logprobs"), "finish_reason": None}]})
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": choice["finish_reason"]}],
                   "amphion_confidence": report})
    if include_usage and response.get("usage"):
        chunks.append({**base, "choices": [], "usage": response["usage"]})
    return sse(chunks)


class ConfidenceGate:
    def __init__(self, app):
        self.app = app
        value = os.environ.get(THRESHOLD_ENV)
        if value is None:
            raise RuntimeError(f"{THRESHOLD_ENV} must hold the head-logprob threshold calibrated for the served checkpoint")
        self.threshold = float(value)
        logger.info("ASR confidence gate on: head %d tokens, threshold %.4f, loop rule on", HEAD_TOKENS, self.threshold)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != PATH:
            return await self.app(scope, receive, send)
        chunks, more = [], True
        while more:
            message = await receive()
            if message["type"] != "http.request":
                return await self.app(scope, _replay(b"".join(chunks), receive), send)
            chunks.append(message.get("body", b""))
            more = message.get("more_body", False)
        body = b"".join(chunks)
        try:
            request = json.loads(body)
        except ValueError:
            request = None
        if not isinstance(request, dict) or request.get("n", 1) != 1 or not has_audio(request.get("messages")):
            return await self.app(scope, _replay(body, receive), send)

        upstream = {k: v for k, v in request.items() if k not in ("stream", "stream_options")}
        upstream.update(stream=False, logprobs=True, top_logprobs=request.get("top_logprobs") or 0)
        if transcript_prefix(request) is not None and request.get("prompt_logprobs") is None:
            upstream["prompt_logprobs"] = 0
        status, headers, payload = await _call(self.app, scope, _replay(json.dumps(upstream).encode(), receive))
        if status != 200:
            return await _respond(send, status, headers, payload)
        response = json.loads(payload)
        rejected, report = decide(request, response, self.threshold)
        report["rejected"] = rejected
        choice = response["choices"][0]
        if rejected:
            choice["message"]["content"] = "" if report["prefix"] else NO_SPEECH
            choice["finish_reason"] = "stop"
            logger.info("ASR output rejected: rule=%s score=%s head_tokens=%d prefix=%s completion_tokens=%s",
                        report["rule"], report["score"], report["head_tokens"], report["prefix"],
                        (response.get("usage") or {}).get("completion_tokens"))
        if not request.get("logprobs") or rejected:
            choice["logprobs"] = None
        if request.get("prompt_logprobs") is None:
            response.pop("prompt_logprobs", None)
        response["amphion_confidence"] = report
        if request.get("stream"):
            include_usage = bool((request.get("stream_options") or {}).get("include_usage"))
            return await _respond(send, 200, headers, as_stream(response, report, include_usage),
                                  b"text/event-stream; charset=utf-8")
        return await _respond(send, 200, headers, json.dumps(response, ensure_ascii=False).encode(), b"application/json")


def _replay(body, receive):
    """Deliver ``body`` once, then wait on the real client, so vLLM still sees a disconnect."""
    sent = False

    async def replay():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


async def _call(app, scope, receive):
    status, headers, parts = 500, [], []

    async def capture(message):
        nonlocal status, headers
        if message["type"] == "http.response.start":
            status, headers = message["status"], message.get("headers", [])
        elif message["type"] == "http.response.body":
            parts.append(message.get("body", b""))

    await app(scope, receive, capture)
    return status, headers, b"".join(parts)


async def _respond(send, status, headers, body, content_type=None):
    """Send ``body`` with the upstream headers, a fresh length and, if given, a new content type."""
    dropped = (b"content-length", b"content-type") if content_type else (b"content-length",)
    headers = [(k, v) for k, v in headers if k.lower() not in dropped]
    if content_type:
        headers.append((b"content-type", content_type))
    await send({"type": "http.response.start", "status": status,
                "headers": headers + [(b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})
