import asyncio
import json

import pytest

from open_audio_llm.integrations.vllm import confidence_gate as gate

AUDIO = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}]}]
HEADER = ["language", " Chinese", "<asr_text>"]


def completion(tokens, logprobs, finish="stop", prompt_logprobs=None):
    content = "".join(t for t in tokens if t not in gate.STOP_TOKENS)
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "asr",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish,
                     "logprobs": {"content": [{"token": t, "logprob": lp, "bytes": None, "top_logprobs": []}
                                              for t, lp in zip(tokens, logprobs)]}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": len(tokens), "total_tokens": 50 + len(tokens)},
        "prompt_logprobs": prompt_logprobs,
    }


class FakeVLLM:
    """Records the request body it receives and answers with a canned response."""

    def __init__(self, response, status=200):
        self.response, self.status, self.bodies = response, status, []

    async def __call__(self, scope, receive, send):
        message = await receive()
        self.bodies.append(message["body"])
        body = json.dumps(self.response).encode()
        await send({"type": "http.response.start", "status": self.status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def run(app, request, path=gate.PATH, method="POST"):
    sent = []

    async def receive():
        return {"type": "http.request", "body": json.dumps(request).encode(), "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app({"type": "http", "method": method, "path": path}, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], dict(start["headers"]), body


def events(body):
    lines = [l[len("data: "):] for l in body.decode().split("\n\n") if l]
    assert lines[-1] == "[DONE]"
    return [json.loads(l) for l in lines[:-1]]


@pytest.fixture
def threshold(monkeypatch):
    monkeypatch.setenv(gate.THRESHOLD_ENV, "-0.8")


def test_threshold_is_required(monkeypatch):
    monkeypatch.delenv(gate.THRESHOLD_ENV, raising=False)
    with pytest.raises(RuntimeError, match=gate.THRESHOLD_ENV):
        gate.ConfidenceGate(FakeVLLM({}))


def test_confident_answer_passes_unchanged_without_logprobs(threshold):
    fake = FakeVLLM(completion(HEADER + ["我", "叫", "张三", "<|im_end|>"], [0, 0, 0, -0.1, -0.05, -0.2, -0.3]))
    status, headers, body = run(gate.ConfidenceGate(fake), {"model": "asr", "messages": AUDIO, "temperature": 0})
    upstream = json.loads(fake.bodies[0])
    assert (upstream["stream"], upstream["logprobs"], upstream["top_logprobs"]) == (False, True, 0)
    assert "prompt_logprobs" not in upstream
    response = json.loads(body)
    assert response["choices"][0]["message"]["content"] == "language Chinese<asr_text>我叫张三"
    assert response["choices"][0]["logprobs"] is None and "prompt_logprobs" not in response
    assert response["amphion_confidence"]["rejected"] is False
    assert response["amphion_confidence"]["head_tokens"] == 3
    assert headers[b"content-length"] == str(len(body)).encode()


def test_unconfident_head_becomes_no_speech(threshold):
    fake = FakeVLLM(completion(HEADER + ["他", "现在", "主要", "<|im_end|>"], [0, 0, 0, -1.5, -1.2, -2.0, -0.1]))
    _, _, body = run(gate.ConfidenceGate(fake), {"model": "asr", "messages": AUDIO})
    response = json.loads(body)
    choice = response["choices"][0]
    assert (choice["message"]["content"], choice["finish_reason"]) == (gate.NO_SPEECH, "stop")
    assert response["amphion_confidence"]["rule"] == "head"
    assert response["amphion_confidence"]["score"] == pytest.approx(-1.5667, abs=1e-4)


def test_head_window_ignores_later_confident_loop(threshold):
    tokens = HEADER + ["那", "我", "就"] * 4 + ["对"] * 200
    logprobs = [0, 0, 0] + [-1.6] * 12 + [-0.01] * 200  # head mean (12 x -1.6 + 4 x -0.01) / 16 = -1.2
    _, _, body = run(gate.ConfidenceGate(FakeVLLM(completion(tokens, logprobs, finish="length"))), {"model": "asr", "messages": AUDIO})
    report = json.loads(body)["amphion_confidence"]
    assert report["rejected"] and report["rule"] == "head" and report["head_tokens"] == gate.HEAD_TOKENS


def test_confident_loop_at_token_limit_is_rejected(threshold):
    tokens = HEADER + ["对"] * 120
    _, _, body = run(gate.ConfidenceGate(FakeVLLM(completion(tokens, [0, 0, 0] + [-0.05] * 120, finish="length"))),
                     {"model": "asr", "messages": AUDIO})
    response = json.loads(body)
    assert response["amphion_confidence"]["rule"] == "loop"
    assert response["choices"][0]["finish_reason"] == "stop"


def test_confident_repeat_that_stops_is_kept(threshold):
    tokens = HEADER + ["对"] * 10 + ["<|im_end|>"]
    _, _, body = run(gate.ConfidenceGate(FakeVLLM(completion(tokens, [0, 0, 0] + [-0.05] * 11))), {"model": "asr", "messages": AUDIO})
    assert json.loads(body)["amphion_confidence"]["rejected"] is False


def test_empty_answer_is_left_alone(threshold):
    tokens = ["language", " None", "<asr_text>", "<|im_end|>"]
    _, _, body = run(gate.ConfidenceGate(FakeVLLM(completion(tokens, [-0.3, -0.2, 0, -0.01]))), {"model": "asr", "messages": AUDIO})
    response = json.loads(body)
    assert response["choices"][0]["message"]["content"] == gate.NO_SPEECH
    assert response["amphion_confidence"] == {"threshold": -0.8, "head_tokens": 0, "prefix": False, "score": None,
                                              "rule": None, "rejected": False}


def test_streaming_client_gets_equivalent_chunks_and_usage(threshold):
    fake = FakeVLLM(completion(HEADER + ["他", "现在", "<|im_end|>"], [0, 0, 0, -2.0, -2.0, -0.1]))
    status, headers, body = run(gate.ConfidenceGate(fake), {"model": "asr", "messages": AUDIO, "stream": True,
                                                         "stream_options": {"include_usage": True}})
    upstream = json.loads(fake.bodies[0])
    assert upstream["stream"] is False and "stream_options" not in upstream
    assert headers[b"content-type"].startswith(b"text/event-stream")
    chunks = events(body)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]) == gate.NO_SPEECH
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "stop"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 6
    assert chunks[-2]["amphion_confidence"]["rejected"] is True


def test_prefix_continuation_scores_the_forced_prefix(threshold):
    request = {"model": "asr", "messages": AUDIO + [{"role": "assistant", "content": "language Chinese<asr_text>那可以"}],
               "continue_final_message": True, "add_generation_prompt": False}
    prompt = [None, {"9": {"logprob": -0.5, "rank": 1, "decoded_token": "<|im_start|>"}},
              {"151704": {"logprob": -0.1, "rank": 1, "decoded_token": "<asr_text>"}},
              {"1": {"logprob": -2.0, "rank": 3, "decoded_token": "那"}},
              {"2": {"logprob": -1.8, "rank": 2, "decoded_token": "可以"}}]
    fake = FakeVLLM(completion(["报", "个", "志愿", "<|im_end|>"], [-0.9, -0.4, -0.6, -0.1], prompt_logprobs=prompt))
    _, _, body = run(gate.ConfidenceGate(fake), request)
    assert json.loads(fake.bodies[0])["prompt_logprobs"] == 0
    response = json.loads(body)
    report = response["amphion_confidence"]
    assert report["prefix"] and report["head_tokens"] == 5 and report["score"] == pytest.approx(-1.14)
    assert response["choices"][0]["message"]["content"] == "" and report["rejected"]
    assert "prompt_logprobs" not in response


def test_client_logprobs_are_kept_when_passed(threshold):
    tokens = HEADER + ["好", "<|im_end|>"]
    _, _, body = run(gate.ConfidenceGate(FakeVLLM(completion(tokens, [0, 0, 0, -0.1, -0.1]))),
                     {"model": "asr", "messages": AUDIO, "logprobs": True, "top_logprobs": 2})
    assert len(json.loads(body)["choices"][0]["logprobs"]["content"]) == 5


def test_requests_without_audio_and_other_routes_pass_through(threshold):
    fake = FakeVLLM({"data": []})
    request = {"model": "asr", "messages": [{"role": "user", "content": "hi"}]}
    _, _, body = run(gate.ConfidenceGate(fake), request)
    assert json.loads(fake.bodies[0]) == request and json.loads(body) == {"data": []}
    run(gate.ConfidenceGate(fake), {}, path="/v1/models", method="GET")
    assert len(fake.bodies) == 2


def test_upstream_errors_are_forwarded(threshold):
    _, _, body = run(gate.ConfidenceGate(FakeVLLM({"error": {"message": "bad audio"}}, status=400)),
                     {"model": "asr", "messages": AUDIO, "stream": True})
    assert json.loads(body) == {"error": {"message": "bad audio"}}
