"""Replay audiollm-go's target-ASR streaming schedule against local vLLM servers.

Mirrors origin/main 0966b463 of audiollm-go:
- previews: internal/transport/astv3ws/pseudo_partial.go (first snapshot immediately,
  later ones only after 1.5 s of new audio; one active decode, one pending slot) and
  internal/asr/amphionv6/target_incremental.go (6 s stable windows, <=7 s tail,
  exact-window reuse, 0.5 s virtual silence on previews);
- finals: target_recovery.go / client.go (27 s target window incl. the 5 s reference,
  balanced chunks, +0.5 s silent tail retry when the text is empty);
- request body: client.go transcribe (temperature 0, max_tokens min(final,128),
  repetition_detection 4/32/4).

Mode "stream" runs whole sessions; mode "prefill" sends max_tokens=1 requests to
isolate prompt-length cost. Each server variant is started from the delivery pack,
measured, and stopped by its own process group.
"""
import argparse
import asyncio
import base64
import gzip
import io
import itertools
import json
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import time
import unicodedata
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf
import yaml

SR = 16000
NONCE = itertools.count(1)


def wav_b64(pcm):
    buf = io.BytesIO()
    sf.write(buf, pcm, SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def load_pcm(path, seconds=None):
    audio, rate = sf.read(path, dtype="int16")
    if rate != SR or audio.ndim != 1:
        raise ValueError(f"{path}: expected 16 kHz mono")
    if seconds is not None:
        count = int(seconds * SR)
        audio = np.pad(audio[:count], (0, max(0, count - len(audio))))
    return audio


def audio_path(p, ref, cache):
    """Resolve a Catalog audio ref through its dataset version's audio index."""
    key = (ref["dataset_id"], ref["version"])
    if key not in cache:
        root = Path(p["datasets_root"]) / ref["dataset_id"]
        with gzip.open(root / "versions" / ref["version"] / "audio-index.jsonl.gz", "rt") as index:
            cache[key] = {row["cut_id"]: root / row["relative_path"] for row in map(json.loads, index)}
    return cache[key][ref["cut_id"]]


def load_items(p):
    """Real-meeting v12 items: enrollment cut + mixture cut + reference."""
    cache = {}
    items = []
    for line in open(p["samples"]):
        sample = json.loads(line)
        split = sample["id"].split(":")[0]
        if not any(split.startswith(prefix) for prefix in p["split_prefixes"]):
            continue
        slots = {s["name"]: s["ref"] for s in sample["record"]["audio_slots"]}
        items.append({
            "id": sample["id"],
            "negative": split.endswith("-neg"),
            "reference": sample["reference"],
            "enroll": audio_path(p, slots["enrollment"], cache),
            "mix": audio_path(p, slots["mixture"], cache),
            "duration": slots["mixture"]["duration"],
        })
    random.Random(p["seed"]).shuffle(items)
    return items


def normalize(text):
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).lower()
                   if not unicodedata.category(ch).startswith(("P", "Z", "S")))


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def split_raw(raw):
    meta, sep, text = raw.partition("<asr_text>")
    return (meta, text.strip()) if sep else ("", raw.strip())


class Client:
    def __init__(self, session, url, model, max_tokens):
        self.session, self.url, self.model, self.max_tokens = session, url, model, max_tokens
        self.records = []

    async def transcribe(self, audio, prompt, kind, tag, max_tokens=None, unique=True):
        if unique:
            # Focus serves with --mm-processor-cache-gb 0 (not usable with the pack runtime), so make
            # every request's audio hash unique instead: one sample within the last 50 ms moves by a
            # few LSB (about -90 to -70 dBFS; (position, step) never repeats). Without this, arms
            # replaying the same audio hit vLLM's processor cache.
            count = next(NONCE)  # process-wide, so clients created per phase never collide
            audio = audio.copy()
            index = len(audio) - 1 - count % min(len(audio), 800)
            audio[index] = np.clip(int(audio[index]) + 1 + count // 800, -32768, 32767)
        payload = {
            "model": self.model, "temperature": 0, "max_tokens": max_tokens or self.max_tokens,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "input_audio",
                                              "input_audio": {"data": wav_b64(audio), "format": "wav"}}]},
            ],
            "repetition_detection": {"min_pattern_size": 4, "max_pattern_size": 32, "min_count": 4},
        }
        started = time.perf_counter()
        async with self.session.post(self.url, json=payload) as response:
            body = await response.json()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {body}")
        elapsed = time.perf_counter() - started
        choice, usage = body["choices"][0], body.get("usage", {})
        raw = choice["message"]["content"] or ""
        self.records.append({**tag, "kind": kind, "latency_ms": elapsed * 1000, "audio_s": len(audio) / SR,
                             "prompt_tokens": usage.get("prompt_tokens"),
                             "completion_tokens": usage.get("completion_tokens"),
                             "finish_reason": choice.get("finish_reason"), "raw": raw})
        return raw


async def preview(client, ref, pcm, prompt, state, tag, p):
    stable, maximum = int(p["stable_s"] * SR), int(p["max_window_s"] * SR)
    silence = np.zeros(int(p["preview_silence_s"] * SR), dtype=np.int16)
    new_state, texts, offset = [], [], 0
    while offset < len(pcm):
        end = len(pcm) if len(pcm) - offset <= maximum else offset + stable
        key = (offset, end)
        index = len(new_state)
        if index < len(state) and state[index][0] == key:
            raw = state[index][1]
        else:
            raw = await client.transcribe(np.concatenate([ref, pcm[offset:end], silence]), prompt, "preview", tag)
        new_state.append((key, raw))
        _, text = split_raw(raw)
        if text:
            texts.append(text)
        offset = end
    return new_state, " ".join(texts)


async def final(client, ref, pcm, prompt, tag, p):
    window = int(p["final_window_s"] * SR) - len(ref)
    chunks = -(-len(pcm) // window)
    window = -(-len(pcm) // chunks)
    texts = []
    for offset in range(0, len(pcm), window):
        audio = np.concatenate([ref, pcm[offset:offset + window]])
        raw = await client.transcribe(audio, prompt, "final", tag)
        if not split_raw(raw)[1]:
            padding = min(int(p["final_empty_retry_s"] * SR), int(p["final_window_s"] * SR) - len(audio))
            if padding > 0:
                raw = await client.transcribe(np.concatenate([audio, np.zeros(padding, dtype=np.int16)]),
                                              prompt, "final_retry", tag)
        text = split_raw(raw)[1]
        if text:
            texts.append(text)
    return " ".join(texts)


async def final_windowed(client, ref, pcm, prompt, state, tag, p):
    """Runtime arm: the final reuses the preview's stable windows and decodes only the rest.

    Windows follow the preview rule (6 s stable, <=7 s tail); a window is reused only when the
    last completed preview decoded exactly the same span. New windows are decoded final-style
    (no virtual silence, empty-text retry).
    """
    stable, maximum = int(p["stable_s"] * SR), int(p["max_window_s"] * SR)
    texts, offset, index = [], 0, 0
    while offset < len(pcm):
        end = len(pcm) if len(pcm) - offset <= maximum else offset + stable
        if index < len(state) and state[index][0] == (offset, end):
            raw = state[index][1]
        else:
            audio = np.concatenate([ref, pcm[offset:end]])
            raw = await client.transcribe(audio, prompt, "final", tag)
            if not split_raw(raw)[1]:
                padding = np.zeros(int(p["final_empty_retry_s"] * SR), dtype=np.int16)
                raw = await client.transcribe(np.concatenate([audio, padding]), prompt, "final_retry", tag)
        text = split_raw(raw)[1]
        if text:
            texts.append(text)
        offset, index = end, index + 1
    return " ".join(texts)


async def run_segment(client, item, arm, p, segments, sid):
    """One VAD segment. ``arm`` picks the prompt and the runtime final policy:
    final_mode full (main) or windowed; final_start_s is when the final decode starts after the
    audio ends (main: at the endpoint decision; silence-time preparation: earlier). The result is
    published no earlier than the endpoint decision."""
    prompt = arm["system"]
    ref, pcm = load_pcm(item["enroll"], p["enroll_s"]), load_pcm(item["mix"])
    duration = len(pcm) / SR
    tag = {"session": sid, "segment": item["id"], "prompt": arm["name"]}
    t0 = time.perf_counter()
    events, state = [], []
    pending = {"end": None}
    wake = asyncio.Event()

    async def worker():
        nonlocal state
        while True:
            await wake.wait()
            wake.clear()
            end, pending["end"] = pending["end"], None
            if end is None:
                continue
            state, text = await preview(client, ref, pcm[:end], prompt, state, tag, p)
            events.append((time.perf_counter() - t0, end / SR, text))

    task = asyncio.create_task(worker())
    last_end, tick = 0, p["first_partial_s"]
    while tick < duration:
        await asyncio.sleep(max(0.0, t0 + tick - time.perf_counter()))
        end = int(tick * SR)
        if not last_end or end - last_end >= int(p["min_new_audio_s"] * SR):
            last_end, pending["end"] = end, end
            wake.set()
        tick += p["partial_interval_s"]
    # main starts the final at the endpoint decision; advancing the segment cancels the preview.
    start_after = arm.get("final_start_s", p["endpoint_delay_s"])
    await asyncio.sleep(max(0.0, t0 + duration + start_after - time.perf_counter()))
    task.cancel()
    final_start = time.perf_counter()
    if arm.get("final_mode", "full") == "windowed":
        text = await final_windowed(client, ref, pcm, prompt, state, tag, p)
    else:
        text = await final(client, ref, pcm, prompt, tag, p)
    decoded = time.perf_counter()
    done = max(decoded, t0 + duration + p["endpoint_delay_s"])
    texts = [e for e in events if e[2]]
    changes = [e[0] for i, e in enumerate(texts) if i == 0 or e[2] != texts[i - 1][2]]
    ref_text, hyp = normalize(item["reference"]), normalize(text)
    segments.append({
        **tag, "negative": item["negative"], "duration_s": duration,
        "first_text_ms": texts[0][0] * 1000 if texts else None,
        "first_response_ms": events[0][0] * 1000 if events else None,
        # How stale each preview is when it returns: completion time minus the end of the audio it saw.
        "preview_lag_ms": [(e[0] - e[1]) * 1000 for e in events],
        "text_gaps_ms": [(b - a) * 1000 for a, b in zip(changes, changes[1:])],
        "final_decode_ms": (decoded - final_start) * 1000,
        "stop_to_final_ms": (done - t0 - duration) * 1000,
        "previews": len(events), "final": text, "reference": item["reference"],
        "errors": edit_distance(ref_text, hyp) if not item["negative"] else None,
        "ref_chars": len(ref_text), "false_output": bool(hyp) if item["negative"] else None,
    })


def pct(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summarize(segments, requests):
    out = {}
    for prompt in sorted({s["prompt"] for s in segments}):
        seg = [s for s in segments if s["prompt"] == prompt]
        req = [r for r in requests if r["prompt"] == prompt]
        pos = [s for s in seg if not s["negative"]]
        gaps = [g for s in seg for g in s["text_gaps_ms"]]
        lags = [g for s in seg for g in s["preview_lag_ms"]]
        row = {"segments": len(seg), "requests": len(req)}
        for name, values in [("first_text_ms", [s["first_text_ms"] for s in pos]),
                             ("text_gap_ms", gaps),
                             ("preview_lag_ms", lags),
                             ("final_decode_ms", [s["final_decode_ms"] for s in seg]),
                             ("stop_to_final_ms", [s["stop_to_final_ms"] for s in seg])]:
            row[f"{name}_p50"], row[f"{name}_p95"] = pct(values, .5), pct(values, .95)
        for kind in ("preview", "final"):
            lat = [r["latency_ms"] for r in req if r["kind"] == kind]
            row[f"{kind}_latency_ms_p50"], row[f"{kind}_latency_ms_p95"] = pct(lat, .5), pct(lat, .95)
            row[f"{kind}_prompt_tokens_mean"] = statistics.fmean(
                [r["prompt_tokens"] for r in req if r["kind"] == kind] or [0])
            row[f"{kind}_completion_tokens_mean"] = statistics.fmean(
                [r["completion_tokens"] for r in req if r["kind"] == kind] or [0])
        errors, chars = sum(s["errors"] for s in pos), sum(s["ref_chars"] for s in pos)
        neg = [s for s in seg if s["negative"]]
        row["final_cer"] = errors / chars if chars else None
        row["negative_false_output_rate"] = sum(s["false_output"] for s in neg) / len(neg) if neg else None
        out[prompt] = row
    return out


async def run_stream(url, model, items, arms, concurrency, seconds, p):
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=p["request_timeout_s"])
    segments = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as http:
        client = Client(http, url, model, p["max_tokens"])
        deadline = time.perf_counter() + seconds
        cursor = {"i": 0}

        async def session_loop(sid):
            # Sessions alternate arms per segment so arms share the same load.
            while time.perf_counter() < deadline:
                index = cursor["i"]
                cursor["i"] += 1
                # Consecutive segments replay one item under every arm, so arms are paired by item.
                item = items[(index // len(arms)) % len(items)]
                await run_segment(client, item, arms[index % len(arms)], p, segments, sid)

        started = time.perf_counter()
        await asyncio.gather(*(session_loop(i) for i in range(concurrency)))
        wall = time.perf_counter() - started
    return segments, client.records, wall


async def run_prefill(url, model, items, prompts, concurrency, count, p):
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=120)) as http:
        client = Client(http, url, model, 1)
        queue = asyncio.Queue()
        for i in range(count):
            item = items[(i // len(prompts)) % len(items)]
            prompt = prompts[i % len(prompts)]
            queue.put_nowait((item, prompt))

        async def worker(wid):
            while not queue.empty():
                item, prompt = queue.get_nowait()
                # repeat_audio arms resend one fixed clip unchanged, so every request after the first
                # hits vLLM's processor cache: the difference to a unique-audio arm is preprocessing.
                repeat = prompt.get("repeat_audio", False)
                item = items[0] if repeat else item
                ref = load_pcm(item["enroll"], p["enroll_s"])
                window = load_pcm(item["mix"])[:int(p["prefill_window_s"] * SR)]
                await client.transcribe(np.concatenate([ref, window]), prompt["system"], "prefill",
                                        {"session": wid, "segment": item["id"], "prompt": prompt["name"]}, 1,
                                        unique=not repeat)

        await asyncio.gather(*(worker(i) for i in range(concurrency)))
    rows = {}
    for prompt in prompts:
        lat = [r["latency_ms"] for r in client.records if r["prompt"] == prompt["name"]]
        toks = [r["prompt_tokens"] for r in client.records if r["prompt"] == prompt["name"]]
        rows[prompt["name"]] = {"requests": len(lat), "latency_ms_p50": pct(lat, .5), "latency_ms_p95": pct(lat, .95),
                                "latency_ms_mean": statistics.fmean(lat), "prompt_tokens_mean": statistics.fmean(toks)}
    return rows, client.records


def port_free(port):
    with socket.socket() as probe:
        probe.settimeout(2)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return False
    # Only server processes count; shells whose command text merely mentions the port do not.
    listing = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout.splitlines()
    server = re.compile(rf"(serve_qwen3\.py|vllm serve|open-audio-llm serve|vllm\.entrypoints).*--port[ =]{port}\b")
    return not any(server.search(line) for line in listing)


def start_server(variant, p, log_path):
    port = p["port"]
    if not port_free(port):
        raise RuntimeError(f"port {port} is in use; refusing to start a second server")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(p["gpu"]), PORT=str(port), GPU_UTIL=str(p["gpu_util"]),
               MAX_MODEL_LEN=str(p["max_model_len"]), AMPHION_TSASR_ENROLL_SEC=str(p["enroll_s"]),
               PYTHON=variant["python"], NAME=p["model_name"], HOST="127.0.0.1")
    env.pop("PYTHONPATH", None)
    env.update({k: str(v) for k, v in variant.get("environment", {}).items()})
    pack = variant.get("pack", p["pack"])
    command = ["bash", str(Path(pack) / "serve.sh"), "tsasr", *p["server_arguments"], *variant.get("arguments", [])]
    log = open(log_path, "w")
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process, log, command


async def wait_ready(p, process, timeout):
    base = f"http://127.0.0.1:{p['port']}"
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as http:
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited with {process.returncode}")
            try:
                async with http.get(f"{base}/v1/models") as r:
                    if r.status == 200:
                        models = await r.json()
                        async with http.get(f"{base}/version") as v:
                            version = await v.json()
                        return {"models": models, "version": version}
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(5)
    raise TimeoutError("server did not become ready")


async def scrape_metrics(p):
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{p['port']}/metrics") as r:
            text = await r.text()
    wanted = ("vllm:request_prefill_time_seconds_sum", "vllm:request_prefill_time_seconds_count",
              "vllm:request_decode_time_seconds_sum", "vllm:request_queue_time_seconds_sum",
              "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count",
              "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
              "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
              "vllm:spec_decode_num_accepted_tokens_total")
    values = {}
    for line in text.splitlines():
        if line.startswith(wanted):
            name, _, value = line.rpartition(" ")
            key = name.split("{")[0]
            values[key] = values.get(key, 0.0) + float(value)
    return values


def stop_server(process, log):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    log.close()


def wandb_log(p, variant, rows):
    import wandb
    os.environ.setdefault("WANDB_MODE", p["wandb"]["mode"])
    run = wandb.init(entity=p["wandb"]["entity"], project=p["wandb"]["project"],
                     name=f"{p['wandb']['name']}-{variant}", dir=p["output"], config={"variant": variant},
                     reinit="finish_previous")
    table = wandb.Table(columns=["phase", "concurrency", "prompt", "metric", "value"])
    for row in rows:
        for metric, value in row["metrics"].items():
            if isinstance(value, (int, float)) and value is not None:
                table.add_data(row["phase"], row["concurrency"], row["prompt"], metric, float(value))
                run.summary[f"{row['phase']}/c{row['concurrency']}/{row['prompt']}/{metric}"] = value
    run.log({"results": table})
    run.finish()


async def main_async(p):
    out = Path(p["output"])
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(p)
    url = f"http://127.0.0.1:{p['port']}/v1/chat/completions"
    summary = {"items": len(items), "variants": {}}
    for variant in p["variants"]:
        name = variant["name"]
        vdir = out / name
        vdir.mkdir(exist_ok=True)
        rows = []
        try:
            process, log, command = start_server(variant, p, vdir / "server.log")
        except Exception as error:  # Keep measuring the other variants.
            summary["variants"][name] = {"error": repr(error)}
            continue
        try:
            runtime = await wait_ready(p, process, p["server_startup_timeout_s"])
            (vdir / "server-runtime.json").write_text(json.dumps({"command": command, **runtime}, indent=2))
            # Warm up graphs and the audio path before measuring.
            await run_prefill(url, p["model_name"], items, p["prompts"] or p["stream_arms"], 4, 16, p)
            if p.get("identity_requests"):
                # Same requests, full decode, sequential: raw outputs are compared across variants.
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as http:
                    client = Client(http, url, p["model_name"], p["max_tokens"])
                    for item in items[:p["identity_requests"]]:
                        ref = load_pcm(item["enroll"], p["enroll_s"])
                        mix = load_pcm(item["mix"])[:int((p["final_window_s"] - p["enroll_s"]) * SR)]
                        await client.transcribe(np.concatenate([ref, mix]), p["stream_arms"][0]["system"], "identity",
                                                {"session": 0, "segment": item["id"], "prompt": "identity"}, unique=False)
                (vdir / "identity.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in client.records))
            for concurrency in p.get("prefill_concurrency", []):
                before = await scrape_metrics(p)
                result, records = await run_prefill(url, p["model_name"], items, p["prompts"], concurrency,
                                                     p["prefill_requests"], p)
                after = await scrape_metrics(p)
                (vdir / f"prefill-c{concurrency}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
                for prompt, metrics in result.items():
                    rows.append({"phase": "prefill", "concurrency": concurrency, "prompt": prompt, "metrics": metrics})
                rows.append({"phase": "prefill", "concurrency": concurrency, "prompt": "server",
                             "metrics": {k: after.get(k, 0) - before.get(k, 0) for k in after}})
            for concurrency in p.get("stream_concurrency", []):
                before = await scrape_metrics(p)
                segments, records, wall = await run_stream(url, p["model_name"], items,
                                                           p.get("stream_arms") or p["stream_prompts"],
                                                           concurrency, p["stream_seconds"], p)
                after = await scrape_metrics(p)
                (vdir / f"stream-c{concurrency}-segments.jsonl").write_text(
                    "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in segments))
                (vdir / f"stream-c{concurrency}-requests.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
                for prompt, metrics in summarize(segments, records).items():
                    metrics["audio_seconds_per_wall_second"] = sum(
                        s["duration_s"] for s in segments if s["prompt"] == prompt) / wall
                    rows.append({"phase": "stream", "concurrency": concurrency, "prompt": prompt, "metrics": metrics})
                server = {k: after.get(k, 0) - before.get(k, 0) for k in after}
                if server.get("vllm:spec_decode_num_draft_tokens_total"):
                    server["spec_acceptance_rate"] = (server["vllm:spec_decode_num_accepted_tokens_total"]
                                                      / server["vllm:spec_decode_num_draft_tokens_total"])
                rows.append({"phase": "stream", "concurrency": concurrency, "prompt": "server", "metrics": server})
                print(json.dumps({"variant": name, "concurrency": concurrency,
                                  **{r["prompt"]: r["metrics"] for r in rows if r["phase"] == "stream"
                                     and r["concurrency"] == concurrency}}, ensure_ascii=False), flush=True)
        except Exception as error:
            rows.append({"phase": "error", "concurrency": 0, "prompt": "", "metrics": {}, "error": repr(error)})
            print(json.dumps({"variant": name, "error": repr(error)}), flush=True)
        finally:
            stop_server(process, log)
        (vdir / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        summary["variants"][name] = rows
        if p["wandb"]["enabled"]:
            wandb_log(p, name, rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parameters = yaml.safe_load(parser.parse_args().config.read_text())["parameters"]
    asyncio.run(main_async(parameters))


if __name__ == "__main__":
    main()
