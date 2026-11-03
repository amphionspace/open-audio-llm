"""Greedy losslessness, draft acceptance and closed-loop speed of speculative-decoding variants.

Each variant is started from a delivery pack (``serve.sh asr|tsasr``) on one GPU, then receives
the same plain-ASR or TS-ASR requests at temperature 0 at each concurrency in turn. Recorded per
variant, mode and concurrency:
- raw outputs, completion tokens and request latency (``requests-<mode>-c<N>.jsonl``);
- vLLM's spec-decode counters over that phase: drafts, draft tokens, accepted tokens in total
  and per draft position (``rows.json``).
``compare_spec.py`` pairs the outputs across variants; base at one concurrency against base at
another gives the batch-nondeterminism floor.

Audio (through the Catalog): plain ASR from an AmphionEval ASR run's samples (``asr_per_source``
per source), TS-ASR from the v12 benchmark samples (``ts_count`` random items, the sample's own
instruction, an ``enroll_s`` enrollment as the pack protocol requires, mixture cut to
``ts_max_mix_s``).
"""
import argparse
import asyncio
import json
import os
import random
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stream_bench import SR, pct, stop_server, wandb_log, wav_b64  # noqa: E402

METRIC = re.compile(r'^(vllm:spec_decode_[a-z_]+?)(?:_total)?\{([^}]*)\}\s+([0-9.eE+-]+)$')


def load_ref(resolver, ref, seconds=None):
    """16 kHz mono int16 through the Catalog, cut or zero-padded to ``seconds`` when given."""
    from audio_data_contract import AudioRef

    cut = resolver.get_cut(AudioRef(**{k: ref[k] for k in ("dataset_id", "version", "split", "cut_id")}))
    if cut.sampling_rate != SR:
        cut = cut.resample(SR)
    audio = np.clip(cut.load_audio().mean(0) * 32768, -32768, 32767).astype(np.int16)
    if seconds is not None:
        count = int(seconds * SR)
        audio = np.pad(audio[:count], (0, max(0, count - len(audio))))
    return audio


def load_requests(p):
    """Requests with their audio already encoded, so loading never delays a measured phase."""
    from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver

    resolver = LhotseCatalogAudioResolver(p["catalog"], p["roots"])
    rng = random.Random(p["seed"])
    asr = {}
    for line in open(p["asr_samples"]):
        sample = json.loads(line)
        asr.setdefault(sample["source"], []).append(sample)
    def plain(sample):
        pcm = load_ref(resolver, sample["record"]["audio_slots"][0]["ref"])
        return {"id": sample["id"], "system": sample.get("context") or "",
                "data": wav_b64(pcm), "audio_s": len(pcm) / SR}

    def target(sample):
        slots = {s["name"]: s["ref"] for s in sample["record"]["audio_slots"]}
        pcm = np.concatenate([load_ref(resolver, slots["enrollment"], p["enroll_s"]),
                              load_ref(resolver, slots["mixture"])[:int(p["ts_max_mix_s"] * SR)]])
        return {"id": sample["id"], "system": sample["context"], "data": wav_b64(pcm),
                "audio_s": len(pcm) / SR}

    measured, rest = [], []
    for source in sorted(asr):
        chosen = rng.sample(asr[source], min(p["asr_per_source"], len(asr[source])))
        measured += chosen
        rest += [s for s in asr[source] if s not in chosen]
    ts = [json.loads(line) for line in open(p["ts_samples"])]
    ts_drawn = rng.sample(ts, p["ts_count"] + p["warmup_requests"])
    # Warm-up requests are disjoint from the measured ones: suffix decoding remembers outputs.
    return {"asr": [plain(s) for s in measured],
            "tsasr": [target(s) for s in ts_drawn[:p["ts_count"]]],
            "warmup-asr": [plain(s) for s in rng.sample(rest, p["warmup_requests"])],
            "warmup-tsasr": [target(s) for s in ts_drawn[p["ts_count"]:]]}


def port_free(port):
    with socket.socket() as probe:
        probe.settimeout(2)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def start_server(variant, mode, p, log_path):
    if not port_free(p["port"]):
        raise RuntimeError(f"port {p['port']} is in use; refusing to start a second server")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(p["gpu"]), PORT=str(p["port"]),
               GPU_UTIL=str(p["gpu_util"]), MAX_MODEL_LEN=str(p["max_model_len"]),
               AMPHION_TSASR_ENROLL_SEC=str(p["enroll_s"]), PYTHON=variant["python"],
               NAME=p["model_name"], HOST="127.0.0.1")
    env.pop("PYTHONPATH", None)
    env.update({k: str(v) for k, v in variant.get("environment", {}).items()})
    pack = variant.get("pack", p["pack"])
    command = ["bash", str(Path(pack) / "serve.sh"), mode, *p["server_arguments"],
               *variant.get("arguments", [])]
    log = open(log_path, "w")
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    return process, log, command


async def wait_ready(p, process):
    base = f"http://127.0.0.1:{p['port']}"
    deadline = time.time() + p["server_startup_timeout_s"]
    async with aiohttp.ClientSession() as http:
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited with {process.returncode}")
            try:
                async with http.get(f"{base}/v1/models") as r:
                    if r.status == 200:
                        async with http.get(f"{base}/version") as v:
                            return {"models": await r.json(), "version": await v.json()}
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(5)
    raise TimeoutError("server did not become ready")


async def spec_metrics(p):
    """Spec-decode counters; per-position counters keyed by their position label."""
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{p['port']}/metrics") as r:
            text = await r.text()
    values = {}
    for line in text.splitlines():
        match = METRIC.match(line)
        if not match or match.group(1).endswith("_created"):
            continue
        name, labels, value = match.groups()
        position = re.search(r'position="(\d+)"', labels)
        key = f"{name}[{position.group(1)}]" if position else name
        values[key] = values.get(key, 0.0) + float(value)
    return values


async def run_phase(url, p, requests, concurrency):
    records = [None] * len(requests)
    queue = asyncio.Queue()
    for index in range(len(requests)):
        queue.put_nowait(index)
    timeout = aiohttp.ClientTimeout(total=p["request_timeout_s"])
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=timeout) as http:

        async def worker():
            while not queue.empty():
                index = queue.get_nowait()
                request = requests[index]
                payload = {
                    "model": p["model_name"], "temperature": 0, "max_tokens": p["max_tokens"],
                    "messages": [
                        {"role": "system", "content": request["system"]},
                        {"role": "user", "content": [{"type": "input_audio", "input_audio": {
                            "data": request["data"], "format": "wav"}}]},
                    ],
                }
                started = time.perf_counter()
                async with http.post(url, json=payload) as response:
                    body = await response.json()
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {body}")
                elapsed = time.perf_counter() - started
                choice, usage = body["choices"][0], body.get("usage", {})
                records[index] = {"id": request["id"], "raw": choice["message"]["content"] or "",
                                  "finish_reason": choice.get("finish_reason"),
                                  "completion_tokens": usage.get("completion_tokens"),
                                  "prompt_tokens": usage.get("prompt_tokens"),
                                  "latency_ms": elapsed * 1000, "audio_s": request["audio_s"]}

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - started
    return records, wall


def wandb_row(row):
    """stream_bench.wandb_log's row shape: numeric metrics per phase and concurrency."""
    metrics = {k: v for k, v in row.items() if isinstance(v, (int, float)) and k != "concurrency"}
    spec = row["spec"]
    for key in ("mean_acceptance_length", "draft_acceptance_rate"):
        if key in spec:
            metrics[key] = spec[key]
    for position, rate in enumerate(spec.get("acceptance_rate_per_position", [])):
        metrics[f"acceptance_rate_pos{position}"] = rate
    return {"phase": row["mode"], "concurrency": row["concurrency"], "prompt": "all", "metrics": metrics}


def acceptance(before, after):
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    drafts = delta.get("vllm:spec_decode_num_drafts", 0.0)
    if not drafts:
        return delta
    accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    delta["mean_acceptance_length"] = 1 + accepted / drafts
    delta["draft_acceptance_rate"] = accepted / delta["vllm:spec_decode_num_draft_tokens"]
    positions = sorted(int(k.split("[")[1][:-1]) for k in delta
                       if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos["))
    delta["acceptance_rate_per_position"] = [
        delta[f"vllm:spec_decode_num_accepted_tokens_per_pos[{i}]"] / drafts for i in positions]
    return delta


async def measure(url, p, variant, mode, concurrency, requests, vdir):
    # Longer phases at high concurrency give stable throughput; only for variants without
    # cross-request state (a repeated request is otherwise the same work).
    requests = requests * p.get("phase_repeats", {}).get(concurrency, 1)
    before = await spec_metrics(p)
    records, wall = await run_phase(url, p, requests, concurrency)
    after = await spec_metrics(p)
    (vdir / f"requests-{mode}-c{concurrency}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    latency = [r["latency_ms"] for r in records]
    tokens = sum(r["completion_tokens"] or 0 for r in records)
    row = {"mode": mode, "concurrency": concurrency, "requests": len(records),
           "wall_s": wall, "completion_tokens": tokens, "output_tokens_per_s": tokens / wall,
           "audio_seconds_per_wall_second": sum(r["audio_s"] for r in records) / wall,
           "latency_ms_p50": pct(latency, .5), "latency_ms_p95": pct(latency, .95),
           "latency_ms_mean": statistics.fmean(latency), "spec": acceptance(before, after)}
    print(json.dumps({"variant": variant["name"], **{k: v for k, v in row.items() if k != "spec"},
                      "mean_acceptance_length": row["spec"].get("mean_acceptance_length"),
                      "per_position": row["spec"].get("acceptance_rate_per_position")}), flush=True)
    return row


async def main_async(p, lane):
    """One lane: its GPU and port, its variants one after another."""
    p = {**p, "gpu": lane["gpu"], "port": lane["port"]}
    out = Path(p["output"])
    out.mkdir(parents=True, exist_ok=True)
    requests = load_requests(p)
    url = f"http://127.0.0.1:{p['port']}/v1/chat/completions"
    variants = {variant["name"]: variant for variant in p["variants"]}
    for variant in (variants[name] for name in lane["variants"]):
        for mode in p["modes"]:
            vdir = out / variant["name"]
            vdir.mkdir(exist_ok=True)
            rows = []
            # Suffix decoding keeps a cross-request tree of past outputs; replaying the same requests
            # at the next concurrency would hit it, so such variants restart for every phase.
            groups = ([[c] for c in p["concurrency"]] if variant.get("fresh_server_per_phase")
                      else [p["concurrency"]])
            for group in groups:
                suffix = "" if len(groups) == 1 else f"-c{group[0]}"
                process, log, command = start_server(variant, mode, p, vdir / f"server-{mode}{suffix}.log")
                try:
                    runtime = await wait_ready(p, process)
                    (vdir / f"server-runtime-{mode}{suffix}.json").write_text(
                        json.dumps({"command": command, **runtime}, indent=2))
                    # Warm up graphs and the audio path; not measured.
                    await run_phase(url, p, requests[f"warmup-{mode}"], 4)
                    for concurrency in group:
                        rows.append(await measure(url, p, variant, mode, concurrency, requests[mode], vdir))
                except Exception as error:
                    rows.append({"mode": mode, "error": repr(error)})
                    print(json.dumps({"variant": variant["name"], "mode": mode, "error": repr(error)}),
                          flush=True)
                finally:
                    stop_server(process, log)
            (vdir / f"rows-{mode}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
            if p.get("wandb", {}).get("enabled"):
                wandb_log({**p, "wandb": {**p["wandb"], "name": f"{p['wandb']['name']}-{mode}"}},
                          variant["name"], [wandb_row(row) for row in rows if "error" not in row])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lane", type=int, required=True, help="Index into parameters.lanes")
    args = parser.parse_args()
    p = yaml.safe_load(args.config.read_text())["parameters"]
    asyncio.run(main_async(p, p["lanes"][args.lane]))


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
