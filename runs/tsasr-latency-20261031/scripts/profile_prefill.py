"""Profile single-request prefill (max_tokens=1) with vLLM's torch profiler.

Two profiled phases on one delivery-pack server: unique audio (encoder runs) and one repeated
clip (encoder output reused from vLLM's cache). The difference between the two CUDA-time tables
is the per-request audio path. Server start/stop and request format come from stream_bench.py.
"""
import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path

import aiohttp
import numpy as np
import yaml

import stream_bench as bench


async def phase(p, client, items, repeat, count, out, name):
    base = f"http://127.0.0.1:{p['port']}"
    async with aiohttp.ClientSession() as http:
        async with http.post(f"{base}/start_profile") as r:
            r.raise_for_status()
    latencies = []
    for i in range(count):
        item = items[0] if repeat else items[i + 1]
        ref = bench.load_pcm(item["enroll"], p["enroll_s"])
        window = bench.load_pcm(item["mix"])[:int(p["prefill_window_s"] * bench.SR)]
        started = time.perf_counter()
        await client.transcribe(np.concatenate([ref, window]), p["system"], "profile",
                                {"session": 0, "segment": item["id"], "prompt": name}, 1, unique=not repeat)
        latencies.append((time.perf_counter() - started) * 1000)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as http:
        async with http.post(f"{base}/stop_profile") as r:
            r.raise_for_status()
    # vLLM writes the trace and CUDA-time table asynchronously; move them into the phase folder.
    target = out / name
    target.mkdir(exist_ok=True)
    source = Path(p["profile_dir"])
    deadline = time.time() + 600
    while time.time() < deadline and not any(source.glob("*.txt")):
        await asyncio.sleep(5)
    await asyncio.sleep(10)
    for path in list(source.iterdir()):
        shutil.move(str(path), target / path.name)
    return {"requests": count, "latency_ms_p50": bench.pct(latencies, .5), "latency_ms_mean": float(np.mean(latencies))}


async def main_async(p):
    out = Path(p["output"])
    out.mkdir(parents=True, exist_ok=True)
    Path(p["profile_dir"]).mkdir(parents=True, exist_ok=True)
    items = bench.load_items(p)
    variant = dict(p["variant"])
    variant["arguments"] = list(variant.get("arguments", [])) + [
        "--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": p["profile_dir"],
                                         "torch_profiler_with_stack": True, "torch_profiler_record_shapes": True})]
    process, log, command = bench.start_server(variant, p, out / "server.log")
    result = {"command": command}
    try:
        result.update(await bench.wait_ready(p, process, p["server_startup_timeout_s"]))
        url = f"http://127.0.0.1:{p['port']}/v1/chat/completions"
        async with aiohttp.ClientSession() as http:
            client = bench.Client(http, url, p["model_name"], 1)
            for i in range(8):  # warm-up outside the profile window
                item = items[100 + i]
                await client.transcribe(np.concatenate([bench.load_pcm(item["enroll"], p["enroll_s"]),
                                                        bench.load_pcm(item["mix"])[:int(p["prefill_window_s"] * bench.SR)]]),
                                        p["system"], "warmup", {"session": 0, "segment": item["id"], "prompt": "warmup"}, 1)
            result["unique"] = await phase(p, client, items, False, p["requests"], out, "unique")
            result["repeat"] = await phase(p, client, items, True, p["requests"], out, "repeat")
    finally:
        bench.stop_server(process, log)
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k in ("unique", "repeat")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    asyncio.run(main_async(yaml.safe_load(parser.parse_args().config.read_text())["parameters"]))


if __name__ == "__main__":
    main()
