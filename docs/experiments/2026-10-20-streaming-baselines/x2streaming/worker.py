#!/usr/bin/env python3
"""AmphionEval worker for X2Streaming-ASR-4B: official ``infer.py`` streaming mode, one shard per GPU.

AmphionEval writes ``requests.jsonl`` (``id``/``task``/``audio``/``context``) and reads one JSON line
per request back. Inference is the unmodified official CLI (transformers backend, ``configs/infer.yaml``),
so the numbers are the official streaming decode; this file only shards requests and maps results to ids.
The official emit records are kept in each line (``emits``/``n_emits``) for later latency analysis.

Usage (run with the ``x2streaming`` conda env python):
  worker.py --input REQUESTS --output PREDICTIONS --gpus 6,7 --repo X2_REPO --model X2_MODEL
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated GPU indices, one shard each")
    parser.add_argument("--repo", type=Path, required=True, help="X2Streaming-ASR checkout")
    parser.add_argument("--model", type=Path, required=True, help="X2Streaming-ASR-4B-1009 weights")
    parser.add_argument("--language", default="zh", choices=["zh", "en", "cs"])
    return parser.parse_args()


def runtime_snapshot(args: argparse.Namespace) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    # `hf download --local-dir` keeps the resolved commit of every file in these metadata files.
    revisions = sorted({
        meta.read_text().splitlines()[0]
        for meta in (args.model / ".cache" / "huggingface" / "download").glob("*.metadata")
    })
    return {
        "backend": "transformers (official infer.py, streaming mode)",
        "config": (args.repo / "configs" / "infer.yaml").read_text(),
        "repo": str(args.repo),
        "repo_commit": commit,
        "model": str(args.model),
        "model_revisions": revisions,
        "language": args.language,
        "python": sys.version.split()[0],
        "torch": version("torch"),
        "transformers": version("transformers"),
    }


def main() -> int:
    args = parse_args()
    requests = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    work = args.output.parent / "x2-worker"
    work.mkdir(exist_ok=True)
    (work / "runtime.json").write_text(json.dumps(runtime_snapshot(args), ensure_ascii=False, indent=2))

    by_audio = {request["audio"]: request["id"] for request in requests}
    shards = [requests[i :: len(gpus)] for i in range(len(gpus))]
    procs = []
    started = time.perf_counter()
    for gpu, shard in zip(gpus, shards):
        if not shard:
            continue
        shard_in, shard_out = work / f"shard-gpu{gpu}.jsonl", work / f"shard-gpu{gpu}-out.jsonl"
        shard_in.write_text("".join(json.dumps({"audio": r["audio"]}) + "\n" for r in shard))
        command = [
            sys.executable, "infer.py",
            "--config", "configs/infer.yaml",
            "--model_path", str(args.model),
            "--language", args.language,
            "--input_file", str(shard_in),
            "--output_file", str(shard_out),
        ]
        log = (work / f"shard-gpu{gpu}.log").open("w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        procs.append((gpu, shard_out, subprocess.Popen(command, cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT)))
        print(f"gpu {gpu}: {len(shard)} requests -> {shard_out}", flush=True)

    codes = {gpu: proc.wait() for gpu, _, proc in procs}
    print(f"shards finished in {time.perf_counter() - started:.0f}s, exit codes {codes}", flush=True)

    answered = set()
    with args.output.open("a", encoding="utf-8") as output:
        for _, shard_out, _ in procs:
            if not shard_out.exists():
                continue
            for line in shard_out.read_text().splitlines():
                if not line.strip():
                    continue
                result = json.loads(line)
                request_id = by_audio[result["audio"]]
                answered.add(request_id)
                output.write(json.dumps({
                    "id": request_id,
                    "status": "success",
                    "text": result["text"],
                    "emits": result.get("emits"),
                    "n_emits": result.get("n_emits"),
                }, ensure_ascii=False) + "\n")
        for request in requests:
            if request["id"] not in answered:
                output.write(json.dumps({
                    "id": request["id"],
                    "status": "failed",
                    "error": "X2 infer.py returned no result; see x2-worker/shard-gpu*.log",
                }) + "\n")
    return max(codes.values(), default=0)


if __name__ == "__main__":
    raise SystemExit(main())
