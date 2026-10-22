"""Greedy held-out evaluation of a checkpoint through the TS-ASR vLLM service.

The service must be the repository TS-ASR backend (``examples/configs/serve/
tsasr.yaml``: SEP on, global audio attention, prefix cache off), which accepts
``mm_processor_kwargs.enroll_seconds`` so each sample keeps its own enrollment
length, as in training.

    python -m open_audio_llm.rl.tsasr_grpo.evaluate run --samples DIR/samples.jsonl \
        --url http://127.0.0.1:8010 --model tsasr --output OUT.jsonl
    python -m open_audio_llm.rl.tsasr_grpo.evaluate compare base OUT0.jsonl grpo OUT1.jsonl
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from .model import DEFAULT_SYSTEM, MEL_HOP, SAMPLE_RATE
from .text import errors, parse_output, units
from .train import category, read_samples


def _request_audio(sample: dict) -> tuple[str, float]:
    enroll, _ = sf.read(sample["enroll_wav"], dtype="float32")
    mix, _ = sf.read(sample["mix_wav"], dtype="float32")
    # The service splits at enroll_seconds, which must fall on a Mel hop.
    enroll = np.pad(enroll, (0, -len(enroll) % MEL_HOP))
    buf = io.BytesIO()
    sf.write(buf, np.concatenate([enroll, mix]), SAMPLE_RATE, format="WAV", subtype="FLOAT")
    return base64.b64encode(buf.getvalue()).decode("ascii"), len(enroll) / SAMPLE_RATE


def run(args) -> None:
    samples = read_samples(args.samples)
    done = set()
    if args.output.exists():
        done = {json.loads(line)["id"] for line in args.output.open(encoding="utf-8")}
    todo = [s for s in samples if s["id"] not in done]

    def one(s):
        data, seconds = _request_audio(s)
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": args.system},
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}]},
            ],
            "temperature": 0.0, "max_tokens": args.max_tokens,
            "mm_processor_kwargs": {"enroll_seconds": seconds},
        }
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
                                         data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            text = parse_output(json.loads(response.read())["choices"][0]["message"]["content"])
        dist, n_ref = errors(s["target"], text, s["language"])
        return {k: s[k] for k in ("id", "label", "language", "target", "present")} | \
            {"hypothesis": text, "errors": dist, "reference_units": n_ref}

    failed = 0
    with ThreadPoolExecutor(args.concurrency) as pool, args.output.open("a", encoding="utf-8") as out:
        futures = {pool.submit(one, s): s["id"] for s in todo}
        for future in as_completed(futures):
            try:
                out.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                out.flush()
            except Exception as exc:  # noqa: BLE001 - reported per sample, rerun resumes
                failed += 1
                print(f"failed {futures[future]}: {type(exc).__name__}: {exc}", flush=True)
    print(f"{len(todo) - failed}/{len(todo)} scored, {failed} failed; rerun to retry", flush=True)


def summarize(path: Path) -> dict:
    groups = defaultdict(lambda: defaultdict(float))
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        for key in (f"{category(r)} ({r['language']})", "all/present" if r["present"] else "all/absent"):
            g = groups[key]
            g["n"] += 1
            if r["present"]:
                g["errors"] += r["errors"]
                g["units"] += r["reference_units"]
                g["miss"] += not units(r["hypothesis"], r["language"])
            else:
                g["false_alarm"] += r["errors"] > 0
    return groups


def compare(args) -> None:
    names, paths = args.runs[0::2], [Path(p) for p in args.runs[1::2]]
    runs = [summarize(p) for p in paths]
    print("group | n | " + " | ".join(names))
    for key in sorted(runs[0], key=lambda k: (k.startswith("all/"), k)):
        cells = []
        for r in runs:
            g = r[key]
            if g["units"]:
                cells.append(f"CER {100 * g['errors'] / g['units']:.1f} miss {100 * g['miss'] / g['n']:.1f}")
            else:
                cells.append(f"FA {100 * g['false_alarm'] / g['n']:.1f}")
        print(f"{key} | {int(runs[0][key]['n'])} | " + " | ".join(cells))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--samples", type=Path, required=True)
    r.add_argument("--url", required=True)
    r.add_argument("--model", required=True, help="served model name")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--system", default=DEFAULT_SYSTEM)
    r.add_argument("--max_tokens", type=int, default=256)
    r.add_argument("--concurrency", type=int, default=32)
    r.add_argument("--timeout", type=float, default=900)
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+", help="name path [name path ...]")
    args = p.parse_args(argv)
    if args.command == "run":
        run(args)
    else:
        if len(args.runs) % 2:
            p.error("compare takes name/path pairs")
        compare(args)


if __name__ == "__main__":
    main()
