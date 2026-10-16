"""Deployment-stack evaluation of the long-enrollment TS-ASR model on Catalog records, with a
consistency check between inference backends.

    python -m open_audio_llm.longenroll.evaluate --data_config data.yaml --split validation \
        --samples 400 --model_dir <export.py output> --output_dir <new dir> \
        --worker_python <serving env python> --backends vllm torch

This process never loads model weights:
  1. prepare: --samples records spread evenly over the split sorted by duration (so every length
     and task is covered), mixture / enrollment audio written as files, requests.jsonl for the
     worker and references.jsonl with the Catalog answers;
  2. one JSONL worker (infer.py) per backend, each its own process in --worker_python;
  3. score each backend with the training validation metric (CER for Chinese, WER otherwise,
     share of non-empty outputs for absent targets), then compare the first backend with each
     other one: identical token ids, identical text, edit rate between the two outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .features import TASK_NAMES, TS_SYSTEM, request_system, task_name
from .train import _edits, _units, score

REQUEST_TASK = {"ts_asr": "ts", "asr": "asr", "asr_hotwords": "hotword"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_config", required=True)
    p.add_argument("--split", choices=("validation", "evaluation"), default="validation")
    p.add_argument("--samples", type=int, required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--worker_python", default=sys.executable)
    p.add_argument("--backends", nargs="+", choices=("vllm", "torch"), default=["vllm"],
                   help="the first is scored and the others are compared with it")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--frontend_batch", type=int, default=16)
    p.add_argument("--feature_workers", type=int, default=4, help="worker threads for audio reading and features")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.4)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--workers", type=int, default=8, help="processes decoding Catalog audio")
    return p.parse_args(argv)


def _identity(item):
    return item


def prepare(args, out: Path) -> list[dict]:
    import copy

    import soundfile as sf
    import torch

    from ..data.catalog_dataset import CatalogSwiftDataset, read_data_config

    config = copy.deepcopy(read_data_config(args.data_config))
    config["validation"] = config[args.split]  # the dataset reads held-out records from "validation"
    dataset = CatalogSwiftDataset(config, training=False, message_format="qwen3_asr", waveforms=True)
    costs = dataset.records.durations
    ordered = sorted(range(len(dataset)), key=lambda i: (float(costs[i]), i))
    count = min(args.samples, len(ordered))
    picks = [ordered[int(k * len(ordered) / count)] for k in range(count)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, sampler=[(0, i, None) for i in picks],
                                         num_workers=args.workers, collate_fn=_identity)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True)

    def write(name, wave):
        wave = np.asarray(wave, dtype=np.float32)
        # 24-bit FLAC is lossless at the model's precision; a mixture above full scale stays float.
        path = audio_dir / (f"{name}.flac" if np.abs(wave).max(initial=0.0) <= 1.0 else f"{name}.wav")
        sf.write(path, wave, 16000, subtype="PCM_24" if path.suffix == ".flac" else "FLOAT")
        return str(path)

    requests, references = [], []
    for index, sample in zip(picks, loader):
        rid = f"{index:07d}"
        messages = sample["messages"]
        answer = messages[-1]["content"]
        task = REQUEST_TASK[sample["task"]]
        system = messages[0]["content"]
        hotwords = system.removeprefix("Hotwords: ") if task == "hotword" and system.startswith("Hotwords: ") else ""
        # The worker rebuilds the system turn from the task; it must be the one the record trains with.
        if request_system(task, hotwords) != (TS_SYSTEM if task == "ts" else system):
            raise ValueError(f"record {rid}: system turn {system!r} cannot be rebuilt from task {task}")
        mixture = sample["mix_wav"] if task == "ts" else sample["audios"][0]
        request = {"id": rid, "task": task, "mixture": write(f"{rid}.mix", mixture)}
        if task == "ts":
            request["enrollment"] = write(f"{rid}.enroll", sample["audios"][0])
        if hotwords:
            request["hotwords"] = hotwords
        requests.append(request)
        references.append({
            "id": rid, "dataset_id": sample.get("dataset_id"), "task": task_name(sample["task"], answer),
            "answer": answer, "language": answer.split("<asr_text>", 1)[0].removeprefix("language ").strip(),
            "mix_sec": len(mixture) / 16000,
            "enroll_sec": len(sample["audios"][0]) / 16000 if task == "ts" else 0.0,
        })
    for name, rows in (("requests.jsonl", requests), ("references.jsonl", references)):
        with open(out / name, "x") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return references


def run_worker(args, out: Path, backend: str) -> dict:
    command = [args.worker_python, "-m", "open_audio_llm.longenroll.infer", "--backend", backend,
               "--model_dir", args.model_dir, "--input", str(out / "requests.jsonl"),
               "--output", str(out / f"predictions-{backend}.jsonl"), "--summary", str(out / f"worker-{backend}.json"),
               "--max_new_tokens", str(args.max_new_tokens), "--frontend_batch", str(args.frontend_batch),
               "--feature_workers", str(args.feature_workers),
               "--gpu_memory_utilization", str(args.gpu_memory_utilization),
               "--max_model_len", str(args.max_model_len)]
    with open(out / f"worker-{backend}.log", "x") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
    return json.loads((out / f"worker-{backend}.json").read_text())


def read_jsonl(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in map(json.loads, path.read_text().splitlines())}


def score_backend(references: list[dict], predictions: dict[str, dict]) -> dict:
    sums = Counter()
    for ref in references:
        edits, units = score(ref["answer"], predictions[ref["id"]]["raw"], ref["language"])
        sums[f"err_{ref['task']}"] += edits
        sums[f"ref_{ref['task']}"] += units
    result = {}
    for task in TASK_NAMES:
        if sums[f"ref_{task}"]:
            name = "false_alarm" if task == "ts_absent" else "error_rate"
            result[f"{name}_{task}"] = sums[f"err_{task}"] / sums[f"ref_{task}"]
    return result


def compare(references: list[dict], first: dict[str, dict], other: dict[str, dict], path: Path) -> dict:
    same_tokens = same_text = edits = units = 0
    with open(path, "x") as handle:
        for ref in references:
            a, b = first[ref["id"]], other[ref["id"]]
            same_tokens += a["token_ids"] == b["token_ids"]
            same_text += a["text"] == b["text"]
            ua, ub = _units(a["text"], ref["language"]), _units(b["text"], ref["language"])
            edits += _edits(ua, ub)
            units += max(len(ua), 1)
            if a["token_ids"] != b["token_ids"]:
                handle.write(json.dumps({"id": ref["id"], "task": ref["task"], "answer": ref["answer"],
                                         a["backend"]: a["raw"], b["backend"]: b["raw"]}, ensure_ascii=False) + "\n")
    n = max(len(references), 1)
    return {"token_identical": same_tokens / n, "text_identical": same_text / n, "edit_rate": edits / units}


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(f"{out}/summary.json exists; evaluations never overwrite")
    started = time.perf_counter()
    references = prepare(args, out)
    summary = {"split": args.split, "samples": len(references), "prepare_s": time.perf_counter() - started,
               "tasks": dict(Counter(r["task"] for r in references)),
               "mixture_seconds": sum(r["mix_sec"] for r in references),
               "enrollment_seconds": sum(r["enroll_sec"] for r in references),
               "backends": {}, "consistency": {}}
    predictions = {}
    for backend in args.backends:
        worker = run_worker(args, out, backend)
        predictions[backend] = read_jsonl(out / f"predictions-{backend}.jsonl")
        summary["backends"][backend] = {**score_backend(references, predictions[backend]), "worker": worker}
    first = args.backends[0]
    for backend in args.backends[1:]:
        summary["consistency"][backend] = compare(references, predictions[first], predictions[backend],
                                                  out / f"mismatch-{first}-{backend}.jsonl")
    with open(out / "summary.json", "x") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
