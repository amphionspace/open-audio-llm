"""Deployment-stack evaluation of the long-enrollment TS-ASR model on Catalog records, with a
consistency check between inference backends.

    python -m open_audio_llm.longenroll.evaluate --data_config data.yaml --split validation \
        --samples 400 --model_dir <export.py output> --output_dir <new dir> \
        --worker_python <serving env python> --backends vllm torch

This process never loads model weights:
  1. prepare: --samples records spread evenly over the split sorted by duration (so every length
     and task is covered), mixture / enrollment audio written as files, requests.jsonl for the
     worker and references.jsonl with the Catalog answers;
     (--prepared reuses them from an earlier output directory instead);
  2. JSONL workers (infer.py) per backend, each its own process in --worker_python: one per
     visible GPU (CUDA_VISIBLE_DEVICES), each on an interleaved share of the requests;
  3. score each backend with the training validation metric (CER for Chinese, WER otherwise,
     share of non-empty outputs for absent targets) plus the miss rate (present target, empty
     output), over the whole split and per source (its ``label``, else dataset@version:split),
     then compare the first backend with each other one: identical token ids, identical text,
     edit rate between the two outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
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
    p.add_argument("--prepared", help="output directory of an earlier run on the same data and split whose "
                   "requests, references and audio are reused (--samples is then ignored)")
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
    owner = np.empty(len(dataset), dtype=np.int64)
    for number, span in enumerate(dataset.source_ranges):
        owner[span.start:span.stop] = number
    labels = [s.get("label") or f"{s['dataset_id']}@{s['version']}:{s['split']}" for s in dataset.sources]
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
        record = getattr(dataset.records[index], "record", None)
        references.append({
            "id": rid, "dataset_id": sample.get("dataset_id"), "source": labels[owner[index]],
            "record_id": getattr(record, "id", None), "task": task_name(sample["task"], answer),
            "answer": answer, "language": answer.split("<asr_text>", 1)[0].removeprefix("language ").strip(),
            "mix_sec": len(mixture) / 16000,
            "enroll_sec": len(sample["audios"][0]) / 16000 if task == "ts" else 0.0,
        })
    for name, rows in (("requests.jsonl", requests), ("references.jsonl", references)):
        with open(out / name, "x") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return references


def _worker_command(args, backend: str, requests: Path, predictions: Path, summary: Path) -> list[str]:
    return [args.worker_python, "-m", "open_audio_llm.longenroll.infer", "--backend", backend,
            "--model_dir", args.model_dir, "--input", str(requests), "--output", str(predictions),
            "--summary", str(summary), "--max_new_tokens", str(args.max_new_tokens),
            "--frontend_batch", str(args.frontend_batch), "--feature_workers", str(args.feature_workers),
            "--gpu_memory_utilization", str(args.gpu_memory_utilization), "--max_model_len", str(args.max_model_len)]


def run_worker(args, out: Path, backend: str) -> dict:
    devices = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if len(devices) <= 1:
        command = _worker_command(args, backend, out / "requests.jsonl", out / f"predictions-{backend}.jsonl",
                                  out / f"worker-{backend}.json")
        with open(out / f"worker-{backend}.log", "x") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
        return json.loads((out / f"worker-{backend}.json").read_text())
    # Requests are sorted by duration; an interleaved share gives every GPU every length.
    lines = (out / "requests.jsonl").read_text().splitlines()
    procs = []
    for shard, device in enumerate(devices):
        stem = f"{backend}-{shard}"
        (out / f"requests-{stem}.jsonl").write_text("".join(f"{line}\n" for line in lines[shard::len(devices)]))
        command = _worker_command(args, backend, out / f"requests-{stem}.jsonl", out / f"predictions-{stem}.jsonl",
                                  out / f"worker-{stem}.json")
        log = open(out / f"worker-{stem}.log", "x")
        procs.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "CUDA_VISIBLE_DEVICES": device}), log))
    # One failed shard fails the run: stop the others instead of letting them finish for nothing.
    while any(proc.poll() is None for proc, _ in procs):
        if any(proc.poll() not in (None, 0) for proc, _ in procs):
            for proc, _ in procs:
                if proc.poll() is None:
                    proc.terminate()
            break
        time.sleep(5)
    codes = [proc.wait() for proc, _ in procs]
    for _, log in procs:
        log.close()
    if any(codes):
        raise RuntimeError(f"{backend} workers exited with {codes}; see worker-{backend}-<shard>.log")
    with open(out / f"predictions-{backend}.jsonl", "x") as merged:
        for shard in range(len(devices)):
            merged.write((out / f"predictions-{backend}-{shard}.jsonl").read_text())
    return {"devices": devices, "shards": [json.loads((out / f"worker-{backend}-{shard}.json").read_text())
                                           for shard in range(len(devices))]}


def read_jsonl(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in map(json.loads, path.read_text().splitlines())}


def _rates(sums: Counter) -> dict:
    result = {}
    for task in TASK_NAMES:
        if sums[f"ref_{task}"]:
            name = "false_alarm" if task == "ts_absent" else "error_rate"
            result[f"{name}_{task}"] = sums[f"err_{task}"] / sums[f"ref_{task}"]
    if sums["count_ts"]:
        result["miss_rate_ts"] = sums["miss_ts"] / sums["count_ts"]
    return result


def score_backend(references: list[dict], predictions: dict[str, dict]) -> dict:
    total, by_source = Counter(), defaultdict(Counter)
    for ref in references:
        raw = predictions[ref["id"]]["raw"]
        edits, units = score(ref["answer"], raw, ref["language"])
        for sums in (total, by_source[ref["source"]]):
            sums["samples"] += 1
            sums[f"err_{ref['task']}"] += edits
            sums[f"ref_{ref['task']}"] += units
            if ref["task"] == "ts":
                sums["count_ts"] += 1
                sums["miss_ts"] += not raw.split("<asr_text>", 1)[-1].strip()
    return {**_rates(total),
            "by_source": {name: {"samples": sums["samples"], **_rates(sums)} for name, sums in by_source.items()}}


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
    if args.prepared:
        # Requests name their audio by absolute path inside the prepared directory.
        for name in ("requests.jsonl", "references.jsonl"):
            shutil.copyfile(Path(args.prepared) / name, out / name)
        references = list(read_jsonl(out / "references.jsonl").values())
    else:
        references = prepare(args, out)
    summary = {"split": args.split, "samples": len(references), "prepared_from": args.prepared,
               "prepare_s": time.perf_counter() - started,
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
