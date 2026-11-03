"""Caption delay from the confidence gate on real speech: snapshot each utterance's cumulative audio like the
pseudo-stream partials of the AmphionASR release path (system block, max_tokens 512, every 640 ms, first 8 s).

One gated server is enough: `amphion_confidence.rejected` says the gate replaced an answer that had text, so a
snapshot shows text without the gate when rejected or non-empty, and with the gate when non-empty. Writes one line
per utterance without transcripts. Resumes by id.

    python partial_caption.py --server URL --model NAME --set NAME=DATA_CONFIG[#SOURCE] --out rows.jsonl [--workers 12]
"""
import argparse, base64, json, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from amphion_eval.integrations.open_audio_llm.catalog import EvaluationDataset, read_config
from amphion_eval.integrations.open_audio_llm.native import encode_for_server, prepare_audio

STEP, LIMIT, MARK = 0.64, 8.0, "<asr_text>"


def items(name, spec):
    path, _, keep = spec.partition("#")
    config = read_config(path)
    if keep:
        config["evaluation"] = [s for s in config["evaluation"] if keep in f"{s['dataset_id']}@{s['version']}:{s['split']}"]
    dataset = EvaluationDataset(config, "evaluation")
    for source, indexes in zip(dataset.sources, dataset.source_ranges):
        key = f"{source['dataset_id']}@{source['version']}:{source['split']}"
        for index in indexes:
            record = dataset.records[index].record
            if (record.target or "").strip():
                yield {"id": f"{name}:{record.id}", "set": name, "source": key, "load": lambda i=index: (prepare_audio(dataset, i)[0], dataset.sampling_rate)}


def one(args, item):
    audio, rate = item.pop("load")()
    dur = len(audio) / rate
    marks, n = [], 1
    while n * STEP < min(dur, LIMIT + 1e-6):
        marks.append(round(n * STEP, 2)); n += 1
    cells = []
    for end in marks:
        wav = encode_for_server(audio[: int(end * rate)], rate)
        body = {"model": args.model, "temperature": 0, "max_tokens": 512, "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}}]}]}
        r = requests.post(f"{args.server}/v1/chat/completions", json=body, timeout=600)
        r.raise_for_status()
        j = r.json()
        content = j["choices"][0]["message"]["content"] or ""
        report = j["amphion_confidence"]
        has = bool(content.partition(MARK)[2].strip())
        cells.append([end, int(has), int(report["rejected"]), j["choices"][0]["finish_reason"]])
    return {**item, "dur": round(dur, 2), "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--set", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    name, spec = args.set.split("=", 1)
    done = {json.loads(l)["id"] for l in open(args.out)} if Path(args.out).exists() else set()
    todo = [i for i in items(name, spec) if i["id"] not in done]
    print(f"{name}: {len(todo)} to do, {len(done)} done", flush=True)
    lock, count = threading.Lock(), [0]
    with open(args.out, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(lambda i: one(args, i), todo):
            with lock:
                f.write(json.dumps(row) + "\n"); f.flush(); count[0] += 1
                if count[0] % 200 == 0:
                    print(count[0], flush=True)


if __name__ == "__main__":
    main()
