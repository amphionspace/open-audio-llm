"""Mine GRPO prompts the current policy sometimes gets wrong.

Draws candidate ``(epoch, record, occurrence)`` tuples from chosen sources of a Catalog data config, renders
each exactly as GRPO training will (the same dataset code and seeds), samples the served model ``--samples``
times and scores the samples with the GRPO rewards. A candidate is kept when its presence or hotword reward
differs inside the group: those are the behaviours mining concentrates, while accuracy-only variation is already
plentiful in the replay draw. Plain-ASR sources only (no TS mixtures).

usage: python -m open_audio_llm.scripts.grpo_mining --data_config D --source ID@VERSION:SPLIT [--source ...]
           --candidates N --epoch E --server URL --model NAME --output kept.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import random
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import requests
import torch
from torch.utils.data import DataLoader, Dataset

from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config
from open_audio_llm.data.qwen3_asr import _wav_bytes
from open_audio_llm.integrations.ms_swift.grpo_prompts import source_key
from open_audio_llm.integrations.ms_swift.rewards import qwen3_asr


class Candidates(Dataset):
    def __init__(self, dataset, entries):
        self.dataset, self.entries = dataset, entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        entry = self.entries[i]
        sample = self.dataset[tuple(entry["index"])]
        if sample.get("mix_wav") is not None or len(sample["audios"]) != 1:
            raise ValueError(f"{entry['source']}: mining supports single-clip plain ASR only")
        audio = sample["audios"][0]
        if not isinstance(audio, bytes):
            audio = _wav_bytes(audio, self.dataset.sampling_rate)
        system = next((m["content"] for m in sample["messages"] if m["role"] == "system"), "")
        return {**entry, "system": system, "solution": sample["solution"],
                "audio": base64.b64encode(audio).decode("ascii")}


def score(item, outputs):
    messages = [{"role": "system", "content": item["system"]}]
    listed = qwen3_asr.prompt_hotwords(messages)
    rows = []
    for text in outputs:
        hotword = qwen3_asr.hotword(text, item["solution"], listed) if listed else None
        rows.append({
            "accuracy": qwen3_asr.accuracy(text, item["solution"]),
            "presence": qwen3_asr.presence(text, item["solution"]),
            "hotword": hotword,
        })
    varies = any(len({r[name] for r in rows}) > 1 for name in ("presence", "hotword"))
    return rows, varies


def request(session, args, item):
    response = session.post(args.server.rstrip("/") + "/v1/chat/completions", timeout=600, json={
        "model": args.model, "n": args.samples, "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "messages": [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": [{"type": "input_audio",
                                          "input_audio": {"data": item["audio"], "format": "wav"}}]},
        ],
    })
    response.raise_for_status()
    outputs = [c["message"]["content"] or "" for c in response.json()["choices"]]
    rows, kept = score(item, outputs)
    item = {k: v for k, v in item.items() if k != "audio"}
    return {**item, "outputs": outputs, "rewards": rows, "kept": kept}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--candidates", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset = CatalogSwiftDataset(read_data_config(args.data_config), grpo=True, message_format="qwen3_asr")
    ranges = {source_key(s): r for s, r in zip(dataset.sources, dataset.source_ranges)}
    rng = random.Random(f"{args.seed}:grpo-mining:{args.epoch}")
    entries = []
    for key in args.source:
        indexes = rng.sample(range(ranges[key].start, ranges[key].stop), min(args.candidates, len(ranges[key])))
        entries += [{"source": key, "index": [args.epoch, i, len(entries) + n],
                     "record_id": dataset.records[i].record.id} for n, i in enumerate(indexes)]
    # A run stopped early then still covers every source in proportion.
    rng.shuffle(entries)
    output = Path(args.output)
    # Forked workers share the record index; forkserver would pickle all 46M records into each worker.
    loader = DataLoader(Candidates(dataset, entries), batch_size=None, num_workers=args.workers)
    kept = total = 0
    with requests.Session() as session, ThreadPoolExecutor(args.concurrency) as pool, \
            output.open("w") as kept_file, output.with_suffix(".all.jsonl").open("w") as all_file:
        pending = set()

        def drain(futures):
            nonlocal kept, total
            for future in futures:
                row = future.result()
                total += 1
                kept += row["kept"]
                all_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                if row["kept"]:
                    kept_file.write(json.dumps({k: row[k] for k in ("source", "index", "record_id")}) + "\n")
                    kept_file.flush()
            print(f"scored {total}/{len(entries)} kept {kept}", flush=True)

        for item in loader:
            pending.add(pool.submit(request, session, args, item))
            if len(pending) >= args.concurrency * 2:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                drain(done)
        drain(wait(pending).done)
    print(json.dumps({"candidates": len(entries), "scored": total, "kept": kept}), flush=True)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
