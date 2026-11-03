"""Confidence-rejection probe: decode with per-token logprobs on the deployment vLLM server.

Every request mirrors the audiollm-go final decode for one segment: user-only prompt (the
runtime has no system block, passed as ``--template``), temperature 0, ``max_tokens`` from
the runtime's ``decodeBudget``. Audio for eval sets comes from AmphionEval's own loader, so
samples and bytes match the regular evaluation runs.

    python confprobe.py --server URL --model NAME --template T.jinja --out items.jsonl \
        [--clips ISSUE91_CLIPS_ROOT] [--set NAME=DATA_CONFIG[#SOURCE_SUBSTRING] ...]

Label: ``reject`` = should produce no text (issue-91 background clips, empty-reference
samples), ``keep`` = real speech that must still be transcribed. Resumes by sample ID.
"""
import argparse, base64, json, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests, soundfile as sf
from amphion_eval.integrations.open_audio_llm.catalog import EvaluationDataset, read_config
from amphion_eval.integrations.open_audio_llm.native import encode_for_server, prepare_audio

SR = 16000
TOP = 20  # vLLM's default --max-logprobs


def decode_budget(samples):
    # audiollm-go internal/asr/qwenclient/decode.go decodeBudget, without a prefix
    return min(512, max(128, 96 + (samples * 24 + SR - 1) // SR))


def items_from_clips(root):
    for line in open(Path(root) / "manifest.jsonl"):
        row = json.loads(line)
        label = "keep" if row["file"].startswith("control/") else "reject"
        yield {"id": "clip:" + row["file"], "set": "issue91", "source": "issue91/" + row["file"].split("/")[0],
               "label": label, "reference": "", "language": "zh", "task": "asr",
               "load": lambda f=row["file"]: sf.read(Path(root) / f, dtype="float32")}


def items_from_set(name, spec):
    path, _, keep = spec.partition("#")
    config = read_config(path)
    if keep:
        config["evaluation"] = [s for s in config["evaluation"] if keep in f"{s['dataset_id']}@{s['version']}:{s['split']}"]
    dataset = EvaluationDataset(config, "evaluation")
    for source, indexes in zip(dataset.sources, dataset.source_ranges):
        key = f"{source['dataset_id']}@{source['version']}:{source['split']}"
        for index in indexes:
            record = dataset.records[index].record
            yield {"id": f"{name}:{record.id}", "set": name, "source": f"{key}/{record.task}/{record.language}",
                   "label": "keep" if (record.target or "").strip() else "reject", "reference": record.target or "",
                   "language": record.language, "task": record.task,
                   "load": lambda i=index: (prepare_audio(dataset, i)[0], dataset.sampling_rate)}


def decode(args, template, item):
    audio, rate = item.pop("load")()
    wav = encode_for_server(audio, rate)
    samples = round(len(audio) * SR / rate)
    body = {"model": args.model, "temperature": 0, "max_tokens": decode_budget(samples),
            "logprobs": True, "top_logprobs": TOP, "chat_template": template,
            "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {
                "data": base64.b64encode(wav).decode(), "format": "wav"}}]}]}
    response = requests.post(f"{args.server}/v1/chat/completions", json=body, timeout=600)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    content = choice["message"]["content"] or ""
    tokens = choice["logprobs"]["content"]
    marker = next((i for i, t in enumerate(tokens) if t["token"] == "<asr_text>"), None)
    body_tokens = tokens[marker + 1:] if marker is not None else []
    stop = [t for t in body_tokens if t["token"] in ("<|im_end|>", "<|endoftext|>")]
    body_tokens = [t for t in body_tokens if t["token"] not in ("<|im_end|>", "<|endoftext|>")]
    first_top = tokens[marker + 1]["top_logprobs"] if marker is not None and marker + 1 < len(tokens) else []
    eos_first = next((t["logprob"] for t in first_top if t["token"] == "<|im_end|>"), None)
    return {**item, "dur": round(samples / SR, 3), "max_tokens": body["max_tokens"],
            "raw": content, "text": content.partition("<asr_text>")[2] if "<asr_text>" in content else content,
            "finish": choice["finish_reason"], "header_lp": [round(t["logprob"], 4) for t in tokens[:marker or 0]],
            "toks": [t["token"] for t in body_tokens], "lp": [round(t["logprob"], 4) for t in body_tokens],
            "stop_lp": round(stop[0]["logprob"], 4) if stop else None,
            "eos_first": None if eos_first is None else round(eos_first, 4),
            "first_top_min": round(min(t["logprob"] for t in first_top), 4) if first_top else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    template = open(args.template).read()
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    items = list(items_from_clips(args.clips)) if args.clips else []
    for spec in args.set:
        name, _, rest = spec.partition("=")
        items += list(items_from_set(name, rest))
    todo = [i for i in items if i["id"] not in done]
    print(f"{len(items)} items, {len(done)} done, {len(todo)} to decode", flush=True)
    lock, failed = threading.Lock(), 0
    with open(out, "a") as fh, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(decode, args, template, dict(i)): i["id"] for i in todo}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - a failed sample is reported and retried on resume
                failed += 1
                print(f"FAILED {futures[future]}: {type(exc).__name__}: {exc}", flush=True)
                continue
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            if n % 500 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    print(f"finished, failed={failed}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
