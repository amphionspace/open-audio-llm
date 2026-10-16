"""JSONL inference worker for the long-enrollment TS-ASR model (AmphionASR v5).

    python -m open_audio_llm.longenroll.infer --model_dir <export.py output> \
        --input requests.jsonl --output predictions.jsonl --summary worker.json [--backend vllm]

One request per line:
    {"id": "...", "task": "ts" | "asr" | "hotword", "mixture": AUDIO,
     "enrollment": AUDIO (ts only), "hotwords": "a,b,c" (hotword only, as in the Catalog prompts)}
AUDIO is a path readable by soundfile, or {"path": ..., "start": s, "duration": s}; it is
downmixed and resampled to 16 kHz. Each output line:
    {"id", "backend", "language", "text", "raw", "token_ids", "finish_reason"}
The summary holds the backend, versions, decoding parameters and timings.

Backends (greedy decoding, same frontend code and request grouping in both):
    vllm   PyTorch frontend + vLLM decoding: the deployment stack.
    torch  LongEnrollTSASR built from the same directory, decoding with Transformers generate
           one request at a time: the training model's code, the reference for consistency.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .frontend import load_waveform, parse_answer


class LongEnrollTorch:
    backend = "torch"

    def __init__(self, model_dir, *, device="cuda", frontend_batch=16, feature_workers=4):
        from .backbone import PromptBuilder, load_qwen3_asr
        from .config import Config
        from .ecapa import ECAPATDNN
        from .features import LongEnrollFeatures
        from .frontend import cast_for_inference, read_extra
        from .model import LongEnrollTSASR

        asr, processor = load_qwen3_asr(str(model_dir), dtype=torch.bfloat16, device="cpu")
        extra_config, state = read_extra(model_dir)
        cfg = Config({"model": extra_config["model"], "train": {"gradient_checkpointing": False}})
        model = LongEnrollTSASR(asr, ECAPATDNN(), cfg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith(("asr.", "thinker.", "tower.")) for k in missing):
            raise RuntimeError(f"extra weights do not match the model: {unexpected[:4]} {missing[:4]}")
        self.model = cast_for_inference(model, device)
        self.features = LongEnrollFeatures(processor)
        self.tokenizer = processor.tokenizer
        self.pad_id = PromptBuilder(processor.tokenizer).pad_id
        self.frontend_batch = int(frontend_batch)
        self.feature_workers = int(feature_workers)
        from importlib.metadata import version

        self.versions = {"transformers": version("transformers"), "qwen-asr": version("qwen-asr"),
                         "torch": torch.__version__}

    def transcribe(self, requests, *, max_new_tokens=256):
        from .frontend import embed_requests

        items, embeds, feature_s, frontend_s = embed_requests(self.model, self.features, requests,
                                                              self.frontend_batch, self.feature_workers)
        started = time.perf_counter()
        results = []
        device = self.model.device
        for item, embed in zip(items, embeds):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                ids = self.model.generate_from_audio(embed, [item["input_ids"]], max_new_tokens, self.pad_id)[0]
            results.append({"raw": self.tokenizer.decode(ids, skip_special_tokens=False), "token_ids": ids,
                            "finish_reason": "length" if len(ids) >= max_new_tokens else "stop"})
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = {"feature_s": feature_s, "frontend_s": frontend_s, "llm_s": time.perf_counter() - started}
        return results, items, timings


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--backend", choices=("vllm", "torch"), default="vllm")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--frontend_batch", type=int, default=16, help="requests per frontend forward")
    p.add_argument("--feature_workers", type=int, default=4,
                   help="threads reading audio and building mel / fbank (8+ contended on a shared 128-core host)")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.4)
    p.add_argument("--max_num_seqs", type=int, default=256)
    p.add_argument("--warmup", type=int, default=8, help="requests run once before timing (not written)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for path in (args.output, args.summary):
        if Path(path).exists():
            raise FileExistsError(f"{path} exists; the worker never overwrites results")
    lines = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    loaded = time.perf_counter()

    def read(r):
        return {"task": r["task"], "mixture": load_waveform(r["mixture"]),
                "enrollment": load_waveform(r["enrollment"]) if r.get("enrollment") else None,
                "hotwords": r.get("hotwords", "")}

    with ThreadPoolExecutor(max(1, args.feature_workers)) as pool:
        requests = list(pool.map(read, lines))
    read_s = time.perf_counter() - loaded

    started = time.perf_counter()
    if args.backend == "vllm":
        from ..integrations.vllm.longenroll import LongEnrollVLLM

        runner = LongEnrollVLLM(args.model_dir, max_model_len=args.max_model_len,
                                gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs,
                                frontend_batch=args.frontend_batch, feature_workers=args.feature_workers)
        decoding = {"temperature": 0.0, "max_tokens": args.max_new_tokens, "max_model_len": args.max_model_len,
                    "gpu_memory_utilization": args.gpu_memory_utilization, "max_num_seqs": args.max_num_seqs}
    else:
        runner = LongEnrollTorch(args.model_dir, frontend_batch=args.frontend_batch,
                                 feature_workers=args.feature_workers)
        decoding = {"greedy": True, "max_new_tokens": args.max_new_tokens}
    load_s = time.perf_counter() - started
    if args.warmup:
        runner.transcribe(requests[:args.warmup], max_new_tokens=args.max_new_tokens)
    started = time.perf_counter()
    results, items, timings = runner.transcribe(requests, max_new_tokens=args.max_new_tokens)
    total_s = time.perf_counter() - started

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x") as handle:
        for line, result in zip(lines, results):
            language, text = parse_answer(result["raw"])
            handle.write(json.dumps({"id": line["id"], "backend": runner.backend, "language": language,
                                     "text": text, **result}, ensure_ascii=False) + "\n")
    mix_sec = sum(item["mix_sec"] for item in items)
    tokens = sum(len(r["token_ids"]) for r in results)
    summary = {
        "backend": runner.backend, "versions": runner.versions, "decoding": decoding,
        "model_dir": str(Path(args.model_dir).resolve()), "frontend_batch": args.frontend_batch,
        "feature_workers": args.feature_workers,
        "requests": len(results), "mixture_seconds": mix_sec,
        "enrollment_seconds": sum(item["enroll_sec"] for item in items), "output_tokens": tokens,
        "read_audio_s": read_s, "load_s": load_s, "total_s": total_s, **timings,
        "rtf": total_s / max(mix_sec, 1e-9), "requests_per_s": len(results) / max(total_s, 1e-9),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }
    with open(args.summary, "x") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
