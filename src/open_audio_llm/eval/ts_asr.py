"""Evaluate native TS-ASR and ordinary ASR, with a per-domain Chinese CER gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from open_audio_llm.data.qwen3_asr import native_language

from .qwen3_asr import normalize, prepare_model_for_eval


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    metrics = {}
    for source, items in groups.items():
        chinese = native_language(items[0]["language"]) == "Chinese"
        if items[0]["task"] == "speaker_attributed_asr":
            from .sot import summarize_sot

            metrics[source] = summarize_sot(items, chinese)
            continue
        errors = units = negatives = false_alarms = positives = misses = 0
        for row in items:
            ref, hyp = normalize(row["reference"]), normalize(row["prediction"])
            ref = list(ref.replace(" ", "")) if chinese else ref.split()
            hyp = list(hyp.replace(" ", "")) if chinese else hyp.split()
            if ref:
                errors += Levenshtein.distance(ref, hyp)
                units += len(ref)
                positives += 1
                misses += not hyp
            else:
                negatives += 1
                false_alarms += bool(hyp)
        metrics[source] = {
            "task": items[0]["task"], "language": items[0]["language"],
            "utterances": len(items), "metric": "CER" if chinese else "WER",
            "errors": errors, "reference_units": units,
            "error_rate": errors / units if units else None,
            "negative_utterances": negatives,
            "false_alarm_rate": false_alarms / negatives if negatives else None,
            "miss_rate": misses / positives if positives else None,
        }
    return metrics


def retention_gate(baseline, candidate, max_cer_increase=0.0):
    """Require identical evaluation inputs and check every Chinese plain-ASR domain."""
    if not math.isfinite(max_cer_increase) or max_cer_increase < 0:
        raise ValueError("max_cer_increase must be non-negative and finite")
    if baseline["protocol"] != candidate["protocol"]:
        raise ValueError("Baseline evaluation protocol or selected records differ")
    if baseline["metrics"].keys() != candidate["metrics"].keys():
        raise ValueError("Baseline and candidate evaluation sources differ")
    checks = {}
    for source, base in baseline["metrics"].items():
        if base["task"] != "asr" or base["metric"] != "CER":
            continue
        tuned = candidate["metrics"][source]
        before, after = base["error_rate"], tuned["error_rate"]
        if (before is None or after is None
                or not math.isfinite(before) or not math.isfinite(after)
                or base["reference_units"] != tuned["reference_units"]):
            raise ValueError(f"Invalid or incomparable Chinese CER: {source}")
        checks[source] = {
            "baseline_cer": before, "candidate_cer": after,
            "increase": after - before,
            "passed": after <= before + max_cer_increase,
        }
    if not checks:
        raise ValueError("Retention evaluation requires Chinese ordinary-ASR sources")
    return {"passed": all(c["passed"] for c in checks.values()),
            "max_cer_increase": max_cer_increase, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    audio_mode = parser.add_mutually_exclusive_group()
    audio_mode.add_argument("--audio_encoder_parallel", action="store_true")
    audio_mode.add_argument("--audio_encoder_batching", action="store_true")
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_group", choices=["validation", "evaluation"], default="validation")
    parser.add_argument("--samples_per_source", type=int, default=256,
                        help="Fixed random subset per source; 0 evaluates all eligible records")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", type=Path, help="Baseline summary.json from this command")
    parser.add_argument("--max_cer_increase", type=float, default=0.0,
                        help="Absolute CER allowance: 0.005 means 0.5 percentage points")
    args = parser.parse_args()
    if args.samples_per_source < 0 or args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("samples_per_source must be non-negative; batch_size must be positive")

    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel

    from open_audio_llm.data.catalog_dataset import (
        CatalogSwiftDataset,
        read_data_config,
    )

    torch.manual_seed(args.seed)
    config = read_data_config(args.data_config)
    config["validation"] = config[args.split_group]
    dataset = CatalogSwiftDataset(config, training=False, message_format="qwen3_asr")
    selected = []
    for source, indexes in zip(dataset.sources, dataset.source_ranges):
        key = f"{source['dataset_id']}@{source['version']}:{source['split']}"
        indexes = list(indexes)
        if args.samples_per_source:
            indexes = random.Random(f"{args.seed}:{key}").sample(
                indexes, min(args.samples_per_source, len(indexes)),
            )
        for index in indexes:
            record = dataset.records[index].record
            if record.task not in {"asr", "ts_asr", "speaker_attributed_asr"}:
                raise ValueError("Use the hotword evaluator for contextual ASR")
            selected.append((f"{key}/{record.task}/{record.language}", index))
    fingerprint = hashlib.sha256()
    for key, index in selected:
        fingerprint.update(json.dumps([key, dataset.records[index].record.to_dict()],
                                      sort_keys=True, ensure_ascii=False).encode())
    protocol = {
        "version": 1, "records_sha256": fingerprint.hexdigest(),
        "sampling_rate": dataset.sampling_rate, "max_new_tokens": args.max_new_tokens,
        "normalization": "NFKC-lower-punctuation-symbols-space; zh characters/en words",
        "split_group": args.split_group, "batch_size": args.batch_size,
        "seed": args.seed,
    }
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    if baseline is not None and baseline["protocol"] != protocol:
        raise ValueError("Baseline evaluation protocol or selected records differ")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(json.dumps(vars(args), default=str, indent=2))
    model = Qwen3ASRModel.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    if args.adapter:
        from peft import PeftModel

        model.model = PeftModel.from_pretrained(model.model, args.adapter).merge_and_unload()
    prepare_model_for_eval(model.model)
    (output / "settings.json").write_text(json.dumps(
        {**vars(args), "generation_use_cache": True, "generation_do_sample": False},
        default=str, indent=2,
    ))
    restore_audio = None
    if args.audio_encoder_batching:
        from open_audio_llm.integrations.ms_swift.audio_batching import enable_batched_audio

        restore_audio = enable_batched_audio(model.model)
    elif args.audio_encoder_parallel:
        from open_audio_llm.integrations.ms_swift.audio_batching import enable_parallel_audio

        restore_audio = enable_parallel_audio(model.model)
    rows = []
    try:
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for start in range(0, len(selected), args.batch_size):
                batch = selected[start:start + args.batch_size]
                samples = [dataset[index] for _, index in batch]
                audio = [sf.read(BytesIO(sample["audios"][0]), dtype="float32") for sample in samples]
                contexts = [sample["messages"][0]["content"] for sample in samples]
                with torch.inference_mode():
                    predictions = model.transcribe(audio=audio, context=contexts)
                if len(predictions) != len(batch):
                    raise RuntimeError("Inference returned an incomplete batch")
                for (source, index), prediction in zip(batch, predictions):
                    record = dataset.records[index].record
                    row = {"source": source, "id": record.id, "task": record.task,
                           "language": record.language, "reference": record.target,
                           "prediction": prediction.text, "predicted_language": prediction.language}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
                stream.flush()
                print(f"Evaluated {len(rows)}/{len(selected)}", flush=True)
    finally:
        if restore_audio is not None:
            restore_audio()
    summary = {"protocol": protocol, "metrics": summarize(rows)}
    if baseline is not None:
        summary["retention"] = retention_gate(baseline, summary, args.max_cer_increase)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if baseline is not None and not summary["retention"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
