"""Evaluate native Qwen3-ASR and optional LoRA on held-out Catalog hotword audio.

Candidates contain annotated positives and sampled absent distractors. This is
an oracle-positive contextual ASR benchmark, not a hotword retrieval benchmark.
"""
from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from rapidfuzz.distance import Levenshtein


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join("".join(
        " " if unicodedata.category(c)[0] in {"P", "S"} else c for c in text
    ).split())


def prepare_model_for_eval(model):
    """Full SFT can persist cache=False; native generation delegates to thinker."""
    model.eval().requires_grad_(False)
    model.thinker.model.config.use_cache = True
    for module in (model, model.thinker):
        module.generation_config.use_cache = True
        module.generation_config.do_sample = False


def contains(text, word, chinese):
    text, word = normalize(text), normalize(word)
    if chinese:
        return word.replace(" ", "") in text.replace(" ", "")
    return f" {word} " in f" {text} "


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset_id"], row["condition"])].append(row)
    summary = {}
    for (dataset, condition), items in groups.items():
        errors = units = hits = positives = false_hits = negatives = 0
        for r in items:
            chinese = r["language"].lower() in {"zh", "chinese"}
            ref, hyp = normalize(r["reference"]), normalize(r["prediction"])
            ref = list(ref.replace(" ", "")) if chinese else ref.split()
            hyp = list(hyp.replace(" ", "")) if chinese else hyp.split()
            errors += Levenshtein.distance(ref, hyp)
            units += len(ref)
            positives += len(r["positives"])
            hits += sum(contains(r["prediction"], h, chinese) for h in r["positives"])
            negatives += len(r["negatives"])
            false_hits += sum(contains(r["prediction"], h, chinese) for h in r["negatives"])
        summary[f"{dataset}/{condition}"] = {
            "utterances": len(items), "metric": "CER" if chinese else "WER",
            "error_rate": errors / units if units else None,
            "errors": errors, "reference_units": units,
            "hotword_recall": hits / positives if positives else None,
            "hotword_hits": hits, "hotword_total": positives,
            "distractor_false_alarm_rate": false_hits / negatives if negatives else None,
            "distractor_hits": false_hits, "distractor_total": negatives,
        }
    return summary


def select_examples(dataset, samples, seed, distractors):
    selected = []
    for source, indices in zip(dataset.sources, dataset.source_ranges):
        rng = random.Random(f"{seed}:{source['dataset_id']}")
        pool = sorted({h for i in indices for h in dataset.records[i].record.hotwords if normalize(h)})
        positive, other = [], []
        for i in indices:
            record = dataset.records[i].record
            chinese = record.language.lower() in {"zh", "chinese"}
            positives = sorted({h for h in record.hotwords if normalize(h) and contains(record.target, h, chinese)})
            (positive if positives else other).append((i, positives))
        # Stratify to measure hotwords as well as context-induced insertions.
        n_positive = min(samples // 2, len(positive))
        n_other = min(samples - n_positive, len(other))
        if n_positive + n_other != samples:
            raise ValueError(f"Insufficient test examples for {source['dataset_id']}")
        chosen = rng.sample(positive, n_positive) + rng.sample(other, n_other)
        rng.shuffle(chosen)
        for i, positives in chosen:
            record = dataset.records[i].record
            chinese = record.language.lower() in {"zh", "chinese"}
            absent = [h for h in pool if not contains(record.target, h, chinese)]
            negatives = rng.sample(absent, min(distractors, len(absent)))
            candidates = positives + negatives
            rng.shuffle(candidates)
            selected.append((i, positives, negatives, candidates))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples_per_source", type=int, default=256)
    parser.add_argument("--distractors", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from qwen_asr import Qwen3ASRModel
    from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config

    torch.manual_seed(args.seed)
    config = read_data_config(args.data_config)
    config["validation"] = config["evaluation"]
    dataset = CatalogSwiftDataset(config, training=False, message_format="qwen3_asr")
    selected = select_examples(dataset, args.samples_per_source, args.seed, args.distractors)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "settings.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Selected {len(selected)} held-out utterances", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", max_inference_batch_size=args.batch_size,
        max_new_tokens=256,
    )
    if args.adapter:
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(model.model, args.adapter).merge_and_unload()
    prepare_model_for_eval(model.model)
    (output / "settings.json").write_text(json.dumps(
        {**vars(args), "generation_use_cache": True, "generation_do_sample": False},
        indent=2,
    ))
    rows = []
    with (output / "predictions.jsonl").open("w") as result_file:
        for offset in range(0, len(selected), args.batch_size):
            batch = selected[offset:offset + args.batch_size]
            audio = [
                sf.read(BytesIO(dataset[i]["audios"][0]), dtype="float32")
                for i, *_ in batch
            ]
            for condition in ("no_hotwords", "hotwords"):
                contexts = ["Hotwords: " + ",".join(c) if condition == "hotwords" else "" for _, _, _, c in batch]
                with torch.inference_mode():
                    predictions = model.transcribe(audio=audio, context=contexts)
                if len(predictions) != len(batch):
                    raise RuntimeError("Inference returned an incomplete batch")
                for (i, positives, negatives, candidates), prediction in zip(batch, predictions):
                    record = dataset.records[i].record
                    row = {
                        "id": record.id, "dataset_id": dataset.records[i].dataset_id,
                        "language": record.language, "condition": condition,
                        "reference": record.target, "prediction": prediction.text,
                        "predicted_language": prediction.language,
                        "positives": positives, "negatives": negatives,
                        "candidates": candidates if condition == "hotwords" else [],
                    }
                    rows.append(row)
                    result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                result_file.flush()
            print(f"Evaluated {min(offset + args.batch_size, len(selected))}/{len(selected)}", flush=True)
    summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
