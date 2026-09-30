"""Evaluate frozen enrollment views using vLLM and online W&B tracking."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .target_sot import summarize_target_sot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.7)
    parser.add_argument("--frontend-device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir must be new; existing evaluation results are preserved")
    import wandb

    from open_audio_llm.data.catalog_dataset import (
        CatalogSwiftDataset,
        read_data_config,
    )
    from open_audio_llm.integrations.vllm.target_sot import TargetSOTVLLM

    config = read_data_config(args.data_config)
    if not config.get("validation") or any(s.get("enrollment") is None for s in config["validation"]):
        parser.error("Conditional evaluation requires validation sources with enrollment enabled")
    dataset = CatalogSwiftDataset(config, training=False, message_format="qwen3_asr")
    for resolved in dataset.records:
        if "fixed_enrollment" not in resolved.record.metadata:
            parser.error(f"Evaluation needs frozen enrollment metadata: {resolved.record.id}")
    args.output_dir.mkdir(parents=True)
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    run = wandb.init(entity="1016097967-amphion", project="open-audio-llm", job_type="target-sot-eval",
                     mode="online", dir=str(args.output_dir), config={"model": args.model,
                     "data_config_sha256": fingerprint, "backend": "vllm"})
    rows, failures = [], []
    completed = False
    try:
        model = TargetSOTVLLM(args.model, max_model_len=args.max_model_len,
                              gpu_memory_utilization=args.gpu_memory_utilization,
                              frontend_device=args.frontend_device)
        if model.backend != "vllm":
            raise RuntimeError("Evaluation requires vLLM")
        with (args.output_dir / "predictions.jsonl").open("w") as sink:
            for index in range(len(dataset)):
                record = dataset.records[index].record
                fixed = record.metadata.get("fixed_enrollment")
                if fixed is None:
                    raise ValueError(f"Evaluation needs frozen enrollment metadata: {record.id}")
                try:
                    sample = dataset[index]
                    result = model.transcribe(sample["audios"][0],
                                              sample["chat_template_kwargs"]["enroll_wavs"],
                                              fixed["mode"], max_tokens=args.max_new_tokens)
                    row = dict(id=record.id, reference=sample["solution"], language=record.language,
                               count=len(fixed["enrollments"]), mode=fixed["mode"],
                               enrollment_seconds=[e["ref"]["duration"] for e in fixed["enrollments"]],
                               **result)
                    rows.append(row)
                except (ValueError, RuntimeError) as exc:
                    row = {"id": record.id, "status": "failed", "reason": str(exc)}
                    failures.append(row)
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                run.log({"completed_examples": len(rows), "failed_examples": len(failures)})
        grouped = defaultdict(list)
        for row in rows:
            durations = ",".join(f"{n:g}" for n in row["enrollment_seconds"])
            grouped[f'{row["language"]}/{row["mode"]}/k{row["count"]}/{durations}s'].append(row)
        metrics = {key: summarize_target_sot(items) for key, items in grouped.items()}
        summary = {"backend": "vllm", "total_examples": len(dataset), "completed_examples": len(rows),
                       "failed_examples": len(failures), "coverage": len(rows) / len(dataset),
                       "metric_scope": "completed_examples_only; failures listed separately",
                       "output_limit_examples": sum(r["finish_reason"] == "length" for r in rows),
                       "metrics": metrics, "failures": failures, "wandb_url": run.url}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        for key, values in metrics.items():
            logged = {f"eval/{key}/{k}": v for k, v in values.items() if isinstance(v, (int, float))}
            logged.update({f"eval/{key}/timestamps/{k}": v for k, v in values["timestamps"].items()
                           if isinstance(v, (int, float))})
            run.log(logged)
        run.log({"evaluation_coverage": summary["coverage"], "output_limit_examples": summary["output_limit_examples"]})
        run_id, entity, project = run.id, run.entity, run.project
        completed = not failures
    finally:
        run.finish(exit_code=0 if completed else 1)
    remote = wandb.Api().run(f"{entity}/{project}/{run_id}")
    if remote.summary.get("completed_examples") != len(rows):
        raise RuntimeError("W&B did not confirm uploaded evaluation metrics")
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "wandb_url": remote.url}))
    if failures:
        raise SystemExit("Evaluation has failures; inspect coverage and predictions.jsonl")


if __name__ == "__main__":
    main()
