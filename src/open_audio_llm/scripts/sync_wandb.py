"""Mirror saved training metrics and completed ASR evaluations to W&B on CPU."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path


TRAIN_KEYS = (
    "loss", "grad_norm", "learning_rate", "ce_asr", "ce_sot", "replay_kl",
    "memory(GiB)", "train_speed(s/it)",
)
CONFIG_KEYS = (
    "max_steps", "learning_rate", "vit_lr", "aligner_lr", "warmup_steps",
    "lr_scheduler_type", "gradient_accumulation_steps", "per_device_train_batch_size",
    "optim", "weight_decay", "max_grad_norm", "bf16", "torch_dtype",
    "freeze_vit", "freeze_aligner", "freeze_llm", "audio_encoder_batching",
    "save_only_model", "seed", "max_length", "model_type", "tuner_type",
)
SOURCE_KEYS = ("dataset_id", "version", "split", "weight", "require_clean_pass",
               "sot_timestamps", "sot_alignment_sha256")


def read_jsonl(path):
    if not path.exists():
        return []
    # The training writer may still be appending the last record.
    lines = path.read_bytes().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b"\n")]


def numeric(values):
    return {key: value for key, value in values.items()
            if isinstance(value, (int, float)) and math.isfinite(value)}


def aggregate_metrics(items):
    units = sum(m["reference_units"] for m in items)
    utterances = sum(m["utterances"] for m in items)
    result = {"utterances": utterances, "reference_units": units}
    if units:
        result[items[0]["metric"]] = 100 * sum(m["errors"] for m in items) / units
    for key in ("speaker_count_accuracy", "format_valid_rate", "empty_output_rate"):
        if all(key in m for m in items):
            result[key] = 100 * sum(m[key] * m["utterances"] for m in items) / utterances
    if all("speaker_attribution" in m for m in items):
        methods = {m["speaker_attribution"]["method"] for m in items}
        if len(methods) != 1:
            raise ValueError("Cannot combine different speaker-attribution methods")
        correct = sum(m["speaker_attribution"]["correct_units"] for m in items)
        scored = sum(m["speaker_attribution"]["scored_units"] for m in items)
        result["attribution_correct_units"] = correct
        result["attribution_scored_units"] = scored
        if scored:
            result["attribution_accuracy"] = 100 * correct / scored
        if units:
            result["attribution_coverage"] = 100 * scored / units
    if all("timestamp_reference_coverage" in m for m in items):
        available = sum(m["timestamp_reference_coverage"]["available_utterances"] for m in items)
        total = sum(m["timestamp_reference_coverage"]["total_utterances"] for m in items)
        result["timestamp_reference_records"] = available
        result["timestamp_total_records"] = total
        if total:
            result["timestamp_reference_coverage"] = 100 * available / total
    if all("timestamps" in m for m in items):
        protocols = {(m["timestamps"]["method"], m["timestamps"]["collar_seconds"]) for m in items}
        if len(protocols) != 1:
            raise ValueError("Cannot aggregate different timestamp scoring protocols")
        timing = {key: sum(m["timestamps"][key] for m in items) for key in (
            "reference_segments", "predicted_segments", "matched_segments",
            "within_collar_segments", "boundary_absolute_error_seconds")}
        refs, hyps = timing["reference_segments"], timing["predicted_segments"]
        matched, correct = timing["matched_segments"], timing["within_collar_segments"]
        result["timestamp_precision"] = 100 * correct / hyps if hyps else 0.0
        result["timestamp_recall"] = 100 * correct / refs if refs else 0.0
        result["timestamp_f1"] = 200 * correct / (refs + hyps) if refs + hyps else 0.0
        if refs:
            result["timestamp_matched_reference_fraction"] = 100 * matched / refs
        if matched:
            result["timestamp_boundary_mae_seconds"] = timing["boundary_absolute_error_seconds"] / (2 * matched)
    return result


def evaluation_metrics(summary, task):
    groups = defaultdict(list)
    for source, metric in summary["metrics"].items():
        if task == "sot":
            language = metric["language"]
            groups[language].append(metric)
            condition = re.search(r"_(\d+)spk_(none|low|medium|high|dense)/", source)
            if condition:
                count, overlap = condition.groups()
                groups[f"{language}/{count}spk/{overlap}"].append(metric)
        else:
            dataset = source.split("@", 1)[0]
            groups[f"{dataset}/{metric['metric']}"].append(metric)
    values = {}
    for group, items in groups.items():
        for key, value in aggregate_metrics(items).items():
            name = f"eval/{task}/{group}"
            if task == "sot" or key != items[0]["metric"]:
                name += f"/{key}"
            values[name] = value
    if "retention" in summary:
        values["eval/retention/passed"] = int(summary["retention"]["passed"])
        for source, check in summary["retention"]["checks"].items():
            dataset = source.split("@", 1)[0]
            language = source.rsplit("/", 1)[-1]
            values[f"eval/retention/{dataset}/{language}/CER_increase_pp"] = 100 * check["increase"]
    return values


def collect_events(root):
    events = {}
    for row in read_jsonl(root / "training/logging.jsonl"):
        if "global_step/max_steps" not in row:
            continue
        step = int(row["global_step/max_steps"].split("/")[0])
        if "loss" in row:
            kind = "train"
            values = {f"train/{key}": row[key] for key in TRAIN_KEYS if key in row}
        elif "eval_loss" in row:
            kind = "validation"
            values = {f"validation/{key.removeprefix('eval_')}": value
                      for key, value in row.items() if key.startswith("eval_")}
        else:
            continue
        events[f"{kind}:{step}"] = {"global_step": step, **numeric(values)}
    for row in read_jsonl(root / "training/performance-rank0.jsonl"):
        step = row["step"]
        values = {f"perf/rank0/{key}": value for key, value in row.items()
                  if key not in {"step", "timestamp", "rank", "loss", "learning_rate"}}
        events[f"performance:{step}"] = {"global_step": step, **numeric(values)}
    for folder in (root / "training/retention-evaluations").glob("checkpoint-*"):
        status = folder / "runner-exit.json"
        if not status.exists():
            continue
        try:
            finished = json.loads(status.read_text())
        except json.JSONDecodeError:
            # The completion marker is a small file written by the evaluator.
            continue
        if finished["returncode"] != 0:
            continue
        step = int(folder.name.removeprefix("checkpoint-"))
        values = {"global_step": step}
        if (folder / "metrics.json").is_file():
            values.update(numeric(json.loads((folder / "metrics.json").read_text())))
            events[f"evaluation:{step}"] = values
            continue
        for task in ("asr", "sot"):
            summary = json.loads((folder / task / "summary.json").read_text())
            values.update(evaluation_metrics(summary, task))
        events[f"evaluation:{step}"] = values
    return events


def run_config(root):
    import yaml

    args = json.loads((root / "training/args.json").read_text())
    config = {key: args[key] for key in CONFIG_KEYS if key in args}
    recipe_file = root / "train-data.yaml"
    recipe = yaml.safe_load(recipe_file.read_text())
    config["data_recipe_sha256"] = hashlib.sha256(recipe_file.read_bytes()).hexdigest()
    config["data_sources"] = [{key: source[key] for key in SOURCE_KEYS if key in source}
                              for source in recipe["train"]]
    config["objective"] = recipe["objective"]
    config["replay"] = recipe["replay"]
    config["batching"] = recipe["batching"]
    config["metric_unit"] = "percent; retention deltas in percentage points"
    config["speaker_attribution_method"] = "unique_reference_units_v1"
    config["logging_mode"] = "external CPU reader of saved metrics"
    audit = json.loads((root / "training/audit-startup-rank0.json").read_text())
    config["world_size"] = audit["world_size"]
    config["resume_step"] = audit["global_step"]
    if "optimizer_steps" in audit:
        config["optimizer_reinitialized"] = audit["optimizer_steps"] == []
    elif audit["global_step"] == 0:
        config["optimizer_reinitialized"] = True
    recovery = root / "recovery.json"
    origin = root
    if recovery.exists():
        phase = json.loads(recovery.read_text())
        origin = Path(phase["source_run"])
        config["parent_run"] = origin.name
        config["parent_checkpoint"] = Path(phase["source_checkpoint"]).name
    # A recovery can carry its own frozen snapshot (including run-specific
    # fixes). Parent provenance is only a fallback when no local record exists.
    provenance = root / "source-provenance.json"
    if not provenance.exists():
        provenance = origin / "source-provenance.json"
    if provenance.exists():
        metadata = json.loads(provenance.read_text())
        # A frozen file snapshot need not correspond to a Git commit. Never
        # substitute the sync checkout's HEAD or a parent run's base revision.
        config["source_git_commit"] = metadata.get("git_commit")
        config["source_base_git_commit"] = metadata.get("base_git_commit")
        config["source_origin"] = metadata.get("source_origin", str(provenance.parent / "source"))
        config["source_run"] = metadata.get("source_run")
        config["source_file_hashes"] = {
            **metadata.get("files_sha256", {}),
            **metadata.get("run_files_sha256", {}),
            **metadata.get("files", {}),
        }
        config["source_snapshot_sha256"] = hashlib.sha256(provenance.read_bytes()).hexdigest()
    return config


def upload_events(run, events, seen):
    pending = sorted(events.keys() - seen, key=lambda key: (events[key]["global_step"], key))
    for key in pending:
        # W&B's internal step stays monotonic; late evaluations use global_step as x.
        run.log({"sync_event": key, **events[key]})
        seen.add(key)
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="open-audio-llm")
    parser.add_argument("--run-id")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=30)
    options = parser.parse_args()
    if options.interval <= 0:
        parser.error("interval must be positive")
    root = options.run_dir.resolve()
    destination = root / "wandb-sync"
    destination.mkdir(exist_ok=True)
    lock = (destination / "sync.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    import wandb

    config = run_config(root)
    settings = wandb.Settings(console="off", disable_git=True, disable_code=True,
                              save_code=False, x_disable_stats=True, x_disable_meta=True)
    with wandb.init(entity=options.entity, project=options.project,
                    id=options.run_id or root.name, name=root.name,
                    group="qwen3-asr-sot-v2", job_type="training-metrics",
                    config=config, resume="allow", dir=str(destination), settings=settings) as run:
        run.define_metric("global_step")
        for prefix in ("train/*", "validation/*", "eval/*", "perf/*"):
            run.define_metric(prefix, step_metric="global_step")
        run.define_metric("sync_event", hidden=True)
        # The remote history is the acknowledgement record after a sync-process restart.
        remote = wandb.Api().run(f"{run.entity}/{run.project}/{run.id}")
        seen = {row["sync_event"] for row in remote.scan_history(keys=["sync_event"], page_size=1000)}
        identity = {"entity": run.entity, "project": run.project, "run_id": run.id,
                    "url": run.url, "pid": os.getpid()}
        (destination / "run.json").write_text(json.dumps(identity, indent=2) + "\n")
        print(json.dumps(identity), flush=True)
        while True:
            events = collect_events(root)
            uploaded = upload_events(run, events, seen)
            latest = max((event["global_step"] for event in events.values()), default=0)
            run.summary["latest_training_step"] = max(
                (event["global_step"] for key, event in events.items() if key.startswith("train:")), default=0)
            run.summary["latest_evaluated_step"] = max(
                (event["global_step"] for key, event in events.items() if key.startswith("evaluation:")), default=0)
            run.summary["sync_event_count"] = len(seen)
            print(json.dumps({"uploaded_events": uploaded, "total_events": len(seen),
                              "latest_step": latest}), flush=True)
            if not options.follow:
                break
            time.sleep(options.interval)


if __name__ == "__main__":
    main()
