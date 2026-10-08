"""Score checkpoint-37000 on the current plain and enrolled test splits.

Plain and enrolled rows both report cpCER/cpWER, tcpCER/tcpWER at 5 s and 1 s,
and DER with a 0.25 s collar. Enrolled T labels stay fixed. The 20261001 Emilia
dialog and long tests are omitted; those recordings were replaced in 20261004.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import yaml

from open_audio_llm.data.sot import TIMESTAMP_FORMAT
from open_audio_llm.eval.sot_timestamps import summarize_timed_sot
from open_audio_llm.eval.target_sot import summarize_target_sot

ROOTS = "/222042021/mingdong/data/sot-multispeaker/synthetic-v2-20260915/roots.json"
CATALOGS = {
    "20261001": "/ai_sds_wuzz/DATA_ASR/Derived/target-sot-supervision-20261001/catalog.jsonl",
    "20261004": "/ai_sds_wuzz/DATA_ASR/Derived/target-sot-supervision-20261004/catalog.jsonl",
    "20261006": "/ai_sds_wuzz/DATA_ASR/Derived/target-sot-supervision-aishell4-test-20261006/catalog.jsonl",
}
ENROLLMENT = {
    "probability": 1.0,
    "min_seconds": 1.0,
    "max_seconds": 5.0,
    "max_targets": 5,
    "absent_probability": 0.0,
    "mode": "all",
}
SPLITS = [
    ("20261004", "dialog_zh_test_2to5_plain", False),
    ("20261004", "dialog_zh_test_2to5_enroll", True),
    ("20261004", "dialog_zh_test_6plus_plain", False),
    ("20261004", "dialog_zh_test_6plus_enroll", True),
    ("20261004", "dialog_en_test_2to5_plain", False),
    ("20261004", "dialog_en_test_2to5_enroll", True),
    ("20261004", "dialog_en_test_6plus_plain", False),
    ("20261004", "dialog_en_test_6plus_enroll", True),
    ("20261004", "long_zh_test_plain", False),
    ("20261004", "long_en_test_plain", False),
    ("20261001", "alimeeting_test_30s_plain", False),
    ("20261001", "alimeeting_test_30s_enroll", True),
    ("20261001", "alimeeting_test_120s_plain", False),
    ("20261001", "alimeeting_test_120s_enroll", True),
    ("20261001", "alimeeting_test_300s_plain", False),
    ("20261001", "alimeeting_test_300s_enroll", True),
    ("20261001", "alimeeting_test_600s_plain", False),
    ("20261001", "alimeeting_test_600s_enroll", True),
    ("20261001", "ami_test_30s_plain", False),
    ("20261001", "ami_test_30s_enroll", True),
    ("20261001", "ami_test_120s_plain", False),
    ("20261001", "ami_test_120s_enroll", True),
    ("20261001", "ami_test_300s_plain", False),
    ("20261001", "ami_test_300s_enroll", True),
    ("20261001", "ami_test_600s_plain", False),
    ("20261001", "ami_test_600s_enroll", True),
    ("20261006", "aishell4_test_30s_plain", False),
    ("20261006", "aishell4_test_30s_enroll", True),
    ("20261006", "aishell4_test_120s_plain", False),
    ("20261006", "aishell4_test_120s_enroll", True),
    ("20261006", "aishell4_test_300s_plain", False),
    ("20261006", "aishell4_test_300s_enroll", True),
    ("20261006", "aishell4_test_600s_plain", False),
    ("20261006", "aishell4_test_600s_enroll", True),
]


def transcript_body(text):
    return text.split("<asr_text>", 1)[-1].strip()


def data_config(output: Path):
    validation = []
    for version, split, enrolled in SPLITS:
        source = {"dataset_id": "target_sot_eval", "version": version, "split": split}
        if enrolled:
            source["enrollment"] = dict(ENROLLMENT)
        validation.append(source)
    return {
        "catalog": str((output / "catalog.jsonl").resolve()),
        "roots": ROOTS,
        "train": [{"dataset_id": "unused", "version": "unused", "split": "unused"}],
        "validation": validation,
        "metadata_cache": str((output / "metadata-cache").resolve()),
        "seed": 42,
    }


def _isolated_env():
    os.environ.pop("AUDIO_DATA_CATALOG", None)
    os.environ.pop("AUDIO_DATA_METADATA_CACHE", None)


def prepare(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.jsonl").exists():
        print(json.dumps({"manifest": "kept"}))
        return
    catalog = output / "catalog.jsonl"
    with catalog.open("w") as sink:
        for version in ("20261001", "20261004", "20261006"):
            sink.write(Path(CATALOGS[version]).read_text().strip() + "\n")
    _isolated_env()
    config = data_config(output)
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config

    dataset = CatalogSwiftDataset(read_data_config(output / "data.yaml", use_env=False),
                                  training=False, message_format="qwen3_asr")
    rows = []
    for source_index, span in enumerate(dataset.source_ranges):
        source = dataset.sources[source_index]
        for index in span:
            record = dataset.records[index].record
            fixed = record.metadata.get("fixed_enrollment") or {}
            rows.append({
                "index": index,
                "id": record.id,
                "split": source["split"],
                "version": source["version"],
                "language": record.language,
                "enrolled": "enrollment" in source,
                "count": len(fixed.get("enrollments") or []),
                "mode": fixed.get("mode", "all"),
                "duration": float(dataset.records.durations[index]),
            })
    rows.sort(key=lambda row: row["duration"], reverse=True)
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "predictions").mkdir(exist_ok=True)
    print(json.dumps({"examples": len(rows), "hours": round(sum(row["duration"] for row in rows) / 3600, 2)}))


def _done_ids(path: Path):
    """Keep completed rows. Failed rows are retried instead of being scored as empty."""
    done = set()
    if not path.exists():
        return done
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok":
            kept.append(line)
            done.add(row["id"])
    path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    return done


@contextmanager
def _queue_lock(output: Path):
    handle = (output / "queue.lock").open("a+")
    fcntl.flock(handle, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _ok_ids(output: Path):
    done = set()
    predictions = output / "predictions"
    if not predictions.exists():
        return done
    for path in predictions.glob("rank*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok":
                done.add(row["id"])
    return done


def ensure_queue(output: Path):
    """One list for all GPUs. A new driver rebuilds it from rows that are not finished."""
    run_id = str(os.getppid())
    ready = output / "queue.ready"
    with _queue_lock(output):
        if ready.exists() and ready.read_text(encoding="utf-8").strip() == run_id:
            lines = [line for line in (output / "queue.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            nxt = int((output / "queue_next").read_text(encoding="utf-8"))
            return max(len(lines) - nxt, 0)
        predictions = output / "predictions"
        predictions.mkdir(exist_ok=True)
        for path in predictions.glob("rank*.jsonl"):
            _done_ids(path)
        manifest = [
            json.loads(line) for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        done = _ok_ids(output)
        pending = [row for row in manifest if row["id"] not in done]
        (output / "queue.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pending), encoding="utf-8")
        (output / "queue_next").write_text("0", encoding="utf-8")
        ready.write_text(run_id, encoding="utf-8")
        return len(pending)


def claim_one(output: Path):
    with _queue_lock(output):
        lines = [line for line in (output / "queue.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        nxt = int((output / "queue_next").read_text(encoding="utf-8"))
        if nxt >= len(lines):
            return None
        (output / "queue_next").write_text(str(nxt + 1), encoding="utf-8")
        return json.loads(lines[nxt])


def work(output: Path, rank: int, model_dir: str, batch_size: int = 16):
    _isolated_env()
    # The engine subprocess deadlocks during CUDA init on this machine.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config
    from open_audio_llm.integrations.vllm.target_sot import TargetSOTVLLM

    sink_path = output / "predictions" / f"rank{rank}.jsonl"
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    queued = ensure_queue(output)
    print(f"rank {rank}: shared queue {queued} left, dynamic {batch_size}, encoder cuda", flush=True)
    if queued == 0:
        return
    dataset = CatalogSwiftDataset(read_data_config(output / "data.yaml", use_env=False),
                                  training=False, message_format="qwen3_asr")
    model = TargetSOTVLLM(model_dir, max_model_len=32768, gpu_memory_utilization=0.7, frontend_device="cuda")
    # Encode one clip when a slot is free, then let vLLM step. Every GPU
    # takes the next recording from the same queue.
    in_flight = {}
    exhausted = False
    completed = 0
    announced_full = False
    announced_mismatch = False

    def write_row(row):
        nonlocal completed
        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        sink.flush()
        completed += 1
        if completed == 1 or completed % batch_size == 0 or exhausted and not in_flight:
            print(
                f"rank {rank}: {completed} done {row['status']} {row.get('split', '')} "
                f"in_flight {len(in_flight)}",
                flush=True,
            )

    with sink_path.open("a", encoding="utf-8") as sink:
        while not exhausted or in_flight:
            if len(in_flight) < batch_size and not exhausted:
                item = claim_one(output)
                if item is None:
                    exhausted = True
                else:
                    record = dataset.records[item["index"]].record
                    if record.id != item["id"]:
                        raise RuntimeError(f"Manifest drifted at {item['index']}: {record.id} != {item['id']}")
                    try:
                        sample = dataset[item["index"]]
                        enrollments = sample.get("chat_template_kwargs", {}).get("enroll_wavs", [])
                        mode = sample.get("enrollment_view", {}).get("mode", "all")
                        request = model._request(
                            sample["audios"][0], enrollments, mode, max_tokens=16384, fit_context=True,
                        )
                        request_id = model.submit(request)
                        in_flight[request_id] = {
                            "item": item,
                            "mode": mode,
                            "count": len(enrollments),
                            "duration": sample["duration"],
                            "reference": transcript_body(sample["solution"]),
                            "prompt_tokens": request["prompt_tokens"],
                            "output_token_budget": request["max_tokens"],
                        }
                        if len(in_flight) == batch_size and not announced_full:
                            announced_full = True
                            print(f"rank {rank}: in_flight {batch_size}", flush=True)
                    except Exception as exc:
                        write_row({**item, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
            if not in_flight:
                if exhausted:
                    break
                continue
            for request_id, result in model.step():
                held = in_flight.pop(request_id, None)
                if held is None:
                    if not announced_mismatch:
                        announced_mismatch = True
                        print(f"rank {rank}: unexpected finished request {request_id}", flush=True)
                    continue
                item = held["item"]
                write_row({
                    **item,
                    "status": "ok",
                    "mode": held["mode"],
                    "count": held["count"],
                    "duration": held["duration"],
                    "reference": held["reference"],
                    "prediction": transcript_body(result["prediction"]),
                    "raw_prediction": result["prediction"],
                    "finish_reason": result["finish_reason"],
                    "prompt_tokens": held["prompt_tokens"],
                    "output_token_budget": held["output_token_budget"],
                })


SCORE_STAMP = "meeteval-resume-v1"


def _load_score_cache(path: Path):
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if payload.get("stamp") != SCORE_STAMP:
        return {}
    return payload.get("metrics", {})


def _save_score_cache(path: Path, metrics):
    payload = {"stamp": SCORE_STAMP, "metrics": metrics}
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def score_rows(rows, cache_path: Path | None = None):
    grouped = defaultdict(list)
    failures = []
    for row in rows:
        if row.get("status") != "ok":
            failures.append(row)
            continue
        grouped[row["split"]].append(row)
    metrics = _load_score_cache(cache_path) if cache_path else {}
    for split, items in sorted(grouped.items()):
        cached = metrics.get(split)
        if cached and cached.get("examples") == len(items):
            print(f"scoring {split} {len(items)} cached", flush=True)
            continue
        print(f"scoring {split} {len(items)}", flush=True)
        if items[0]["enrolled"]:
            metrics[split] = summarize_target_sot(items)
        else:
            timed = [{
                "source": split, "language": row["language"], "task": "speaker_attributed_asr",
                "duration": row["duration"], "sot_output_format": TIMESTAMP_FORMAT,
                "reference": row["reference"], "prediction": row["prediction"],
            } for row in items]
            metrics[split] = summarize_timed_sot(timed, items[0]["language"] in {"zh", "Chinese"})
        metrics[split]["examples"] = len(items)
        metrics[split]["output_limit_examples"] = sum(row.get("finish_reason") == "length" for row in items)
        if cache_path:
            _save_score_cache(cache_path, metrics)
            print(
                f"saved {split} cp={_fmt(metrics[split].get('cp_error_rate'))} "
                f"tcp5={_fmt(metrics[split].get('tcp_error_rate_5s'))} "
                f"der={_fmt(metrics[split].get('der'))}",
                flush=True,
            )
    return {split: metrics[split] for split in sorted(grouped)}, failures


def load_predictions(output: Path):
    rows = []
    for path in sorted((output / "predictions").glob("rank*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def write_summary(output: Path):
    manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines() if line.strip()]
    rows = load_predictions(output)
    metrics, failures = score_rows(rows, output / "score-partial.json")
    completed = [row for row in rows if row.get("status") == "ok"]
    summary = {
        "checkpoint": "checkpoint-37000",
        "examples": len(manifest),
        "completed_examples": len(completed),
        "failed_examples": len(failures),
        "coverage": len(rows) / len(manifest) if manifest else 0,
        "scored_coverage": len(completed) / len(manifest) if manifest else 0,
        "note": "tcp token times are estimated from utterance bounds. DER collar is 0.25s and counts overlap. Enrolled T labels are fixed.",
        "metrics": metrics,
        "failures": [{"id": row.get("id"), "split": row.get("split"), "reason": row.get("reason")} for row in failures],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["split\texamples\tcp\ttcp_5s\ttcp_1s\tder\tformat_valid\ttruncated"]
    for split, values in metrics.items():
        lines.append("\t".join([
            split,
            str(values.get("examples")),
            _fmt(values.get("cp_error_rate")),
            _fmt(values.get("tcp_error_rate_5s")),
            _fmt(values.get("tcp_error_rate_1s")),
            _fmt(values.get("der")),
            _fmt(values.get("format_valid_rate")),
            str(values.get("output_limit_examples")),
        ]))
    (output / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _fmt(value):
    return "" if value is None else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "work", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--model", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "work":
        if not args.model:
            parser.error("work requires --model")
        work(args.output, args.rank, args.model, args.batch_size)
    else:
        write_summary(args.output)


if __name__ == "__main__":
    main()
