"""Per-rank SFT throughput and timing; metadata never enters model.forward."""
from __future__ import annotations

import json
import gc
import logging
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from types import MethodType

import torch
from transformers import TrainerCallback

LOG = logging.getLogger(__name__)


class PerformanceCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, rows):
        started = time.perf_counter()
        clean, metadata = [], []
        for row in rows:
            row = dict(row)
            metadata.append(row.pop("_performance"))
            clean.append(row)
        batch = self.collator(clean)
        totals = {k: sum(m.get(k, 0.0) for m in metadata) for k in
                  ("decode_s", "wave_augment_s", "audio_seconds", "encode_s", "prepare_s", "prepare_cpu_s")}
        totals["samples"] = len(rows)
        totals["prepare_max_s"] = max(m["prepare_s"] for m in metadata)
        totals["sources"] = dict(Counter(m["dataset_id"] for m in metadata))
        for name, mask in (("tokens", batch.get("attention_mask")),
                           ("audio_frames", batch.get("feature_attention_mask"))):
            if mask is not None:
                totals[name] = int(mask.sum())
                totals[name + "_capacity"] = mask.numel()
                totals[name + "_max_length"] = mask.shape[-1]
        if batch.get("labels") is not None:
            totals["supervised_tokens"] = int((batch["labels"] != -100).sum())
        totals["collate_s"] = time.perf_counter() - started
        batch["_performance"] = totals
        return batch


class PerformanceCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.totals = Counter()
        self.sources = Counter()
        self.maxima = {}
        self.events = []
        self.monitor = None
        self.gpu_file = None
        self.started = time.perf_counter()
        self.last_step = 0
        self.gc_pauses = []
        self.gc_registered = False
        self.pid = os.getpid()

    def record_gc(self, phase, info):
        if info["generation"] != 2 or os.getpid() != self.pid:
            return
        if phase == "start":
            self.gc_started = time.perf_counter()
        else:
            self.gc_pauses.append({"timestamp": time.time(),
                                   "seconds": time.perf_counter() - self.gc_started,
                                   "collected": info["collected"]})

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.perf_counter()
        self.last_step = state.global_step
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.path = output / f"performance-rank{args.process_index}.jsonl"
        self.gc_path = output / f"gc-rank{args.process_index}.json"
        gc.callbacks.append(self.record_gc)
        self.gc_registered = True
        if state.is_world_process_zero and torch.cuda.is_available() and shutil.which("nvidia-smi"):
            self.gpu_file = (output / "gpu.csv").open("a")
            command = ["nvidia-smi", "--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem", "--format=csv,nounits", "--loop-ms=1000"]
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                command += ["--id=" + os.environ["CUDA_VISIBLE_DEVICES"]]
            self.monitor = subprocess.Popen(command, stdout=self.gpu_file, stderr=subprocess.STDOUT)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        self.totals["update_wall_s"] += time.perf_counter() - self.step_started

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs or not self.totals["samples"]:
            return
        # Synchronize once per logging interval, never once per training batch.
        if self.events:
            self.events[-1][1].synchronize()
            self.totals["forward_backward_cuda_span_s"] = sum(a.elapsed_time(b) / 1000 for a, b in self.events)
        now = time.perf_counter()
        wall = now - self.started
        row = dict(self.totals)
        row.update(self.maxima)
        row.update({k: logs[k] for k in ("loss", "grad_norm", "learning_rate") if k in logs})
        row.update(timestamp=time.time(), rank=args.process_index, step=state.global_step,
                   updates=state.global_step - self.last_step, interval_wall_s=wall,
                   samples_per_s=row["samples"] / wall,
                   audio_seconds_per_s=row["audio_seconds"] / wall,
                   batch_fetch_fraction=row["batch_fetch_s"] / wall,
                   sources=dict(self.sources))
        for name in ("tokens", "audio_frames"):
            if row.get(name + "_capacity"):
                row[name + "_padding_fraction"] = 1 - row[name] / row[name + "_capacity"]
        if torch.cuda.is_available():
            row["cuda_allocated_gib"] = torch.cuda.memory_allocated() / 1024**3
            row["cuda_reserved_gib"] = torch.cuda.memory_reserved() / 1024**3
            row["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if state.is_world_process_zero:
            LOG.warning("Training performance: %s", json.dumps(row))
        self.totals.clear()
        self.sources.clear()
        self.maxima.clear()
        self.events.clear()
        self.started = time.perf_counter()
        self.last_step = state.global_step

    def on_train_end(self, args, state, control, **kwargs):
        self.close()

    def close(self):
        if self.gc_registered:
            gc.callbacks.remove(self.record_gc)
            self.gc_registered = False
            self.gc_path.write_text(json.dumps(self.gc_pauses, indent=2) + "\n")
        if self.monitor is not None:
            self.monitor.terminate()
            self.monitor.wait(timeout=5)
            self.monitor = None
        if self.gpu_file is not None:
            self.gpu_file.close()
            self.gpu_file = None


def install_performance_logging(trainer):
    callback = PerformanceCallback(trainer)
    trainer.add_callback(callback)
    get_batches = trainer.get_batch_samples
    training_step = trainer.training_step

    def measured_batches(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return get_batches(*args, **kwargs)
        finally:
            callback.totals["batch_fetch_s"] += time.perf_counter() - started

    def measured_step(self, model, inputs, *args, **kwargs):
        metrics = inputs.pop("_performance")
        callback.sources.update(metrics.pop("sources"))
        for key in list(metrics):
            if key == "prepare_max_s" or key.endswith("_max_length"):
                callback.maxima[key] = max(callback.maxima.get(key, 0), metrics.pop(key))
        callback.totals.update(metrics)
        callback.totals["microbatches"] += 1
        started = time.perf_counter()
        events = None
        if torch.cuda.is_available():
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record()
        result = training_step(model, inputs, *args, **kwargs)
        if events:
            events[1].record()
            callback.events.append(events)
        callback.totals["forward_backward_host_s"] += time.perf_counter() - started
        return result

    trainer.get_batch_samples = MethodType(measured_batches, trainer)
    trainer.training_step = MethodType(measured_step, trainer)
    return callback
