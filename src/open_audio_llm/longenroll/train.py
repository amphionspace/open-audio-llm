"""Train LongEnrollTSASR (AmphionASR v5) on a Catalog data recipe.

Launched by ``open-audio-llm train`` (torch.distributed.run). fp32 master weights
with bf16 autocast; AdamW state sharded over ranks (ZeRO-1). Batches come from
the Catalog duration sampler, which counts TS-ASR mixtures only, so the
minutes-long enrollments do not shrink the batch. Metrics are written in the
framework's training/logging.jsonl and performance-rank0.jsonl formats, which
the CPU W&B recorder uploads.

A checkpoint holds model.pt (trainable weights, fp32), optimizer.rank{r}.pt
(that rank's AdamW shard) and trainer_state.pt (scheduler, step, world size,
Catalog sampler cursor), written last.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
import unicodedata
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data.catalog_dataset import read_data_config
from ..data.catalog_sampler import CatalogBatchSampler
from .backbone import PromptBuilder, load_qwen3_asr
from .config import Config
from .data import TASK_NAMES, LongEnrollCatalogDataset, collate
from .ecapa import load_ecapa
from .model import EnrollmentState, LongEnrollTSASR


def str2bool(value):
    if str(value).lower() not in {"true", "false"}:
        raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")
    return str(value).lower() == "true"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_config", required=True)
    p.add_argument("--model", required=True, help="Qwen3-ASR-1.7B directory")
    p.add_argument("--ecapa_model", required=True, help="wespeaker ECAPA-TDNN directory (avg_model.pt)")
    p.add_argument("--model_config", required=True, help="YAML with the long-enrollment `model:` section")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from_checkpoint")
    p.add_argument("--max_steps", type=int, required=True)
    p.add_argument("--learning_rate", type=float, default=1e-4, help="new modules")
    p.add_argument("--backbone_lr", type=float, default=1e-5, help="Qwen3-ASR encoder layers and LLM")
    p.add_argument("--gate_lr_mult", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--adam_fused", type=str2bool, default=True,
                   help="fused CUDA AdamW kernel; the per-tensor update cost ~0.18 s per step on 2 GPUs")
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--min_lr_ratio", type=float, default=0.2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    p.add_argument("--deepspeed", choices=("none", "zero1"), default="none",
                   help="zero1: DeepSpeed ZeRO-1 with bf16 weights and fp32 master weights in the optimizer; "
                        "none: fp32 weights with bf16 autocast, torch ZeRO-1 over ranks")
    p.add_argument("--memory_probe", type=int, default=0,
                   help="run this many steps on a worst-case synthetic batch, report peak memory and exit")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--eval_on_start", type=str2bool, default=False)
    p.add_argument("--decode_samples", type=int, default=64,
                   help="validation examples per rank decoded greedily for error rates")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--save_steps", type=int, default=2000, help="0 disables checkpoints")
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--dataloader_num_workers", type=int, default=8)
    p.add_argument("--dataloader_prefetch_factor", type=int, default=4)
    p.add_argument("--dataloader_eval_num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def lr_lambda(step, warmup, total, min_ratio):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


# ------------------------------------------------------------------ error rates
def _normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)


def _units(text, language):
    text = _normalize(text)
    if language == "Chinese" or re.search(r"[一-鿿]", text):
        return [c for c in text if not c.isspace()]
    return text.split()


def _edits(ref, hyp):
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def score(answer, hypothesis, language):
    """(edits, reference units) on the text after <asr_text>; absent targets count outputs."""
    ref = answer.split("<asr_text>", 1)[-1]
    hyp = hypothesis.split("<asr_text>", 1)[-1]
    if not ref.strip():
        return int(bool(hyp.strip())), 1
    ref_units = _units(ref, language)
    return _edits(ref_units, _units(hyp, language)), len(ref_units)


# ------------------------------------------------------------------ helpers
TRAIN_KEYS = (*(f"{kind}_{task}" for task in TASK_NAMES for kind in ("ce", "tok")),
              "steps", "samples", "mix_sec", "enroll_sec", "ts_samples", "wait_s", "compute_s",
              "worker_s", "decode_s", "augment_s", "grad_norm", "forward_s", "backward_s", "optimizer_s",
              "tokens_padded", "tokens_real")
EVAL_KEYS = tuple(f"{kind}_{task}" for task in TASK_NAMES for kind in ("ce", "tok", "err", "ref"))


class Metrics:
    """Sums over a fixed key list, all-reduced on read. Every rank sends the same keys in
    the same order, even when its batches held no example of some task."""

    def __init__(self, keys):
        self.keys = tuple(keys)
        self.values = dict.fromkeys(self.keys, 0.0)

    def add(self, key, value):
        self.values[key] += float(value)

    def add_tasks(self, prefix, tasks, sample_ce, sample_tokens):
        for task, ce, tokens in zip(tasks, sample_ce.tolist(), sample_tokens.tolist()):
            self.add(f"{prefix}ce_{task}", ce)
            self.add(f"{prefix}tok_{task}", tokens)

    def reduce(self, device, distributed):
        stats = torch.tensor([self.values[k] for k in self.keys], device=device, dtype=torch.float64)
        if distributed:
            dist.all_reduce(stats)
        self.values = dict.fromkeys(self.keys, 0.0)
        return dict(zip(self.keys, stats.tolist()))


def task_losses(stats, prefix=""):
    out = {}
    for task in TASK_NAMES:
        tokens = stats.get(f"{prefix}tok_{task}", 0.0)
        if tokens:
            out[f"ce_{task}"] = stats[f"{prefix}ce_{task}"] / tokens
    return out


def append_jsonl(path, row):
    with open(path, "a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def loader_for(dataset, sampler, workers, prefetch, pad_id, persistent):
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=workers,
        collate_fn=partial(collate, pad_id=pad_id), pin_memory=True,
        prefetch_factor=prefetch if workers else None,
        persistent_workers=persistent and workers > 0,
    )


def deepspeed_config(args):
    return {
        # Batches come from the Catalog sampler; DeepSpeed only needs a placeholder size.
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": args.max_grad_norm,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 1, "overlap_comm": True, "contiguous_gradients": True,
                              "reduce_bucket_size": 5e8, "allgather_bucket_size": 5e8},
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 2**31 - 1,
        "wall_clock_breakdown": False,
    }


# Worst case the batching limits allow for one TS-ASR batch: every mixture at the 30 s cap,
# every enrollment at the 5-minute cap, and a long answer.
PROBE_MIX_SECONDS, PROBE_ENROLL_SECONDS, PROBE_ANSWER_CHARS = 30.0, 300.0, 250


def worst_case_batch(features, batching, pad_id):
    import numpy as np

    count = int(batching.get("max_samples") or 1)
    mix_seconds = min(PROBE_MIX_SECONDS, float(batching.get("max_duration") or 1e9) / count)
    noise = np.random.default_rng(0)
    examples = []
    for _ in range(count):
        sample = {
            "task": "ts_asr", "dataset_id": "memory_probe",
            "audios": [noise.standard_normal(int(PROBE_ENROLL_SECONDS * 16000)).astype(np.float32) * 0.05],
            "mix_wav": noise.standard_normal(int(mix_seconds * 16000)).astype(np.float32) * 0.05,
            "messages": [{"role": "system", "content": ""}, {"role": "user", "content": "<audio>"},
                         {"role": "assistant", "content": "language Chinese<asr_text>" + "测" * PROBE_ANSWER_CHARS}],
        }
        examples.append(features(sample, None))
    return collate(examples, pad_id)


def save_checkpoint(out_dir, step, model, local_optimizer, scheduler, sampler, rank, world, limit, distributed,
                    engine=None):
    ckpt = out_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    if engine is not None:
        # bf16 module weights + per-rank fp32 master / AdamW partitions, DeepSpeed layout.
        engine.save_checkpoint(str(out_dir), tag=ckpt.name)
    else:
        torch.save(local_optimizer.state_dict(), ckpt / f"optimizer.rank{rank}.pt")
        if rank == 0:
            torch.save(model.trainable_state_dict(), ckpt / "model.pt")
    if distributed:
        dist.barrier()
    if rank == 0:  # trainer_state.pt last: it marks the checkpoint complete
        torch.save({"scheduler": scheduler.state_dict(), "step": step, "world": world,
                    "sampler": sampler.state_dict()}, ckpt / "trainer_state.pt")
        done = sorted((c for c in out_dir.glob("checkpoint-*") if (c / "trainer_state.pt").exists()),
                      key=lambda c: int(c.name.split("-")[-1]))
        for old in done[:-limit] if limit > 0 else []:
            shutil.rmtree(old, ignore_errors=True)
    if distributed:
        dist.barrier()


def validation_batches(dataset, rank, world):
    """Every validation record once, sorted by batching cost and cut with the recipe's
    batching limits, then dealt to ranks. (CatalogBatchSampler mixes training sources
    and rejects a validation subset that leaves a source empty.)"""
    options = dataset.config.get("batching", {})
    max_duration, max_samples = options.get("max_duration"), options.get("max_samples")
    costs = dataset.records.durations
    batches, batch, total = [], [], 0.0
    for index in sorted(range(len(dataset)), key=lambda i: float(costs[i])):
        cost = float(costs[index])
        if batch and ((max_samples and len(batch) >= max_samples)
                      or (max_duration and total + cost > max_duration)):
            batches.append(batch)
            batch, total = [], 0.0
        batch.append((0, index, None))
        total += cost
    if batch:
        batches.append(batch)
    return batches[rank::world]


@torch.no_grad()
def evaluate(model, dataset, args, rank, world, device, distributed, tokenizer, pad_id):
    batches = validation_batches(dataset, rank, world)
    # Batches run from the shortest clips to the longest: decode from every stride-th
    # batch so the error rates cover all durations and tasks, not only the shortest.
    stride = max(1, len(batches) // max(1, args.decode_samples))
    per_batch = max(1, math.ceil(args.decode_samples / max(1, math.ceil(len(batches) / stride))))
    loader = loader_for(dataset, batches, args.dataloader_eval_num_workers, 2, pad_id, persistent=False)
    model.eval()
    metrics, decoded = Metrics(EVAL_KEYS), 0
    started = time.perf_counter()
    for number, batch in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
        metrics.add_tasks("", batch["task"], out["sample_ce"], out["sample_tokens"])
        if decoded >= args.decode_samples or number % stride:
            continue
        take = min(len(batch["task"]), args.decode_samples - decoded, per_batch)
        prompts = []
        for row in range(take):
            labels = batch["labels"][row]
            answer_start = int(torch.nonzero(labels.ne(-100))[0])
            prompts.append(batch["input_ids"][row, :answer_start].tolist())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enroll = model.encode_enrollment(batch["enroll_mels"][:take], batch["ecapa_fbank"][:take])
            ablation = ["both" if m == 0 else "none" for m in batch["mode"][:take].tolist()]
            hypotheses = []
            for mode in ("both", "none"):  # generate() takes one ablation per call
                rows = [i for i, a in enumerate(ablation) if a == mode]
                if not rows:
                    continue
                index = torch.tensor(rows, device=device)
                state = EnrollmentState(enroll.memory.index_select(index),
                                        enroll.global_tokens.index_select(0, index),
                                        enroll.ecapa.index_select(0, index))
                ids = model.generate([batch["mix_mels"][i] for i in rows], [prompts[i] for i in rows], state,
                                     ablation=mode, max_new_tokens=args.max_new_tokens, pad_id=pad_id)
                hypotheses += list(zip(rows, ids))
        for row, ids in sorted(hypotheses):
            text = tokenizer.decode(ids, skip_special_tokens=False)
            edits, units = score(batch["answer"][row], text, batch["language"][row])
            metrics.add(f"err_{batch['task'][row]}", edits)
            metrics.add(f"ref_{batch['task'][row]}", units)
        decoded += take
    stats = metrics.reduce(device, distributed)
    tokens = sum(stats.get(f"tok_{t}", 0.0) for t in TASK_NAMES)
    result = {"eval_loss": sum(stats.get(f"ce_{t}", 0.0) for t in TASK_NAMES) / max(tokens, 1.0)}
    result.update({f"eval_{k}": v for k, v in task_losses(stats).items()})
    for task in TASK_NAMES:
        if stats.get(f"ref_{task}"):
            # Absent targets: share of outputs that are not empty.
            name = "false_alarm" if task == "ts_absent" else "error_rate"
            result[f"eval_{name}_{task}"] = stats[f"err_{task}"] / stats[f"ref_{task}"]
    result["eval_runtime"] = time.perf_counter() - started
    model.train()
    return result


def main(argv=None):
    args = parse_args(argv)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if args.deepspeed != "none":
        import deepspeed

        deepspeed.init_distributed(dist_backend="nccl", timeout=timedelta(hours=2))
    elif distributed:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    model_cfg = Config(yaml.safe_load(Path(args.model_config).read_text()))
    cfg = Config({"model": model_cfg.model, "train": {"gradient_checkpointing": args.gradient_checkpointing}})
    asr, processor = load_qwen3_asr(args.model, dtype=torch.float32, device=device)
    ecapa = load_ecapa(args.ecapa_model).to(device)
    model = LongEnrollTSASR(asr, ecapa, cfg).to(device)
    groups = model.optimizer_groups(args.learning_rate, args.backbone_lr, args.gate_lr_mult, args.weight_decay)
    adam = {"betas": (args.adam_beta1, args.adam_beta2), "fused": args.adam_fused}
    schedule = partial(lr_lambda, warmup=args.warmup_steps, total=args.max_steps, min_ratio=args.min_lr_ratio)
    engine = None
    if args.deepspeed != "none":
        engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model, optimizer=torch.optim.AdamW(groups, **adam), config=deepspeed_config(args),
            lr_scheduler=lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, schedule))
        local_optimizer = optimizer
        # DeepSpeed casts the whole module to bf16. The frozen ECAPA (no optimizer state) goes back
        # to fp32: its BatchNorm cannot mix bf16 statistics with fp32 activations.
        model.ecapa.float()
    elif distributed:
        optimizer = ZeroRedundancyOptimizer(groups, optimizer_class=torch.optim.AdamW, **adam)
        local_optimizer = optimizer.optim
    else:
        optimizer = local_optimizer = torch.optim.AdamW(groups, **adam)
    if engine is None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    data_config = read_data_config(args.data_config)
    pad_id = PromptBuilder(processor.tokenizer).pad_id
    train_set = LongEnrollCatalogDataset(data_config, processor, training=True, collect_metrics=True)
    sampler = CatalogBatchSampler(train_set, rank=rank, world_size=world, shuffle=True)
    valid_set = (LongEnrollCatalogDataset(data_config, processor, training=False)
                 if data_config.get("validation") else None)

    step = 0
    if args.resume_from_checkpoint:
        ckpt = Path(args.resume_from_checkpoint)
        state = torch.load(ckpt / "trainer_state.pt", map_location="cpu", weights_only=False)
        if int(state["world"]) != world:
            raise RuntimeError(f"{ckpt} was saved with {state['world']} ranks; resume needs the same count")
        if engine is not None:
            engine.load_checkpoint(str(ckpt.parent), tag=ckpt.name)
        else:
            model.load_trainable_state_dict(torch.load(ckpt / "model.pt", map_location="cpu", mmap=True))
            local_optimizer.load_state_dict(torch.load(ckpt / f"optimizer.rank{rank}.pt", map_location="cpu"))
            scheduler.load_state_dict(state["scheduler"])
            for group, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
                group["lr"] = lr
        sampler.load_state_dict(state["sampler"])
        step = int(state["step"])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(json.dumps({"event": "start", "trainable_params": trainable, "world": world, "step": step,
                          "train_records": len(train_set),
                          "validation_records": len(valid_set) if valid_set else 0}), flush=True)

    model.train()
    if engine is not None:
        ddp = engine
    else:
        ddp = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True) if distributed else model

    def train_step(batch, marks):
        marks[0].record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = ddp(batch)
        marks[1].record()
        if engine is not None:
            engine.backward(out["loss"])
            marks[2].record()
            engine.step()  # clipping, optimizer, LR schedule, zero_grad
            grad_norm = engine.get_global_grad_norm()
        else:
            out["loss"].backward()
            marks[2].record()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        marks[3].record()
        return out, float(grad_norm if grad_norm is not None else 0.0)
    loader = loader_for(train_set, sampler, args.dataloader_num_workers, args.dataloader_prefetch_factor,
                        pad_id, persistent=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    def run_eval():
        result = evaluate(model, valid_set, args, rank, world, device, distributed, processor.tokenizer, pad_id)
        if rank == 0:
            append_jsonl(out_dir / "logging.jsonl", {"global_step/max_steps": f"{step}/{args.max_steps}", **result})
            print(json.dumps({"event": "eval", "step": step, **result}), flush=True)

    if args.memory_probe:
        batch = worst_case_batch(train_set.features, data_config.get("batching", {}), pad_id)
        for probe in range(args.memory_probe):
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            train_step(batch, [torch.cuda.Event(enable_timing=True) for _ in range(4)])
            torch.cuda.synchronize(device)
            peak = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                                device=device, dtype=torch.float64)
            if distributed:
                dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(json.dumps({"event": "memory_probe", "step": probe + 1, "batch_samples": len(batch["task"]),
                                  "llm_tokens_padded": batch["input_ids"].numel(),
                                  "peak_allocated_gib": round(peak[0].item() / 2**30, 2),
                                  "peak_reserved_gib": round(peak[1].item() / 2**30, 2),
                                  "step_s": round(time.perf_counter() - started, 3)}), flush=True)
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    if valid_set is not None and args.eval_on_start and step == 0:
        run_eval()

    sampler.set_epoch(sampler.epoch)
    data_iter = iter(loader)
    window = Metrics(TRAIN_KEYS)
    tick = time.perf_counter()
    while step < args.max_steps:
        waited = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            sampler.set_epoch(sampler.epoch + 1)
            data_iter = iter(loader)
            batch = next(data_iter)
        computed = time.perf_counter()
        marks = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        out, grad_norm = train_step(batch, marks)
        sampler.mark_consumed()
        torch.cuda.synchronize(device)
        finished = time.perf_counter()
        step += 1
        # GPU timeline per phase; backward includes the overlapped DDP gradient all-reduce.
        for key, (a, b) in (("forward_s", (0, 1)), ("backward_s", (1, 2)), ("optimizer_s", (2, 3))):
            window.add(key, marks[a].elapsed_time(marks[b]) / 1000)

        window.add_tasks("", batch["task"], out["sample_ce"], out["sample_tokens"])
        window.add("steps", 1)
        window.add("samples", len(batch["task"]))
        window.add("mix_sec", float(batch["mix_sec"].sum()))
        window.add("enroll_sec", float(batch["enroll_sec"].sum()))
        window.add("ts_samples", int((batch["mode"] == 0).sum()))
        window.add("wait_s", computed - waited)
        window.add("compute_s", finished - computed)
        window.add("worker_s", batch["example_s"])
        window.add("decode_s", batch["decode_s"])
        window.add("augment_s", batch["augment_s"])
        window.add("grad_norm", grad_norm)
        window.add("tokens_padded", batch["input_ids"].numel())
        window.add("tokens_real", int(batch["attention_mask"].sum()))

        if step % args.logging_steps == 0 or step == args.max_steps:
            elapsed = time.perf_counter() - tick
            stats = window.reduce(device, distributed)
            if rank == 0:
                steps = stats["steps"] / world
                tokens = sum(stats.get(f"tok_{t}", 0.0) for t in TASK_NAMES)
                loss = sum(stats.get(f"ce_{t}", 0.0) for t in TASK_NAMES) / max(tokens, 1.0)
                lrs = scheduler.get_last_lr()
                append_jsonl(out_dir / "logging.jsonl", {
                    "global_step/max_steps": f"{step}/{args.max_steps}",
                    "loss": loss,
                    "grad_norm": stats["grad_norm"] / stats["steps"],
                    "learning_rate": lrs[0],
                    "memory(GiB)": torch.cuda.max_memory_allocated(device) / 2**30,
                    "train_speed(s/it)": elapsed / steps,
                })
                performance = {
                    "step": step,
                    "backbone_lr": lrs[-1],
                    "samples_per_s": stats["samples"] / elapsed,
                    # Hours of mixture (enrollment) audio trained per wall-clock hour, all ranks.
                    "mix_rtf_x": stats["mix_sec"] / elapsed,
                    "enroll_rtf_x": stats["enroll_sec"] / elapsed,
                    "batch_samples": stats["samples"] / stats["steps"],
                    "ts_share": stats["ts_samples"] / max(stats["samples"], 1),
                    "enroll_sec_per_ts": stats["enroll_sec"] / max(stats["ts_samples"], 1),
                    "data_wait_s": stats["wait_s"] / stats["steps"],
                    "compute_s": stats["compute_s"] / stats["steps"],
                    "forward_s": stats["forward_s"] / stats["steps"],
                    "backward_s": stats["backward_s"] / stats["steps"],
                    "optimizer_s": stats["optimizer_s"] / stats["steps"],
                    # LLM input tokens per GPU per second (padded = what the GPU computes).
                    "tokens_padded_per_s_gpu": stats["tokens_padded"] / elapsed / world,
                    "tokens_real_per_s_gpu": stats["tokens_real"] / elapsed / world,
                    "worker_s_per_sample": stats["worker_s"] / max(stats["samples"], 1),
                    "decode_s_per_sample": stats["decode_s"] / max(stats["samples"], 1),
                    "augment_s_per_sample": stats["augment_s"] / max(stats["samples"], 1),
                    **task_losses(stats),
                    **{f"gate_{k}": v for k, v in model.gate_values().items()},
                }
                append_jsonl(out_dir / "performance-rank0.jsonl", performance)
                print(json.dumps({"event": "train", "loss": round(loss, 4), **{
                    k: round(v, 4) for k, v in performance.items()}}), flush=True)
            torch.cuda.reset_peak_memory_stats(device)
            tick = time.perf_counter()

        if valid_set is not None and (step % args.eval_steps == 0 or step == args.max_steps):
            run_eval()
            tick = time.perf_counter()
        if args.save_steps > 0 and (step % args.save_steps == 0 or step == args.max_steps):
            save_checkpoint(out_dir, step, model, local_optimizer, scheduler, sampler, rank, world,
                            args.save_total_limit, distributed, engine)
            tick = time.perf_counter()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
