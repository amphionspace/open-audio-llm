"""Context-correction SFT from 11.17, held by a replay KL to 11.17 (runs/zh-ctx-kl-anchor-20261126, README "设计").

The cause it answers (runs/zh-ctx-grpo-k-20261122 README "TS 回退的来源"): SFT continued on a GRPO model with CE on the
recipe's labels pulls the model back toward its SFT state and undoes GRPO's TS calibration (the spelling-think control,
no context-correction data, from 11.06: v12 FA 2.70 -> 3.49*). Here CE is only on the context-correction stream, and
11.17's own task mix is replayed with token KL(11.17 || student) and no CE, as the TS line's tasks 25/28 do for the
audio tower (oal-clean-teacher runs/ts-overlap-20261119/scripts/ft_enc_replay.py). The KL is 0 at the start and only
resists drift.
- start and teacher: 11.17; the teacher is a second, frozen bf16 load of the whole model. Trainable as in the spelling-think
  control: everything except parameters.frozen_prefixes (token embedding, tied to lm_head; the audio tower's conv front
  end). MTP is not loaded; the exported checkpoints link 11.17's mtp/ (greedy outputs do not depend on it).
- each step, every rank: one batch of the CE stream (one sample) and one replay batch; loss = CE + kl_weight * KL;
  gradients averaged over the ranks by hand (as ft_enc_replay.py), clipped, AdamW in fp32 under bf16 autocast.
- steps = views x the CE batches of one epoch per rank, so each CE record is seen about `views` times; linear warmup,
  then cosine from lr to min_lr.
- rank 0 appends ms-swift shaped rows to <output_dir>/logging.jsonl (read by the launcher's W&B recorder) and the
  per-task replay KL to <output_dir>/replay-kl-by-task.jsonl.
- checkpoints at parameters.save_fractions of the steps: full model dirs under parameters.checkpoint_root (11.17's
  files, the trained tensors written over its weights, mtp/ linked to 11.17's), written by rank 0.
usage: torchrun ... -m ctx_kl_train --config EFFECTIVE.yaml --output_dir DIR   (the experiment launcher passes both)
"""
import argparse
import json
import math
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from safetensors import safe_open
from safetensors.torch import save_file
from swift.model import get_model_processor
from swift.template import get_template

import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401
from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config
from open_audio_llm.data.catalog_sampler import CatalogBatchSampler

# catalog_task codes CatalogSwiftDataset sets under objective.sample_mean (replay only; used for the per-task KL log).
TASKS = ("asr", "ts_positive", "ts_negative", "sot", "asr_hotwords", "asr_no_kl")
# Batch keys of training-only heads of the recipe; the Qwen3-ASR forward does not take them.
DROP = ("catalog_task", "twin_enroll_features", "ts_activity")
GRAD_BUCKET = 200_000_000  # floats per all-reduce, as ft_enc_replay.py
# Rank 0 writes a 4.7 GB checkpoint to quarkfs while the others wait in a barrier; NCCL's default 10 min may not cover it.
PG_TIMEOUT = timedelta(hours=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    P = yaml.safe_load(Path(args.config).read_text())["parameters"]
    out = Path(args.output_dir)
    dist.init_process_group("nccl", timeout=PG_TIMEOUT)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = f"cuda:{torch.cuda.current_device()}"
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(P["seed"])

    def load():
        loaded, proc = get_model_processor(P["init_model"], model_type="amphion_asr_1.7b", torch_dtype=torch.bfloat16,
                                           attn_impl="sdpa", device_map=dev)
        loaded.thinker.audio_tower.config._attn_implementation = "sdpa"
        return loaded, proc

    model, processor = load()
    # A second load, not copy.deepcopy(model): attach_sep_token keeps the loaded thinker's own forward and
    # get_audio_features in closures, which deepcopy shares, so a copied teacher ran the student's thinker without its
    # enroll_n_frames (attempt 001: masked_scatter assertion on the first TS replay sample; the KL would have been 0).
    teacher = load()[0].eval().requires_grad_(False)
    template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b")
    template.set_mode("train")

    frozen = tuple(P["frozen_prefixes"])
    names, params = [], []
    for name, p in model.named_parameters():
        if name.startswith(frozen):
            p.requires_grad_(False)
            continue
        p.data = p.data.float()
        p.requires_grad_(True)
        names.append(name)
        params.append(p)
    found = {prefix for prefix in frozen if any(n.startswith(prefix) for n, _ in model.named_parameters())}
    assert found == set(frozen), set(frozen) - found  # a misspelt prefix would silently train that part
    opt = torch.optim.AdamW(params, lr=P["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=P["weight_decay"])

    ce_set = CatalogSwiftDataset(read_data_config(P["ce_data"]), encode=template.encode, message_format="qwen3_asr")
    rp_set = CatalogSwiftDataset(read_data_config(P["replay_data"]), encode=template.encode, message_format="qwen3_asr")
    ce_sampler = CatalogBatchSampler(ce_set, batch_size=1, rank=rank, world_size=world, shuffle=True)
    rp_sampler = CatalogBatchSampler(rp_set, batch_size=1, rank=rank, world_size=world, shuffle=True)
    ce_sampler.set_epoch(0)
    per_epoch = torch.tensor([len(ce_sampler)], device=dev)
    dist.all_reduce(per_epoch, op=dist.ReduceOp.MIN)
    steps = math.ceil(P["views"] * int(per_epoch))
    warmup = max(1, round(P["warmup_ratio"] * steps))
    saves = sorted({max(1, round(f * steps)) for f in P["save_fractions"]})

    def lr_at(step):
        if step <= warmup:
            return P["lr"] * step / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return P["min_lr"] + (P["lr"] - P["min_lr"]) * 0.5 * (1 + math.cos(math.pi * progress))

    def batches(dataset, sampler, workers):
        epoch = 0
        while True:
            sampler.set_epoch(epoch)
            yield from torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                                                   collate_fn=template.data_collator, prefetch_factor=2)
            epoch += 1

    def to_dev(batch):
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)
        return batch

    def replay_kl(batch):
        """Mean over supervised tokens of KL(teacher || student), and per-sample means for the task log."""
        labels = batch["labels"][:, 1:]
        mask = labels != -100
        inputs = {k: v for k, v in batch.items() if k not in DROP and k != "labels"}
        with torch.no_grad():
            t_logits = teacher(**inputs, use_cache=False).logits[:, :-1][mask].float()
        s_logits = model(**inputs, use_cache=False).logits[:, :-1][mask].float()
        tok = F.kl_div(s_logits.log_softmax(-1), t_logits.log_softmax(-1), log_target=True, reduction="none").sum(-1)
        rows = mask.nonzero()[:, 0]
        per = torch.zeros(mask.shape[0], device=dev).index_add_(0, rows, tok.detach()) / mask.sum(1).clamp_min(1)
        return tok.mean(), per

    def allreduce():
        group, size = [], 0

        def flush():
            flat = torch.cat([p.grad.reshape(-1) for p in group])
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            i = 0
            for p in group:
                n = p.grad.numel()
                p.grad.copy_(flat[i:i + n].view_as(p.grad))
                i += n

        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)  # every rank must join every bucket
            group.append(p)
            size += p.grad.numel()
            if size >= GRAD_BUCKET:
                flush()
                group, size = [], 0
        if group:
            flush()

    def export(step):
        base = Path(P["export_base"])
        dst = Path(P["checkpoint_root"]) / f"checkpoint-{step}"
        tmp = dst.with_name(dst.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        trained = {n: p.detach().to(torch.bfloat16).cpu() for n, p in zip(names, params)}
        handle = safe_open(str(base / "model.safetensors"), "pt")
        keys = list(handle.keys())
        assert set(trained) <= set(keys), sorted(set(trained) - set(keys))[:5]
        assert "thinker.lm_head.weight" not in trained and "thinker.model.embed_tokens.weight" not in trained
        weights = {}
        for k in keys:
            ref = handle.get_tensor(k)
            weights[k] = (trained[k].to(ref.dtype) if k in trained else ref).contiguous()
        save_file(weights, str(tmp / "model.safetensors"), metadata=handle.metadata())
        for f in base.iterdir():
            if f.is_file() and f.name != "model.safetensors":
                shutil.copy2(f, tmp / f.name)
        (tmp / "mtp").symlink_to(base / "mtp")
        (tmp / "SOURCE.json").write_text(json.dumps(
            {"step": step, "steps": steps, "trained_tensors": len(trained), "base": str(base),
             "init_model": P["init_model"], "config": args.config,
             "mtp": "symlink to 11.17's mtp/ (not trained; greedy verification)"}, indent=1) + "\n")
        shutil.rmtree(dst, ignore_errors=True)
        tmp.rename(dst)
        print(f"saved {dst} ({len(trained)} trained tensors of {len(keys)})", flush=True)

    if rank == 0:
        print(f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable in {len(params)} tensors; CE records {len(ce_set)}, "
              f"replay records {len(rp_set)}; {int(per_epoch)} CE batches per rank per epoch, {steps} steps "
              f"({P['views']} views), warmup {warmup}, saves at {saves}", flush=True)
    model.train()
    ce_it = batches(ce_set, ce_sampler, P["ce_workers"])
    rp_it = batches(rp_set, rp_sampler, P["replay_workers"])
    acc = torch.zeros(3, device=dev)  # CE sum, KL sum, steps since the last log
    task_sum = torch.zeros(len(TASKS), device=dev)
    task_n = torch.zeros(len(TASKS), device=dev)
    t_log = time.time()
    for step in range(1, steps + 1):
        lr = lr_at(step)
        for group in opt.param_groups:
            group["lr"] = lr
        cb = to_dev(next(ce_it))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce = model(**{k: v for k, v in cb.items() if k not in DROP}, use_cache=False).loss
        ce.backward()
        rb = to_dev(next(rp_it))
        tasks = rb["catalog_task"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            kl, per = replay_kl(rb)
            (P["kl_weight"] * kl).backward()
        allreduce()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, P["max_grad_norm"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        acc += torch.stack([ce.detach().float(), kl.detach().float(), torch.ones((), device=dev)])
        task_sum.index_add_(0, tasks, per)
        task_n.index_add_(0, tasks, torch.ones_like(per))
        if step % P["log_every"] == 0 or step == steps or step <= 3:
            stats = torch.cat([acc, task_sum, task_n])
            dist.all_reduce(stats)
            n = stats[2].clamp_min(1)
            ce_mean, kl_mean = (stats[0] / n).item(), (stats[1] / n).item()
            ts, tn = stats[3:3 + len(TASKS)], stats[3 + len(TASKS):]
            if rank == 0:
                speed = (time.time() - t_log) / max(1, int(acc[2].item()))
                row = {"loss": ce_mean + P["kl_weight"] * kl_mean, "ce_asr": ce_mean, "replay_kl": kl_mean,
                       "grad_norm": grad_norm.item(), "learning_rate": lr,
                       "memory(GiB)": torch.cuda.max_memory_allocated() / 2 ** 30, "train_speed(s/it)": speed,
                       "global_step/max_steps": f"{step}/{steps}"}
                with (out / "logging.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                by_task = {name: (ts[k] / tn[k]).item() for k, name in enumerate(TASKS) if tn[k] > 0}
                with (out / "replay-kl-by-task.jsonl").open("a") as f:
                    f.write(json.dumps({"step": step, **by_task}) + "\n")
                print(f"step {step}/{steps} ce {ce_mean:.4f} kl {kl_mean:.5f} gn {grad_norm.item():.2f} lr {lr:.3g} "
                      f"mem {row['memory(GiB)']:.1f}G {speed:.2f}s/it", flush=True)
            acc.zero_()
            task_sum.zero_()
            task_n.zero_()
            t_log = time.time()
        if step in saves:
            if rank == 0:
                export(step)
            dist.barrier()
    dist.barrier()
    dist.destroy_process_group()
    os._exit(0)  # as ft_enc_replay.py: live DataLoader workers of the endless generators can hold the exit


if __name__ == "__main__":
    main()
