"""On-policy GRPO for TS-ASR, one process per GPU (torchrun).

Per step each rank takes ``--prompts`` samples, generates ``--G`` completions
per sample with a bf16 copy of the policy, scores them (``reward.py``) and
takes one AdamW step on the fp32 policy under bf16 autocast. One update per
batch keeps the PPO ratio at 1, so the loss is group-baselined REINFORCE:
``-sum_i A_i * log pi(y_i) / LOSS_TOKENS``, averaged over every completion on
every rank. There is no reference model or KL term; the learning rate and the
number of steps bound the drift.

The bf16 copy exists because generation in bf16 is several times faster than
fp32 under autocast, and an fp32 master is needed because AdamW updates at
1e-6 are below bf16 resolution for most weights.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
import torch.distributed as dist

from . import model as ts_model
from .reward import DEFAULT_MISS_REWARD, reward
from .text import parse_output, units

# Same frozen parameters as the AmphionASR SFT recipes.
FROZEN = ("thinker.model.embed_tokens", "thinker.audio_tower.conv2d1", "thinker.audio_tower.conv2d2",
          "thinker.audio_tower.conv2d3", "thinker.audio_tower.conv_out")
# Constant token normaliser for the summed sequence log-prob (Dr. GRPO style), so
# long and short completions are not rescaled per sequence.
LOSS_TOKENS = 32.0
# Generation cap per batch: about 1.5 tokens per reference unit plus room for the
# "language X<asr_text>" header; anything longer is wrong regardless.
CAP_TOKENS_PER_UNIT, CAP_EXTRA_TOKENS = 1.5, 24


def read_samples(path: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        for key in ("enroll_wav", "mix_wav"):
            if not Path(row[key]).is_absolute():
                row[key] = str(path.parent / row[key])
        rows.append(row)
    return rows


def category(sample: dict) -> str:
    base = sample["label"].rsplit("-", 1)[0]
    return f"{base}/{'present' if sample['present'] else 'absent'}"


def save_checkpoint(gen_model, model_dir: Path, out: Path, step: int) -> None:
    """Write a drop-in replacement of the source ``model/`` directory."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    target, tmp = out / f"checkpoint-{step}", out / f".tmp-checkpoint-{step}"
    shutil.copytree(model_dir, tmp, ignore=shutil.ignore_patterns("*.safetensors", "*.safetensors.index.json"))
    keys, meta = set(), None
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            keys |= set(h.keys())
            meta = meta or h.metadata()
    state = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in gen_model.state_dict().items()}
    if set(state) != keys:
        raise RuntimeError(f"checkpoint keys differ from {model_dir}: "
                           f"extra {sorted(set(state) - keys)[:5]}, missing {sorted(keys - set(state))[:5]}")
    save_file(state, str(tmp / "model.safetensors"), metadata=meta)
    tmp.rename(target)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", type=Path, required=True, help="model/ directory of a delivered AmphionASR package")
    p.add_argument("--train", type=Path, required=True, help="samples.jsonl written by tsasr_grpo.prepare")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--prompts", type=int, default=4, help="samples per rank per step")
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--std_floor", type=float, default=0.1,
                   help="lower bound of the group reward std in the advantage, so 1-unit CER noise is not amplified")
    p.add_argument("--micro_batch", type=int, default=4, help="completions per forward/backward")
    p.add_argument("--save_steps", type=int, default=60)
    p.add_argument("--miss_reward", type=float, default=DEFAULT_MISS_REWARD)
    p.add_argument("--system", default=ts_model.DEFAULT_SYSTEM,
                   help="TS system prompt; use the one the checkpoint was trained with")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    policy, processor = ts_model.load(str(args.model), torch.float32, device)
    policy.train()
    for name, param in policy.named_parameters():
        param.requires_grad_(not name.startswith(FROZEN))
    gen_model, _ = ts_model.load(str(args.model), torch.bfloat16, device)
    gen_model.eval().requires_grad_(False)
    trainable = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    gen_params = dict(gen_model.named_parameters())
    opt = torch.optim.AdamW([p for _, p in trainable], lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=0.0)
    warmup = max(1, round(args.steps * args.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))

    pool = read_samples(args.train)
    random.Random(args.seed).shuffle(pool)
    mine = pool[rank::world]
    cursor = 0
    if rank == 0:
        (out / "grpo-args.json").write_text(json.dumps(vars(args), default=str, indent=1))
        train_log = (out / "logging.jsonl").open("a")
        grpo_log = (out / "grpo.jsonl").open("a")

    for step in range(1, args.steps + 1):
        started = time.time()
        batch = [mine[(cursor + i) % len(mine)] for i in range(args.prompts)]
        cursor += args.prompts

        def work():
            stats = defaultdict(float)
            prompts = []
            for s in batch:
                enroll, _ = sf.read(s["enroll_wav"], dtype="float32")
                mix, _ = sf.read(s["mix_wav"], dtype="float32")
                prompts.append(ts_model.build_inputs(processor, enroll, mix, args.system))
            cap = min(args.max_new_tokens, max(
                int(CAP_TOKENS_PER_UNIT * len(units(s["target"], s["language"]))) + CAP_EXTRA_TOKENS for s in batch))
            completions = ts_model.generate(gen_model, prompts, args.num_generations, args.temperature, cap,
                                            device, torch.bfloat16, seed=args.seed * 100003 + step * 1009 + rank)
            torch.cuda.empty_cache()
            groups = []
            for s, prompt, comps in zip(batch, prompts, completions):
                texts = [parse_output(processor.tokenizer.decode(c)) for c in comps]
                r = torch.tensor([reward(s["target"], t, s["language"], args.miss_reward) for t in texts])
                adv = (r - r.mean()) / max(float(r.std(unbiased=False)), args.std_floor)
                cat = category(s)
                stats[f"{cat}/n"] += 1
                stats[f"{cat}/reward"] += float(r.mean())
                stats[f"{cat}/reward_varies"] += float(r.std() > 0)
                stats[f"{cat}/{'miss' if s['present'] else 'false_alarm'}"] += \
                    sum((not units(t, s["language"])) == s["present"] for t in texts) / len(texts)
                groups.append((prompt, comps, adv))
            gen_seconds = time.time() - started
            total = world * args.prompts * args.num_generations
            loss_sum, n_tokens = 0.0, 0
            for prompt, comps, adv in groups:
                if float(adv.abs().sum()) == 0:  # constant reward: no gradient
                    continue
                for i in range(0, len(comps), args.micro_batch):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lp, nt = ts_model.completion_logprobs(policy, prompt, comps[i:i + args.micro_batch],
                                                              device, torch.float32)
                    loss = -(adv[i:i + args.micro_batch].to(device) * lp).sum() / LOSS_TOKENS / total
                    loss.backward()
                    loss_sum += float(loss.detach())
                    n_tokens += int(nt.sum())
            return stats, gen_seconds, loss_sum, n_tokens

        # Shared GPUs: when any rank runs out of memory, every rank drops the step together.
        try:
            stats, gen_seconds, loss_sum, n_tokens = work()
            ok = 1.0
        except torch.OutOfMemoryError:
            ok = 0.0
        flag = torch.tensor([ok], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if flag.item() == 0:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        else:
            for _, param in trainable:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                dist.all_reduce(param.grad)  # the loss is already divided by the global completion count
            grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            with torch.no_grad():
                for name, param in trainable:
                    gen_params[name].copy_(param.to(torch.bfloat16))

            names = [None] * world
            dist.all_gather_object(names, sorted(stats))
            union = sorted({k for ks in names for k in ks})
            vec = torch.tensor([stats.get(k, 0.0) for k in union] + [loss_sum, n_tokens],
                               device=device, dtype=torch.float64)
            dist.all_reduce(vec)
        if rank == 0:
            if flag.item() == 0:
                print(f"step {step}: skipped, out of memory on a rank", flush=True)
                grpo_log.write(json.dumps({"step": step, "skipped_oom": 1}) + "\n")
                grpo_log.flush()
            else:
                agg = dict(zip(union, vec[:-2].tolist()))
                elapsed = time.time() - started
                train_log.write(json.dumps({
                    "global_step/max_steps": f"{step}/{args.steps}", "loss": float(vec[-2]),
                    "grad_norm": grad_norm, "learning_rate": sched.get_last_lr()[0],
                    "memory(GiB)": torch.cuda.max_memory_allocated(device) / 2**30,
                    "train_speed(s/it)": elapsed}) + "\n")
                train_log.flush()
                row = {"step": step, "generation_seconds": gen_seconds, "step_seconds": elapsed,
                       "completion_tokens": int(vec[-1])}
                for key in union:
                    cat, metric = key.rsplit("/", 1)
                    if metric != "n":
                        row[key] = agg[key] / agg[f"{cat}/n"]
                    else:
                        row[key] = int(agg[key])
                grpo_log.write(json.dumps(row) + "\n")
                grpo_log.flush()
                print(json.dumps(row), flush=True)
            if step % args.save_steps == 0 or step == args.steps:
                save_checkpoint(gen_model, args.model, out, step)
                print(f"saved checkpoint-{step}", flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
