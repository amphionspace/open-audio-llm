"""Component capacity test: from 11.17, fine-tune only the audio tower (enc), only the LLM + lm_head (dec), or both,
on generated v12-like scenes (train speakers), target-text cross-entropy (ms-swift template, as SFT). Each sample is the
mixture (p 0.7) or the clean version (target + noise, p 0.3) with its 5 s enrollment and the TS system prompt.
Trainable parameters in fp32, forward in bf16 autocast, AdamW, warmup 200 + cosine, grad clip 1. Saves the trainable
parameters. W&B: AmphionASR-ts-overlap-component-ft-<arm>-001.
Data parallel under torchrun (gradients averaged by all-reduce; each rank draws its own samples).
usage: [torchrun --nproc_per_node N] ft_component.py ARM MANIFEST WAV_DIR OUT_PT STEPS [MICRO_BATCH ACCUM]"""
import json, math, os, random, sys, time
import numpy as np, soundfile as sf, torch
from swift.model import get_model_processor
from swift.template import get_template
import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401
from open_audio_llm.data.qwen3_asr import prepare_ts_clips
arm, manifest, wav_dir, out_pt, steps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
mb = int(sys.argv[6]) if len(sys.argv) > 6 else 8; accum = int(sys.argv[7]) if len(sys.argv) > 7 else 2
CK = "/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365"
SYSTEM = "Transcribe only the enrolled speaker. Output nothing if they do not speak."
LR = {"enc": {"tower": 2e-5}, "dec": {"llm": 1e-5}, "both": {"tower": 2e-5, "llm": 1e-5}}[arm]
import torch.distributed as dist
ddp = "LOCAL_RANK" in os.environ
if ddp:
    dist.init_process_group("nccl"); rank = dist.get_rank(); world = dist.get_world_size(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else:
    rank, world = 0, 1
dev = f"cuda:{torch.cuda.current_device()}"
random.seed(rank); torch.manual_seed(0)
model, processor = get_model_processor(CK, model_type="amphion_asr_1.7b", torch_dtype=torch.bfloat16, attn_impl="sdpa", device_map=dev)
thinker = model.thinker; tower = thinker.audio_tower; tower.config._attn_implementation = "sdpa"
template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b"); template.set_mode("train")
for p in model.parameters(): p.requires_grad_(False)
groups = {"tower": list(tower.parameters()), "llm": list(thinker.model.parameters()) + list(thinker.lm_head.parameters())}
params, seen = [], set()
for g, lr in LR.items():
    uniq = []
    for p in groups[g]:  # lm_head is tied to the input embedding
        if id(p) not in seen: seen.add(id(p)); uniq.append(p)
    for p in uniq: p.data = p.data.float(); p.requires_grad_(True)
    params.append({"params": uniq, "lr": lr})
n_train = sum(p.numel() for g in params for p in g["params"])
opt = torch.optim.AdamW(params, weight_decay=0.0, betas=(0.9, 0.98))
trainable = [p for g in params for p in g["params"]]
def allreduce_grads():
    if not ddp: return
    for p in trainable:
        if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
rows = [json.loads(l) for l in open(manifest, encoding="utf-8")]
def example(r):
    cond = "mix" if random.random() < 0.7 else "clean"
    e = sf.read(f"{wav_dir}/{r['key']}.enroll.wav", dtype="float32")[0]; a = sf.read(f"{wav_dir}/{r['key']}.{cond}.wav", dtype="float32")[0]
    e1, a1 = prepare_ts_clips([e, a], 16000)
    return template.encode({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "<audio>"},
                                         {"role": "assistant", "content": "language Chinese<asr_text>" + r.get("label", r["reference"])}],
                            "audios": [e1], "mix_wav": a1})
import wandb
if rank != 0: os.environ["WANDB_MODE"] = "disabled"
run = wandb.init(entity="1016097967-amphion", project="open-audio-llm", name=f"AmphionASR-ts-overlap-component-ft-{arm}-001",
                 job_type="train", config={"arm": arm, "lr": LR, "steps": steps, "micro_batch": mb, "accum": accum,
                                           "trainable_params": n_train, "start": "11.17 model-afea989da22b77adc0ce9365",
                                           "data": manifest, "mix_prob": 0.7, "world_size": world})
print(f"arm {arm}: {n_train / 1e6:.1f}M trainable, run {run.url}", flush=True)
model.train(); t0 = time.time()
for step in range(1, steps + 1):
    total = 0.0
    for _ in range(accum):
        batch = template.data_collator([example(r) for r in random.sample(rows, mb)])
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, use_cache=False).loss / accum
        loss.backward(); total += loss.item()
    allreduce_grads()
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 20 == 0 and rank == 0:
        run.log({"train/loss": total, "train/grad_norm": gn.item(), "train/lr": sched.get_last_lr()[0]}, step=step)
    if rank == 0 and (step % 200 == 0 or step <= 3):
        print(f"step {step} loss {total:.4f} gn {gn.item():.3f} {time.time() - t0:.0f}s", flush=True)
if rank == 0:
    state = {n: p.detach().to(torch.bfloat16).cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"arm": arm, "state": state}, out_pt); run.summary["saved_params"] = len(state); run.finish()
    print("saved", out_pt, len(state), flush=True)
if ddp: dist.barrier(); dist.destroy_process_group()
