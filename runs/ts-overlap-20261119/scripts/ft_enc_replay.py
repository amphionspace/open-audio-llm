"""tasks/25: ft_e2e2.py with the LLM frozen and 11.17's own task mix replayed by distillation.

Task 23 showed (encoder-only hybrid) that the overlap gain of simulator training lives in the audio tower and the
forgetting in the LLM; the tower still drifted on data the simulator does not cover (FA, plain ASR, English TS). So:
- trainable: the audio tower incl. the conv front end and the projection (lr 2e-5, cosine, as task 23); LLM frozen;
- simulated scenes (tse_data POOLS, default train_final2 = final-v2 text): target-text CE plus the mel-IRM aux heads
  (HEAD_LAYERS, LAMBDA), exactly task 23's objective;
- replay (REPLAY_CONFIG = a catalog data config of 11.17's task mix, final text): token KL(teacher || student) on every
  supervised token, no CE. The teacher is 11.17 itself (its audio tower, frozen, with the shared frozen LLM), so the
  replay loss is 0 at the start and only resists drift. CE on that mix drifted even with the LLM frozen (task 08).
loss = CE_sim + LAMBDA * aux_sim + KL_WEIGHT * KL_replay, one simulated micro-batch and one replay batch per step.
usage: [POOLS=train_final2 HEAD_LAYERS=5,11 WANDB_READY_FILE=F] REPLAY_CONFIG=Y.yaml torchrun ... ft_enc_replay.py
       OUT_DIR STEPS [MICRO_BATCH WORKERS LAMBDA TAG SAVE_EVERY KL_WEIGHT REPLAY_WORKERS]   (oal-train python)"""
import copy, math, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.distributed as dist, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tse"))
import tse_data
from swift.model import get_model_processor
from swift.template import get_template
import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401
from open_audio_llm.data.qwen3_asr import prepare_ts_clips
from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset, read_data_config
from open_audio_llm.data.catalog_sampler import CatalogBatchSampler
from open_audio_llm.tsasr.sep_token import conv_token_count
out, steps = Path(sys.argv[1]), int(sys.argv[2])
a = sys.argv[3:] + [None] * 8
mb, nw = int(a[0] or 4), int(a[1] or 5); lam = float(a[2] or 4.0); tag = a[3] or "001"
save_every, kl_w, rnw = int(a[4] or 10000), float(a[5] or 1.0), int(a[6] or 6)
CK = "/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365"
SYSTEM = "Transcribe only the enrolled speaker. Output nothing if they do not speak."
# catalog_task codes set by CatalogSwiftDataset under objective.sample_mean (logged per task only).
TASKS = ("asr", "ts_positive", "ts_negative", "sot", "asr_hotwords", "asr_no_kl")
# Batch keys of training-only heads of the formal recipe; Qwen3-ASR forward does not take them.
DROP = ("catalog_task", "twin_enroll_features", "ts_activity", "labels")
ddp = "LOCAL_RANK" in os.environ
if ddp:
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else:
    rank, world = 0, 1
dev = f"cuda:{torch.cuda.current_device()}"; out.mkdir(parents=True, exist_ok=True); torch.manual_seed(0)
model, processor = get_model_processor(CK, model_type="amphion_asr_1.7b", torch_dtype=torch.bfloat16, attn_impl="sdpa", device_map=dev)
HEAD_LAYERS = [int(x) for x in os.environ.get("HEAD_LAYERS", "5,11").split(",")]
thinker = model.thinker; tower = thinker.audio_tower; tower.config._attn_implementation = "sdpa"
teacher_tower = copy.deepcopy(tower).eval().requires_grad_(False)  # before any hook is registered on the student
template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b"); template.set_mode("train")
fe = processor.feature_extractor; MELF = torch.from_numpy(fe.mel_filters).float(); WIN = torch.hann_window(fe.n_fft)
POOLS = os.environ.get("POOLS", "train_final2")
recipe, pools, splits, n_spk = tse_data.setup(POOLS)
REPLAY_CONFIG = os.environ["REPLAY_CONFIG"]


def mel_power(w):
    s = torch.stft(torch.from_numpy(w).float(), fe.n_fft, fe.hop_length, window=WIN, return_complex=True)
    return MELF.T @ (s[..., :-1].abs() ** 2)


class Scenes(torch.utils.data.IterableDataset):
    def __iter__(self):
        info = torch.utils.data.get_worker_info(); rng = random.Random(f"encreplay-{tag}-r{rank}-w{info.id if info else 0}")
        prev = []
        while True:
            split = rng.choice(splits)
            try:
                plan = tse_data.mixer.plan_scene(rng, split["params"], pools, None)
            except (tse_data.mixer.PlanError, IndexError, ValueError):
                continue
            target = next((t for t in plan["tracks"] if t.role == "target"), None)
            if plan["params"].get("mix_effect") or target is None:
                continue
            t, o, n = tse_data.stems(plan, tse_data.build, tse_data.mixer)
            load = lambda utt: tse_data.build.read_audio(utt.path)
            e = tse_data.mixer.render_enrollment(plan, load, lambda p: tse_data.sf.read(p, dtype="float32")[0],
                                                 lambda kind, path: tse_data.build._cached_noise(path))
            c = 5 * 16000; e = np.pad(e[:c], (0, max(0, c - len(e)))).astype(np.float32)
            m = (t + o + n).astype(np.float32); text = tse_data.build._label_text(target.items)
            own = e; neg = rng.random() < 0.15 and prev
            if neg:
                e = prev[rng.randrange(len(prev))]; t = np.zeros_like(t); content = "language None<asr_text>"
            else:
                content = "language Chinese<asr_text>" + text  # final text is punctuated
            prev = (prev + [own])[-8:]
            e1, m1 = prepare_ts_clips([e, m], 16000)
            enc = template.encode({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "<audio>"},
                                                {"role": "assistant", "content": content}], "audios": [e1], "mix_wav": m1})
            tt = np.pad(t[:len(m1)], (0, max(0, len(m1) - len(t)))).astype(np.float32)
            pt, pr = mel_power(tt), mel_power(m1 - tt)
            yield enc, (pt / (pt + pr).clamp_min(1e-10)).float()  # (128, frames), 0 where the target is absent


def collate(items):
    return template.data_collator([e for e, _ in items]), [i for _, i in items]


replay_set = CatalogSwiftDataset(read_data_config(REPLAY_CONFIG), encode=template.encode, message_format="qwen3_asr")
replay_sampler = CatalogBatchSampler(replay_set, batch_size=1, rank=rank, world_size=world, shuffle=True)


def replay_batches():
    epoch = 0
    while True:
        replay_sampler.set_epoch(epoch)
        yield from torch.utils.data.DataLoader(replay_set, batch_sampler=replay_sampler, num_workers=rnw,
                                               collate_fn=template.data_collator, prefetch_factor=2)
        epoch += 1


for p in model.parameters(): p.requires_grad_(False)
params = []
for p in tower.parameters():
    p.data = p.data.float(); p.requires_grad_(True); params.append(p)
heads = nn.ModuleList([nn.Linear(1024, 128) for _ in HEAD_LAYERS]).to(dev)
trainable = params + list(heads.parameters())
opt = torch.optim.AdamW([{"params": params, "lr": 2e-5}, {"params": list(heads.parameters()), "lr": 1e-3}],
                        weight_decay=0.0, betas=(0.9, 0.98))


def allreduce(ps, bucket=200_000_000):
    group, size = [], 0
    def flush():
        flat = torch.cat([p.grad.reshape(-1) for p in group]); dist.all_reduce(flat, op=dist.ReduceOp.AVG); i = 0
        for p in group:
            n = p.grad.numel(); p.grad.copy_(flat[i:i + n].view_as(p.grad)); i += n
    for p in ps:
        if p.grad is None: p.grad = torch.zeros_like(p)  # every rank must join every bucket
        group.append(p); size += p.grad.numel()
        if size >= bucket: flush(); group, size = [], 0
    if group: flush()


def to_dev(batch):
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch["input_features"] = batch["input_features"].to(torch.bfloat16)
    return batch


def replay_kl(batch):
    """Mean over supervised tokens of KL(teacher || student); per-sample means for the task metrics."""
    labels = batch["labels"][:, 1:]; mask = labels != -100
    inputs = {k: v for k, v in batch.items() if k not in DROP}
    with torch.no_grad():
        thinker.audio_tower = teacher_tower
        try:
            t_logits = model(**inputs, use_cache=False).logits[:, :-1][mask].float()
        finally:
            thinker.audio_tower = tower
    s_logits = model(**inputs, use_cache=False).logits[:, :-1][mask].float()
    tok = F.kl_div(s_logits.log_softmax(-1), t_logits.log_softmax(-1), log_target=True, reduction="none").sum(-1)
    rows = mask.nonzero()[:, 0]
    per = torch.zeros(mask.shape[0], device=dev).index_add_(0, rows, tok.detach()) / mask.sum(1).clamp_min(1)
    return tok.mean(), per


sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
cap = {i: [] for i in HEAD_LAYERS}
for _i in HEAD_LAYERS:
    tower.layers[_i].register_forward_hook(lambda mod, inp, o, _i=_i: cap[_i].append(o[0] if isinstance(o, tuple) else o))
loader = torch.utils.data.DataLoader(Scenes(), batch_size=mb, num_workers=nw, collate_fn=collate, prefetch_factor=2, persistent_workers=True)
run = None
if rank == 0:
    import wandb
    ready = os.environ.get("WANDB_READY_FILE")
    if ready:
        for _ in range(180):
            if os.path.exists(ready): break
            time.sleep(10)
        else:
            os.environ["WANDB_MODE"] = "offline"; print("W&B forward not ready: logging offline", flush=True)
    run = wandb.init(entity="1016097967-amphion", project="open-audio-llm", name=f"AmphionASR-ts-overlap-encreplay-{tag}", job_type="train",
                     config={"lr": 2e-5, "steps": steps, "micro_batch": mb, "world": world, "aux_lambda": lam, "head_layers": HEAD_LAYERS,
                             "aux": "linear heads -> mel IRM per mixture token (area-pooled), simulated scenes only",
                             "start": "11.17 model-afea989da22b77adc0ce9365", "trainable": "audio tower (LLM frozen)",
                             "train_speakers": n_spk, "pools": POOLS, "neg_prob": 0.15, "replay_config": REPLAY_CONFIG,
                             "replay_loss": "token KL(11.17 || student), no CE", "kl_weight": kl_w,
                             "replay_records": len(replay_set), "trainable_params_M": sum(p.numel() for p in params) / 1e6})
    print(f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable; speakers {n_spk}; replay {len(replay_set)}; {run.url}", flush=True)
model.train(); teacher_tower.eval(); t0 = time.time(); it = iter(loader); rit = replay_batches()
task_sum = torch.zeros(len(TASKS), device=dev); task_n = torch.zeros(len(TASKS), device=dev)
for step in range(1, steps + 1):
    batch, irms = next(it)
    for v in cap.values(): v.clear()
    ef = batch["enroll_n_frames"].tolist(); fl = batch["feature_attention_mask"].sum(-1).tolist()
    batch = to_dev(batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ce = model(**batch, use_cache=False).loss
        per_layer, terms = {}, []
        for j, li in enumerate(HEAD_LAYERS):
            lt = []
            for h, irm, e_f, f_l in zip(cap[li], irms, ef, fl):
                h = h.reshape(-1, h.shape[-1]); ne = conv_token_count(e_f); nm = conv_token_count(f_l - e_f)
                assert h.shape[0] == ne + 1 + nm, (h.shape, ne, nm)
                tgt = F.interpolate(irm[:, : f_l - e_f][None].to(dev), size=nm, mode="area")[0].T  # (nm, 128)
                lt.append(((torch.sigmoid(heads[j](h[ne + 1:].float())) - tgt) ** 2).mean())
            per_layer[li] = torch.stack(lt).mean(); terms.append(per_layer[li])
        aux = torch.stack(terms).mean()
        (ce + lam * aux).backward()
    for v in cap.values(): v.clear()
    rb = to_dev(next(rit)); tasks = rb["catalog_task"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        kl, per = replay_kl(rb)
        (kl_w * kl).backward()
    for v in cap.values(): v.clear()
    task_sum.index_add_(0, tasks, per); task_n.index_add_(0, tasks, torch.ones_like(per))
    if ddp:
        allreduce(trainable)
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 20 == 0:
        stats = torch.stack([task_sum, task_n])
        if ddp: dist.all_reduce(stats)
        if rank == 0:
            run.log({"train/ce_sim": ce.item(), "train/aux_mask_mse": aux.item(), "train/replay_kl": kl.item(), "train/grad_norm": gn.item(),
                     "train/lr": sched.get_last_lr()[0], **{f"train/aux_mask_mse_l{li + 1}": v.item() for li, v in per_layer.items()},
                     **{f"train/replay_kl_{name}": (stats[0, k] / stats[1, k]).item() for k, name in enumerate(TASKS) if stats[1, k] > 0}},
                    step=step)
        task_sum.zero_(); task_n.zero_()
    if rank == 0 and (step % 200 == 0 or step <= 3 or (os.environ.get("SMOKE") and step % 10 == 0)):
        print(f"step {step} ce {ce.item():.4f} aux {aux.item():.4f} kl {kl.item():.5f} gn {gn.item():.2f} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G {time.time() - t0:.0f}s", flush=True)
    if (step % save_every == 0 or step == steps) and rank == 0:
        state = {n: p.detach().to(torch.bfloat16).cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"arm": "encreplay", "step": step, "init_step": 0, "state": state,
                    "heads": {li: heads[j].state_dict() for j, li in enumerate(HEAD_LAYERS)}}, out / f"step{step}.pt")
        print("saved", step, flush=True)
    if (step % save_every == 0 or step == steps) and ddp:
        dist.barrier()
if rank == 0:
    run.finish()
if ddp: dist.barrier(); dist.destroy_process_group()
os._exit(0)
