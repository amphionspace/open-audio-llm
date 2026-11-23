"""End-to-end TS-ASR training from 11.17 on on-the-fly v12-generator scenes (train-split speakers, KeSpeech uncapped):
all parameters (audio tower incl. the conv front end, LLM), lr 2e-5, target-text cross-entropy. 15% of samples take
another scene's enrollment (wrong speaker) and the absent output 'language None<asr_text>'. Labels: reference + '。'.
ARM 'aux' adds a linear head on the audio tower's layer-12 output predicting, per mixture token, the mel ideal ratio
mask of the target (Whisper mel power of target vs mixture - target, area-pooled to the token rate), MSE x LAMBDA;
the head is training-only. Data parallel under torchrun (all-reduce). W&B AmphionASR-ts-overlap-e2e-<arm>-<tag>.
usage: torchrun --nproc_per_node N ft_e2e.py ARM(ce|aux) OUT_DIR STEPS [MICRO_BATCH WORKERS LAMBDA TAG SAVE_EVERY]"""
import math, os, random, sys, time
from pathlib import Path
import numpy as np, torch, torch.distributed as dist, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tse"))
import tse_data
from swift.model import get_model_processor
from swift.template import get_template
import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401
from open_audio_llm.data.qwen3_asr import prepare_ts_clips
from open_audio_llm.tsasr.sep_token import conv_token_count
arm, out, steps = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
mb = int(sys.argv[4]) if len(sys.argv) > 4 else 8; nw = int(sys.argv[5]) if len(sys.argv) > 5 else 10
lam = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0; tag = sys.argv[7] if len(sys.argv) > 7 else "001"
save_every = int(sys.argv[8]) if len(sys.argv) > 8 else 4000
CK = "/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365"
SYSTEM = "Transcribe only the enrolled speaker. Output nothing if they do not speak."
ddp = "LOCAL_RANK" in os.environ
if ddp:
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else:
    rank, world = 0, 1
dev = f"cuda:{torch.cuda.current_device()}"; out.mkdir(parents=True, exist_ok=True); torch.manual_seed(0)
model, processor = get_model_processor(CK, model_type="amphion_asr_1.7b", torch_dtype=torch.bfloat16, attn_impl="sdpa", device_map=dev)
thinker = model.thinker; tower = thinker.audio_tower; tower.config._attn_implementation = "sdpa"
template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b"); template.set_mode("train")
fe = processor.feature_extractor; MELF = torch.from_numpy(fe.mel_filters).float(); WIN = torch.hann_window(fe.n_fft)
recipe, pools, splits, n_spk = tse_data.setup("train")


def mel_power(w):
    s = torch.stft(torch.from_numpy(w).float(), fe.n_fft, fe.hop_length, window=WIN, return_complex=True)
    return MELF.T @ (s[..., :-1].abs() ** 2)


class Scenes(torch.utils.data.IterableDataset):
    def __iter__(self):
        info = torch.utils.data.get_worker_info(); rng = random.Random(f"e2e-{tag}-r{rank}-w{info.id if info else 0}")
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
                content = "language Chinese<asr_text>" + text + "。"
            prev = (prev + [own])[-8:]
            e1, m1 = prepare_ts_clips([e, m], 16000)
            enc = template.encode({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "<audio>"},
                                                {"role": "assistant", "content": content}], "audios": [e1], "mix_wav": m1})
            irm = None
            if arm == "aux":
                tt = np.pad(t[:len(m1)], (0, max(0, len(m1) - len(t)))).astype(np.float32)
                pt, pr = mel_power(tt), mel_power(m1 - tt)
                irm = (pt / (pt + pr).clamp_min(1e-10)).float()  # (128, frames), 0 where the target is absent
            yield enc, irm


def collate(items):
    return template.data_collator([e for e, _ in items]), [i for _, i in items]


for p in model.parameters(): p.requires_grad_(False)
params, seen = [], set()
for p in list(tower.parameters()) + list(thinker.model.parameters()) + list(thinker.lm_head.parameters()):
    if id(p) not in seen:
        seen.add(id(p)); p.data = p.data.float(); p.requires_grad_(True); params.append(p)
head = nn.Linear(1024, 128).to(dev) if arm == "aux" else None
trainable = params + (list(head.parameters()) if head is not None else [])
opt = torch.optim.AdamW([{"params": params, "lr": 2e-5}] + ([{"params": list(head.parameters()), "lr": 1e-3}] if head is not None else []),
                        weight_decay=0.0, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
cap = []
if head is not None:
    tower.layers[11].register_forward_hook(lambda mod, inp, o: cap.append(o[0] if isinstance(o, tuple) else o))
loader = torch.utils.data.DataLoader(Scenes(), batch_size=mb, num_workers=nw, collate_fn=collate, prefetch_factor=2, persistent_workers=True)
run = None
if rank == 0:
    import wandb
    run = wandb.init(entity="1016097967-amphion", project="open-audio-llm", name=f"AmphionASR-ts-overlap-e2e-{arm}-{tag}", job_type="train",
                     config={"arm": arm, "lr": 2e-5, "steps": steps, "micro_batch": mb, "world": world, "aux_lambda": lam if head is not None else 0,
                             "aux": "layer-12 linear head -> mel IRM per mixture token (area-pooled)" if head is not None else None,
                             "start": "11.17 model-afea989da22b77adc0ce9365", "train_speakers": n_spk, "neg_prob": 0.15,
                             "trainable_params_M": sum(p.numel() for p in params) / 1e6})
    print(f"arm {arm}: {sum(p.numel() for p in params) / 1e6:.1f}M trainable; speakers {n_spk}; {run.url}", flush=True)
model.train(); t0 = time.time(); it = iter(loader)
for step in range(1, steps + 1):
    batch, irms = next(it); cap.clear()
    ef = batch["enroll_n_frames"].tolist(); fl = batch["feature_attention_mask"].sum(-1).tolist()
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch["input_features"] = batch["input_features"].to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ce = model(**batch, use_cache=False).loss
        aux = torch.zeros((), device=dev)
        if head is not None:
            terms = []
            for h, irm, e_f, f_l in zip(cap, irms, ef, fl):
                h = h.reshape(-1, h.shape[-1]); ne = conv_token_count(e_f); nm = conv_token_count(f_l - e_f)
                assert h.shape[0] == ne + 1 + nm, (h.shape, ne, nm)
                tgt = torch.nn.functional.interpolate(irm[:, : f_l - e_f][None].to(dev), size=nm, mode="area")[0].T  # (nm, 128)
                pred = torch.sigmoid(head(h[ne + 1:].float()))
                terms.append(((pred - tgt) ** 2).mean())
            aux = torch.stack(terms).mean()
        loss = ce + lam * aux
    loss.backward()
    if ddp:
        for p in trainable:
            if p.grad is not None: dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if rank == 0 and step % 20 == 0:
        run.log({"train/loss": loss.item(), "train/ce": ce.item(), "train/aux_mask_mse": aux.item(), "train/grad_norm": gn.item(),
                 "train/lr": sched.get_last_lr()[0]}, step=step)
    if rank == 0 and (step % 200 == 0 or step <= 3):
        print(f"step {step} ce {ce.item():.4f} aux {aux.item():.4f} gn {gn.item():.2f} {time.time() - t0:.0f}s", flush=True)
    if (step % save_every == 0 or step == steps) and rank == 0:
        state = {n: p.detach().to(torch.bfloat16).cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"arm": arm, "step": step, "state": state}, out / f"step{step}.pt"); print("saved", step, flush=True)
    if (step % save_every == 0 or step == steps) and ddp:
        dist.barrier()
if rank == 0:
    run.finish()
if ddp: dist.barrier(); dist.destroy_process_group()
os._exit(0)
