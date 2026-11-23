"""tasks/23 = ft_e2e.py scaled up: warm start from a ft_e2e.py state (INIT env), mask heads on several tower layers
(HEAD_LAYERS env, 0-based layer indices, default 5,11 = after layers 6 and 12), saved with every checkpoint; gradients
all-reduced in flat ~200M-element buckets (multi-node); rank 0 waits for WANDB_READY_FILE (the W&B forward letting the
nodes through) for up to 30 min, else logs offline. Everything else as ft_e2e.py:
End-to-end TS-ASR training from 11.17 on on-the-fly v12-generator scenes (train-split speakers, KeSpeech uncapped):
all parameters (audio tower incl. the conv front end, LLM), lr 2e-5, target-text cross-entropy. 15% of samples take
another scene's enrollment (wrong speaker) and the absent output 'language None<asr_text>'. Labels: reference + '。'.
ARM 'aux' adds a linear head on the audio tower's layer-12 output predicting, per mixture token, the mel ideal ratio
mask of the target (Whisper mel power of target vs mixture - target, area-pooled to the token rate), MSE x LAMBDA;
the head is training-only. Data parallel under torchrun (all-reduce). W&B AmphionASR-ts-overlap-e2e-<arm>-<tag>.
usage: [INIT=STATE.pt HEAD_LAYERS=5,11 WANDB_READY_FILE=F] torchrun ... ft_e2e2.py ARM(ce|aux) OUT_DIR STEPS [MICRO_BATCH WORKERS LAMBDA TAG SAVE_EVERY]"""
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
INIT = os.environ.get("INIT"); init_step = 0
if INIT:
    _own = dict(model.named_parameters()); _ck = torch.load(INIT, map_location="cpu")
    for _n, _v in _ck["state"].items(): _own[_n].data.copy_(_v.to(_own[_n].dtype))
    init_step = _ck.get("step", 0); del _ck
HEAD_LAYERS = [int(x) for x in os.environ.get("HEAD_LAYERS", "5,11").split(",")]
thinker = model.thinker; tower = thinker.audio_tower; tower.config._attn_implementation = "sdpa"
template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b"); template.set_mode("train")
fe = processor.feature_extractor; MELF = torch.from_numpy(fe.mel_filters).float(); WIN = torch.hann_window(fe.n_fft)
recipe, pools, splits, n_spk = tse_data.setup("train")


def mel_power(w):
    s = torch.stft(torch.from_numpy(w).float(), fe.n_fft, fe.hop_length, window=WIN, return_complex=True)
    return MELF.T @ (s[..., :-1].abs() ** 2)


class Scenes(torch.utils.data.IterableDataset):
    def __iter__(self):
        info = torch.utils.data.get_worker_info(); rng = random.Random(f"e2e2-{tag}-r{rank}-w{info.id if info else 0}")
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
heads = nn.ModuleList([nn.Linear(1024, 128) for _ in HEAD_LAYERS]).to(dev) if arm == "aux" else None
head = heads
trainable = params + (list(heads.parameters()) if heads is not None else [])
opt = torch.optim.AdamW([{"params": params, "lr": 2e-5}] + ([{"params": list(heads.parameters()), "lr": 1e-3}] if heads is not None else []),
                        weight_decay=0.0, betas=(0.9, 0.98))
def allreduce(ps, bucket=200_000_000):
    group, size = [], 0
    def flush():
        flat = torch.cat([p.grad.reshape(-1) for p in group]); dist.all_reduce(flat, op=dist.ReduceOp.AVG); i = 0
        for p in group:
            n = p.grad.numel(); p.grad.copy_(flat[i:i + n].view_as(p.grad)); i += n
    for p in ps:
        if p.grad is None: continue
        group.append(p); size += p.grad.numel()
        if size >= bucket: flush(); group, size = [], 0
    if group: flush()
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
cap = {i: [] for i in HEAD_LAYERS}
if heads is not None:
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
    run = wandb.init(entity="1016097967-amphion", project="open-audio-llm", name=f"AmphionASR-ts-overlap-e2e2-{arm}-{tag}", job_type="train",
                     config={"arm": arm, "lr": 2e-5, "steps": steps, "micro_batch": mb, "world": world, "aux_lambda": lam if head is not None else 0,
                             "aux": "layer-12 linear head -> mel IRM per mixture token (area-pooled)" if head is not None else None,
                             "start": INIT or "11.17 model-afea989da22b77adc0ce9365", "init_step": init_step, "head_layers": HEAD_LAYERS,
                             "train_speakers": n_spk, "neg_prob": 0.15,
                             "trainable_params_M": sum(p.numel() for p in params) / 1e6})
    print(f"arm {arm}: {sum(p.numel() for p in params) / 1e6:.1f}M trainable; speakers {n_spk}; {run.url}", flush=True)
model.train(); t0 = time.time(); it = iter(loader)
for step in range(1, steps + 1):
    batch, irms = next(it)
    for v in cap.values(): v.clear()
    ef = batch["enroll_n_frames"].tolist(); fl = batch["feature_attention_mask"].sum(-1).tolist()
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch["input_features"] = batch["input_features"].to(torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ce = model(**batch, use_cache=False).loss
        aux = torch.zeros((), device=dev)
        per_layer = {}
        if heads is not None:
            terms = []
            for j, li in enumerate(HEAD_LAYERS):
                lt = []
                for h, irm, e_f, f_l in zip(cap[li], irms, ef, fl):
                    h = h.reshape(-1, h.shape[-1]); ne = conv_token_count(e_f); nm = conv_token_count(f_l - e_f)
                    assert h.shape[0] == ne + 1 + nm, (h.shape, ne, nm)
                    tgt = torch.nn.functional.interpolate(irm[:, : f_l - e_f][None].to(dev), size=nm, mode="area")[0].T  # (nm, 128)
                    lt.append(((torch.sigmoid(heads[j](h[ne + 1:].float())) - tgt) ** 2).mean())
                per_layer[li] = torch.stack(lt).mean(); terms.append(per_layer[li])
            aux = torch.stack(terms).mean()
        loss = ce + lam * aux
    loss.backward()
    if ddp:
        allreduce(trainable)
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if rank == 0 and step % 20 == 0:
        run.log({"train/loss": loss.item(), "train/ce": ce.item(), "train/aux_mask_mse": aux.item(), "train/grad_norm": gn.item(),
                 "train/lr": sched.get_last_lr()[0], **{f"train/aux_mask_mse_l{li + 1}": v.item() for li, v in per_layer.items()}}, step=step)
    if rank == 0 and (step % 200 == 0 or step <= 3):
        print(f"step {step} ce {ce.item():.4f} aux {aux.item():.4f} gn {gn.item():.2f} {time.time() - t0:.0f}s", flush=True)
    if (step % save_every == 0 or step == steps) and rank == 0:
        state = {n: p.detach().to(torch.bfloat16).cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save({"arm": arm, "step": step, "init_step": init_step, "state": state,
                    "heads": ({li: heads[j].state_dict() for j, li in enumerate(HEAD_LAYERS)} if heads is not None else None)}, out / f"step{step}.pt")
        print("saved", step, flush=True)
    if (step % save_every == 0 or step == steps) and ddp:
        dist.barrier()
if rank == 0:
    run.finish()
if ddp: dist.barrier(); dist.destroy_process_group()
os._exit(0)
