"""v5 = v4 warm-started from a v4 checkpoint (INIT) with target-selection training: crops are 50% centred on target
activity, 25% uniform (may hold no target speech), 25% with another scene's enrollment (wrong speaker, target = silence).
A crop whose target energy is below 1e-3 of the mixture's is a silent target: its output-0 loss is the output energy
relative to the mixture in dB (floored at -30 dB) instead of -SI-SDR. Learning rates halved (body 5e-5, FiLM 5e-4).
v4 doc: v3 plus FiLM inside the mask net (after layers 6, 12, 18; zero-initialised), warm-started from a v3 checkpoint
(INIT env), conditioning layers at 10x the learning rate. v3 doc: fine-tune the CAM++-conditioned MossFormer2 extractor (tse_mf2.py) on on-the-fly v12-generator scenes
(train-split speakers). 4 s crops centred on a random sample where the target is active; inputs at unit RMS.
Loss: -SI-SDR(output 0, target image) - 0.5 x SI-SDR(output 1, mixture - target). AdamW 1e-4, warmup 500, cosine,
grad clip 5, bf16 autocast. Every EVAL steps rank 0 reports windowed SI-SDR improvement on the fixed dev scenes and
keeps the best. W&B: AmphionASR-ts-overlap-tse-mossformer2-selection-005.
usage: torchrun --nproc_per_node N tse_train3.py OUT_DIR DEV_DIR STEPS [BATCH WORKERS EVAL]"""
import json, math, os, random, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.distributed as dist
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tse_data
from tse_mf2 import MF2TSE, WIN
from spk_embed import SpeakerEmbed
out, dev_dir, steps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
bs = int(sys.argv[4]) if len(sys.argv) > 4 else 2; nw = int(sys.argv[5]) if len(sys.argv) > 5 else 12
every = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
ddp = "LOCAL_RANK" in os.environ
if ddp:
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else:
    rank, world = 0, 1
dev = torch.device("cuda", torch.cuda.current_device()); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(0)


class Crops(tse_data.Scenes):
    def __iter__(self):
        info = torch.utils.data.get_worker_info(); rng = random.Random(f"{self.seed}:{info.id if info else 0}")
        prev = []  # recent enrollments of other scenes (other speakers) for the wrong-enrollment crops
        while True:
            _, e, m, t = tse_data.scene(rng, self.pools, self.splits)
            mode = rng.random()
            active = np.flatnonzero(np.abs(t) > 0.05 * np.abs(t).max())
            if len(m) > WIN:
                if mode < 0.5 and len(active):
                    c = int(active[rng.randrange(len(active))]); s = min(max(0, c - WIN // 2), len(m) - WIN)
                else:
                    s = rng.randrange(len(m) - WIN + 1)
                m, t = m[s:s + WIN], t[s:s + WIN]
            own = e
            if mode >= 0.75 and prev:
                e = prev[rng.randrange(len(prev))]; t = np.zeros_like(t)
            prev = (prev + [own])[-8:]
            yield e, m, t


recipe, pools, splits, n_spk = tse_data.setup("train")
loader = torch.utils.data.DataLoader(Crops(pools, splits, seed=f"tse3-train-r{rank}"), batch_size=bs, num_workers=nw,
                                     collate_fn=tse_data.collate, prefetch_factor=4, persistent_workers=True)
model = MF2TSE(pretrained=False, deep=True)
init = torch.load(os.environ["INIT"], map_location="cpu"); model.load_state_dict(init["state"], strict=True)
init_step = init["step"]; model = model.to(dev); spk_model = SpeakerEmbed().to(dev)
net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index], find_unused_parameters=False) if ddp else model
cond = [p for n, p in model.named_parameters() if n.startswith("film")]
body = [p for n, p in model.named_parameters() if not n.startswith("film")]
opt = torch.optim.AdamW([{"params": body, "lr": 5e-5}, {"params": cond, "lr": 5e-4}], weight_decay=1e-2)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
dev_rows = [json.loads(l) for l in open(dev_dir / "manifest.jsonl")]
def load_dev(r): return [torch.from_numpy(sf.read(dev_dir / "wav" / f"{r['key']}.{s}.wav", dtype="float32")[0])[None].to(dev) for s in ("enroll", "mix", "target")]
def sisdr1(est, ref):
    ref = ref - ref.mean(); est = est - est.mean(); a = (est * ref).sum() / ref.pow(2).sum().clamp_min(1e-10)
    return 10 * torch.log10((a * ref).pow(2).sum() / (est - a * ref).pow(2).sum().clamp_min(1e-10))
@torch.no_grad()
def evaluate():
    model.eval(); imp = []
    for r in dev_rows:
        e, m, t = load_dev(r); s = model.extract(m, spk_model(e))[0]; imp.append((sisdr1(s, t[0]) - sisdr1(m[0], t[0])).item())
    model.train(); return float(np.mean(imp)), float(np.median(imp))
def sisdr(est, ref, valid, lens):
    est, ref = est * valid, ref * valid
    est = est - (est.sum(-1, keepdim=True) / lens[:, None]) * valid; ref = ref - (ref.sum(-1, keepdim=True) / lens[:, None]) * valid
    a = (est * ref).sum(-1, keepdim=True) / ref.pow(2).sum(-1, keepdim=True).clamp_min(1e-8)
    return 10 * torch.log10((a * ref).pow(2).sum(-1).clamp_min(1e-8) / (est - a * ref).pow(2).sum(-1).clamp_min(1e-8))
run = None
if rank == 0:
    import wandb
    run = wandb.init(entity="1016097967-amphion", project="open-audio-llm", name="AmphionASR-ts-overlap-tse-mossformer2-selection-005", job_type="train",
                     config={"steps": steps, "batch_per_gpu": bs, "world": world, "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
                             "train_speakers": n_spk, "splits": len(splits), "init": "MossFormer2_SS_16K (ClearerVoice 6b3774d)",
                             "loss": "-SI-SDR(target) - 0.5 SI-SDR(rest)", "lr": {"body": 5e-5, "film": 5e-4}, "crops": "50% target-centred, 25% uniform, 25% wrong enrollment (silent target)", "crop_s": 4,
                             "init_ckpt": os.environ["INIT"], "init_step": init_step, "deep_film_after_layers": [6, 12, 18]})
    ident, _ = evaluate(); run.log({"dev/sisdr_improvement": ident}, step=0)
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M; dev SI-SDRi at warm start (v3 step", init_step, ")", round(ident, 2), run.url, flush=True)
if ddp: dist.barrier()
best, t0, it = -1e9, time.time(), iter(loader)
for step in range(1, steps + 1):
    e, m, t, lens = (x.to(dev, non_blocking=True) for x in next(it))
    valid = (torch.arange(m.shape[-1], device=dev)[None] < lens[:, None]).float()
    k = ((m.pow(2) * valid).sum(-1, keepdim=True) / lens[:, None]).clamp_min(1e-10).sqrt(); m, t = m / k, t / k
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o0, o1 = net(m * valid, spk_model(e))
    si_t = sisdr(o0.float(), t, valid, lens); si_r = sisdr(o1.float(), m - t, valid, lens)
    e_t = (t.pow(2) * valid).sum(-1); e_m = (m.pow(2) * valid).sum(-1).clamp_min(1e-8)
    silent = e_t < 1e-3 * e_m
    supp = (10 * torch.log10((o0.float().pow(2) * valid).sum(-1) / e_m + 1e-3)).clamp_min(-30.0)
    loss_t = torch.where(silent, supp, -si_t)
    loss = loss_t.mean() - 0.5 * si_r.mean()
    opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
    if rank == 0 and step % 50 == 0:
        run.log({"train/loss": loss.item(), "train/silent_share": silent.float().mean().item(),
                 "train/sisdr_target_present": (si_t[~silent].mean().item() if (~silent).any() else float("nan")),
                 "train/suppression_db_silent": (supp[silent].mean().item() if silent.any() else float("nan")), "train/sisdr_target": si_t.mean().item(), "train/sisdr_rest": si_r.mean().item(),
                 "train/grad_norm": gn.item(), "train/lr": sched.get_last_lr()[0]}, step=step)
    if rank == 0 and (step % 250 == 0 or step <= 3):
        print(f"step {step} loss {loss.item():.3f} si_target {si_t.mean().item():.2f} gn {gn.item():.2f} {time.time() - t0:.0f}s", flush=True)
    if step % every == 0 or step == steps:
        if rank == 0:
            mean_i, med_i = evaluate(); run.log({"dev/sisdr_improvement": mean_i, "dev/sisdr_improvement_median": med_i}, step=step)
            print(f"step {step} dev SI-SDRi mean {mean_i:.2f} median {med_i:.2f}", flush=True)
            torch.save({"state": model.state_dict(), "step": step, "dev": mean_i, "arch": "mf2", "deep": True}, out / "last.pt")
            if mean_i > best: best = mean_i; torch.save({"state": model.state_dict(), "step": step, "dev": mean_i, "arch": "mf2", "deep": True}, out / "best.pt")
        if ddp: dist.barrier()
if rank == 0:
    run.summary["best_dev_sisdr_improvement"] = best; run.finish(); print("best dev SI-SDRi", best, flush=True)
if ddp: dist.barrier(); dist.destroy_process_group()
os._exit(0)
