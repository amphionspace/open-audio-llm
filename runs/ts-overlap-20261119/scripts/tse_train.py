"""Train the enrollment-conditioned mask estimator on on-the-fly v12-generator scenes (train-split speakers).
Loss: MSE between power-compressed (0.3) magnitudes of mask x mixture and of the target image, over valid frames,
both scaled by the mixture RMS. AdamW 1e-3, warmup 1k, cosine; grad clip 5; fp32. Every EVAL steps rank 0 reports
SI-SDR improvement on the fixed dev scenes (dev-split speakers) and keeps the best. W&B: AmphionASR-ts-overlap-tse-mask-001.
usage: torchrun --nproc_per_node N tse_train.py OUT_DIR DEV_DIR STEPS [BATCH WORKERS EVAL]"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.distributed as dist
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tse_data
from tse_model import TSE, rms, HOP
out, dev_dir, steps = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
bs = int(sys.argv[4]) if len(sys.argv) > 4 else 8; nw = int(sys.argv[5]) if len(sys.argv) > 5 else 12
every = int(sys.argv[6]) if len(sys.argv) > 6 else 2000
ddp = "LOCAL_RANK" in os.environ
if ddp:
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else:
    rank, world = 0, 1
dev = torch.device("cuda", torch.cuda.current_device()); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
recipe, pools, splits, n_spk = tse_data.setup("train")
loader = torch.utils.data.DataLoader(tse_data.Scenes(pools, splits, seed=f"tse-train-r{rank}"), batch_size=bs, num_workers=nw,
                                     collate_fn=tse_data.collate, prefetch_factor=4, persistent_workers=True)
model = TSE().to(dev)
net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index]) if ddp else model
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
dev_rows = [json.loads(l) for l in open(dev_dir / "manifest.jsonl")]
def load_dev(r): return [torch.from_numpy(sf.read(dev_dir / "wav" / f"{r['key']}.{s}.wav", dtype="float32")[0])[None].to(dev) for s in ("enroll", "mix", "target")]
def sisdr(est, ref):
    ref = ref - ref.mean(); est = est - est.mean(); a = (est * ref).sum() / ref.pow(2).sum().clamp_min(1e-10)
    return 10 * torch.log10((a * ref).pow(2).sum() / (est - a * ref).pow(2).sum().clamp_min(1e-10))
@torch.no_grad()
def evaluate():
    model.eval(); imp = []
    for r in dev_rows:
        e, m, t = load_dev(r); s = model.extract(e, m)[0]; imp.append((sisdr(s, t[0]) - sisdr(m[0], t[0])).item())
    model.train(); return float(np.mean(imp)), float(np.median(imp))
run = None
if rank == 0:
    import wandb
    run = wandb.init(entity="1016097967-amphion", project="open-audio-llm", name="AmphionASR-ts-overlap-tse-mask-001", job_type="train",
                     config={"steps": steps, "batch_per_gpu": bs, "world": world, "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
                             "train_speakers": n_spk, "splits": len(splits), "loss": "compressed magnitude MSE (0.3)", "lr": 1e-3})
    print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M; speakers", n_spk, "; splits", len(splits), run.url, flush=True)
best, t0, it = -1e9, time.time(), iter(loader)
for step in range(1, steps + 1):
    e, m, t, lens = (x.to(dev, non_blocking=True) for x in next(it))
    M, Y = net(e, m, lens)
    T = model.spec(t / rms(m, lens))
    frames = torch.arange(M.shape[-1], device=dev)[None] < (lens // HOP + 1)[:, None]
    d = ((M * Y.abs()).clamp_min(1e-8) ** 0.3 - T.abs().clamp_min(1e-8) ** 0.3) ** 2
    loss = (d * frames[:, None]).sum() / (frames.sum() * M.shape[1])
    opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
    if rank == 0 and step % 50 == 0:
        run.log({"train/loss": loss.item(), "train/grad_norm": gn.item(), "train/lr": sched.get_last_lr()[0]}, step=step)
    if rank == 0 and (step % 500 == 0 or step <= 3):
        print(f"step {step} loss {loss.item():.5f} gn {gn.item():.2f} {time.time() - t0:.0f}s", flush=True)
    if step % every == 0 or step == steps:
        if rank == 0:
            mean_i, med_i = evaluate(); run.log({"dev/sisdr_improvement": mean_i, "dev/sisdr_improvement_median": med_i}, step=step)
            print(f"step {step} dev SI-SDRi mean {mean_i:.2f} median {med_i:.2f}", flush=True)
            torch.save({"state": model.state_dict(), "step": step, "dev": mean_i}, out / "last.pt")
            if mean_i > best: best = mean_i; torch.save({"state": model.state_dict(), "step": step, "dev": mean_i}, out / "best.pt")
        if ddp: dist.barrier()
if rank == 0:
    run.summary["best_dev_sisdr_improvement"] = best; run.finish(); print("best dev SI-SDRi", best, flush=True)
if ddp: dist.barrier(); dist.destroy_process_group()
os._exit(0)  # persistent on-the-fly workers
