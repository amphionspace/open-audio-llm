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
tasks/28 (MEETING_PLAN = a meeting_train_plan.py dir): plus CE_meeting on one micro-batch per step of real meetings sent
the way speaker-enroll sends them (Meetings below); without it the script is task 25's.
tasks/29 (each off when unset, then the script is task 28's):
- SIM_NEG = p: with probability p a simulated sample is a v12-generator scene without the target (NEG_FAMILIES below,
  CE on "language None"), and the short target-present scenes of the nearshort recipe join the present splits;
- SPK_EMB = train_speaker_embeddings.py output: the speaker cosine the generator's hard (similar-voice) negatives need;
- ASR_REPLAY_CONFIG = a data config: a second replay batch per step from it, same token KL and weight.
usage: [POOLS=train_final2 HEAD_LAYERS=5,11 WANDB_READY_FILE=F MEETING_PLAN=DIR SIM_NEG=p SPK_EMB=F.npz
        ASR_REPLAY_CONFIG=A.yaml ASR_REPLAY_WORKERS=n] REPLAY_CONFIG=Y.yaml torchrun ...
       ft_enc_replay.py OUT_DIR STEPS [MICRO_BATCH WORKERS LAMBDA TAG SAVE_EVERY KL_WEIGHT REPLAY_WORKERS]   (oal-train python)"""
import copy, gzip, json, math, os, random, sys, time
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
# tasks/29: the CE above had one negative only (a swapped enrollment, below), so the target-absent shapes v12 counts as
# false alarms that it lacked (no speech at all; one short phrase by someone else, often a similar voice) were held
# only by the replay KL. SIM_NEG draws them from the generator's own target-absent splits (recipe v4-20261022, train
# pools), three families with equal probability; splits with a recorded registration (enroll_file: the business
# testers' recordings that v12 uses) are left out. NEARSHORT: the nearshort recipe's 1.5-3 s scenes (adc
# scripts/ts_benchmark/recipe-nearshort.yaml v3-20261022): the two absent ones go to "short", their target-present
# counterparts (the enrolled speaker says one short phrase, alone or with another short phrase) to the present splits.
SIM_NEG = float(os.environ.get("SIM_NEG", "0"))
NEARSHORT = [
    {"group": "nearshort", "name": "short-rand-neg", "kind": "neg", "k": 1, "duration": [1.5, 3.0], "interferer_utts": "single_short", "interferer_start": [0.1, 0.6]},
    {"group": "nearshort", "name": "short-hard-neg", "kind": "neg", "k": 1, "duration": [1.5, 3.0], "interferer_utts": "single_short", "interferer_start": [0.1, 0.6], "hard": True},
    {"group": "nearshort", "name": "solo-short-pos", "kind": "pos", "schedule": "short", "k": 0, "duration": [1.5, 3.0]},
    {"group": "nearshort", "name": "short-k1-pos", "kind": "pos", "schedule": "short", "k": 1, "duration": [1.5, 3.0], "interferer_utts": "single_short", "interferer_start": [0.1, 0.6]}]
NEG_FAMILIES = ("nonspeech", "short", "absent")
neg_splits = {f: [] for f in NEG_FAMILIES}
if SIM_NEG:
    for s in tse_data.build.expand_splits(recipe) + tse_data.build.expand_splits({**recipe, "splits": NEARSHORT}):
        kind = s["params"].get("kind")
        if s["params"].get("enroll_file"):
            continue
        if kind == "nonspeech":
            neg_splits["nonspeech"].append(s)
        elif kind == "neg":
            neg_splits["short" if s["name"] in ("abs-short", "short-rand-neg", "short-hard-neg") else "absent"].append(s)
        elif s["name"] in ("solo-short-pos", "short-k1-pos"):
            splits = splits + [s]
# A hard negative's enrollment is one of the 5 pool speakers closest to an interferer. v12 drew them from its test pool;
# among the 26,585 train speakers the 5 closest are closer (chosen cosine median 0.66, p90 0.73, against v12's
# short-hard-neg 0.56 / 0.66 and abs-hard 0.53 / 0.62). A random HARD_POOL-speaker candidate pool per scene gives v12's
# difficulty (median 0.56, p10 0.48, p90 0.64; simulated on the same embeddings), so the CE does not teach rejecting
# voices closer than any the benchmark or the 11.17 recipe asks to tell apart.
HARD_POOL = 600
SPK_EMB = os.environ.get("SPK_EMB")
similarity = None
if SPK_EMB:
    _z = np.load(SPK_EMB); _vec = dict(zip(_z["keys"].tolist(), _z["vectors"]))
    similarity = lambda a, b: float(_vec[a] @ _vec[b])
hard_candidates = sorted(pools["mixed"]["speakers"])
ASR_REPLAY_CONFIG = os.environ.get("ASR_REPLAY_CONFIG")
REPLAY_CONFIG = os.environ["REPLAY_CONFIG"]
MEETING_PLAN = os.environ.get("MEETING_PLAN")
# tasks/28 meeting stream. Weights: the recipe's weights of these ts_real_train sources (alimeeting 1.5 + 2.0, aishell4
# 1.8 + 2.0, ramc 1.7). NEG_PROB: speaker-enroll sends every segment to every registered person, about half absent in
# the downstream meeting (task 26). SIM_TEMP: a negative span is drawn with p ~ exp(sim / SIM_TEMP), sim = ECAPA cosine
# with the closest talker of the sent audio (task 27's impostor_sim), so similar voices dominate. NEAR_PROB (AliMeeting):
# the enrollment comes from the person's headset, for targets and non-targets alike, so the channel says nothing about
# the answer. NOISE_*: one meeting-room noise recording on both enrollment and segment at one gain (one recorder), the
# segment NOISE_SNR dB above it, the enrollment at least ENROLL_SNR_MIN dB (speaker-enroll only keeps such pieces).
# CROP_PROB: send one utterance of the window's target (up to CROP_PAD s around it, not touching its other utterances),
# the shape of a speaker-enroll segment; the same crop is sent to targets and non-targets. A non-target is a participant
# silent in the audio sent (so in a crop also one who talks elsewhere in the window), with TALK_MARGIN s of slack for
# the annotation's boundaries.
M_WEIGHTS = {"alimeeting": 3.5, "aishell4": 3.8, "ramc": 1.7}
NEG_PROB, SIM_TEMP, NEAR_PROB, CROP_PROB, CROP_PAD = 0.5, 0.1, 0.35, 0.5, 0.5
TALK_MARGIN = 0.3
NOISE_PROB, NOISE_SNR, ENROLL_SNR_MIN = 0.35, (3.0, 15.0), 6.0
NOISE = ("meeting_room_noise", "v2-20261014", "train")  # the recipe's interference source


def mel_power(w):
    s = torch.stft(torch.from_numpy(w).float(), fe.n_fft, fe.hop_length, window=WIN, return_complex=True)
    return MELF.T @ (s[..., :-1].abs() ** 2)


def load_noise(kind, path):
    """The generator's noise loader (build.render_job): `quiet` is generated."""
    if kind == "quiet":
        return np.random.default_rng(0).standard_normal(16000 * 30).astype(np.float32)
    return tse_data.build._cached_noise(path)


class Scenes(torch.utils.data.IterableDataset):
    """Yields (encoding, (mel IRM, family, hard cosine or nan)); family is present / swap / one of NEG_FAMILIES."""

    def __iter__(self):
        info = torch.utils.data.get_worker_info(); rng = random.Random(f"encreplay-{tag}-r{rank}-w{info.id if info else 0}")
        prev = []
        while True:
            fam = rng.choice(NEG_FAMILIES) if SIM_NEG and rng.random() < SIM_NEG else "present"
            split = rng.choice(neg_splits[fam] if fam != "present" else splits)
            view = pools
            if split["params"].get("hard"):  # candidates: HARD_POOL random speakers of the union pool
                mixed = pools["mixed"]; sub = rng.sample(hard_candidates, HARD_POOL)
                view = {**pools, "mixed": {"speakers": {s: mixed["speakers"][s] for s in sub},
                                           "short": {s: mixed["short"][s] for s in sub if s in mixed["short"]}}}
            try:
                plan = tse_data.mixer.plan_scene(rng, split["params"], view, similarity)
            except (tse_data.mixer.PlanError, IndexError, ValueError):
                continue
            if fam != "present":  # no target: the generator's own rendering (mixture effects included), empty mask
                load = lambda utt: tse_data.build.read_audio(utt.path)
                load_rir = lambda p: tse_data.sf.read(p, dtype="float32")[0]
                m = tse_data.mixer.render_mixture(plan, load, load_rir, load_noise).astype(np.float32)
                e = tse_data.mixer.render_enrollment(plan, load, load_rir, load_noise)
                c = 5 * 16000; e = np.pad(e[:c], (0, max(0, c - len(e)))).astype(np.float32)
                e1, m1 = prepare_ts_clips([e, m], 16000)
                enc = template.encode({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "<audio>"},
                                                    {"role": "assistant", "content": "language None<asr_text>"}], "audios": [e1], "mix_wav": m1})
                n_fr = mel_power(m1).shape[1]
                yield enc, (torch.zeros(128, n_fr), fam, float(plan.get("hard_cosine", math.nan)))
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
            # (128, frames), 0 where the target is absent
            yield enc, ((pt / (pt + pr).clamp_min(1e-10)).float(), "swap" if neg else "present", math.nan)


def collate(items):
    return template.data_collator([e for e, _ in items]), [i for _, i in items]


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def read_span(path, start, dur):
    w, sr = tse_data.sf.read(path, start=int(round(start * 16000)), frames=int(round(dur * 16000)), dtype="float32", always_2d=True)
    assert sr == 16000, path
    return w[:, 0]


class Meetings(torch.utils.data.IterableDataset):
    """Real meeting windows (meeting_train_plan.py) sent to their target or to a similar-voiced other participant of
    the same recording; yields (encoding, facts) with facts = (negative, sim, near, noise, crop) for the logs."""

    def __init__(self, plan_dir):
        plan = json.loads((Path(plan_dir) / "plan.json").read_text())
        self.windows = plan["windows"]; self.corpora = [c for c in M_WEIGHTS if self.windows.get(c)]
        self.span = {}  # corpus -> span index -> (start, duration, speaker)
        for c in self.corpora:
            for rec in plan["recordings"][c].values():
                for spk, items in rec["spans"].items():
                    for s, d, i in items: self.span.setdefault(c, {})[i] = (s, d, spk)
        self.near = {(c, p): r["near"] for c in self.corpora for p, r in plan["recordings"][c].items()}
        import yaml
        from audio_data_contract import load_catalog, resolve_split
        catalog = load_catalog(yaml.safe_load(Path(REPLAY_CONFIG).read_text())["catalog"])
        roots = json.loads(Path("/workspace/data/registry/roots.json").read_text())
        self.noises = []
        for p in resolve_split(catalog, *NOISE, "recordings", roots):
            for line in gzip.open(p, "rt"):
                r = json.loads(line); self.noises.append((r["sources"][0]["source"], r["num_samples"]))

    def noise(self, rng, n):
        """n samples of one noise recording from a random offset, wrapping around its end."""
        path, total = self.noises[rng.randrange(len(self.noises))]
        parts, pos = [], rng.randrange(total)
        while n > 0:
            x = tse_data.sf.read(path, start=pos, frames=min(n, total - pos), dtype="float32", always_2d=True)[0][:, 0]
            if not len(x):
                raise ValueError(f"{path}: no samples at {pos} of {total}")
            parts.append(x); n -= len(x); pos = 0
        return np.concatenate(parts)

    def __iter__(self):
        info = torch.utils.data.get_worker_info(); rng = random.Random(f"meeting-{tag}-r{rank}-w{info.id if info else 0}")
        weights = [M_WEIGHTS[c] for c in self.corpora]
        while True:
            c = rng.choices(self.corpora, weights)[0]; w = rng.choice(self.windows[c])
            lo, hi = w["start"], w["start"] + w["duration"]; text = w["text"]
            crop = rng.random() < CROP_PROB
            if crop:  # one utterance of the target, not touching its other utterances
                own = [u for u in w["utts"] if u[0] == w["target"]]; k = rng.randrange(len(own))
                us, ud = w["start"] + own[k][1], own[k][2]
                prev_end = max([w["start"] + u[1] + u[2] for u in own[:k]], default=lo)
                next_start = min([w["start"] + u[1] for u in own[k + 1:]], default=hi)
                lo = max(lo, prev_end, us - rng.uniform(0, CROP_PAD)); hi = min(hi, next_start, us + ud + rng.uniform(0, CROP_PAD))
                if hi - lo < 0.3:
                    continue
                text = w["pieces"][k].lstrip("，。、；：！？,.;:!? ").rstrip("，、；：,;:")
                if not text:
                    continue
                if text[-1] not in "。？！?!…":
                    text += "。"
            # who talks in the audio sent (annotation boundaries are not exact: TALK_MARGIN s either side)
            talkers = sorted({u[0] for u in w["utts"] if w["start"] + u[1] < hi + TALK_MARGIN and w["start"] + u[1] + u[2] > lo - TALK_MARGIN})
            if w["target"] not in talkers:
                continue
            cand = [x for x in w["neg"] if x[0] not in talkers]
            neg = rng.random() < NEG_PROB and bool(cand)
            if neg:
                cols = [w["talkers"].index(t) for t in talkers]
                sims = np.array([max(x[2][j] for j in cols) for x in cand])
                p = np.exp((sims - sims.max()) / SIM_TEMP); j = rng.choices(range(len(sims)), p.tolist())[0]
                spk, si, sim = cand[j][0], cand[j][1], float(sims[j])
            else:
                spk, si, sim = w["target"], rng.choice(w["pos"]), float("nan")
            s, d, _ = self.span[c][si]
            ed = min(d, 5.0); e = read_span(w["path"], s, ed); m = read_span(w["path"], lo, hi - lo)
            near = c == "alimeeting" and spk in self.near[(c, w["path"])] and rng.random() < NEAR_PROB
            if near:
                h = read_span(self.near[(c, w["path"])][spk], s, ed)[: len(e)]
                if rms(h) < 1e-6:
                    continue
                e = h * (rms(e) / rms(h))
            noisy = rng.random() < NOISE_PROB
            if noisy:
                ne, nm = self.noise(rng, len(e)), self.noise(rng, len(m))
                g = rms(m) / max(rms(nm), 1e-6) / 10 ** (rng.uniform(*NOISE_SNR) / 20)
                ge = min(g, rms(e) / max(rms(ne), 1e-6) / 10 ** (ENROLL_SNR_MIN / 20))
                m = m + g * nm; e = e + ge * ne
            peak = max(np.abs(e).max(), np.abs(m).max())
            if peak > 0.99:
                e, m = e * (0.99 / peak), m * (0.99 / peak)
            cn = 5 * 16000; e = np.pad(e[:cn], (0, max(0, cn - len(e)))).astype(np.float32); m = m.astype(np.float32)
            content = "language None<asr_text>" if neg else "language Chinese<asr_text>" + text
            e1, m1 = prepare_ts_clips([e, m], 16000)
            enc = template.encode({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "<audio>"},
                                                {"role": "assistant", "content": content}], "audios": [e1], "mix_wav": m1})
            yield enc, (bool(neg), sim, near, noisy, crop)


replay_set = CatalogSwiftDataset(read_data_config(REPLAY_CONFIG), encode=template.encode, message_format="qwen3_asr")
replay_sampler = CatalogBatchSampler(replay_set, batch_size=1, rank=rank, world_size=world, shuffle=True)
asr_set = CatalogSwiftDataset(read_data_config(ASR_REPLAY_CONFIG), encode=template.encode, message_format="qwen3_asr") if ASR_REPLAY_CONFIG else None
asr_sampler = CatalogBatchSampler(asr_set, batch_size=1, rank=rank, world_size=world, shuffle=True) if asr_set else None


def replay_batches(dataset=replay_set, sampler=replay_sampler, workers=rnw):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        yield from torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
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
meetings = Meetings(MEETING_PLAN) if MEETING_PLAN else None
mloader = iter(torch.utils.data.DataLoader(meetings, batch_size=mb, num_workers=nw, collate_fn=collate, prefetch_factor=2,
                                           persistent_workers=True)) if meetings else None
m_facts = []  # (negative, sim, near, noise, crop) of the meeting requests since the last log, this rank
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
                             "replay_records": len(replay_set), "trainable_params_M": sum(p.numel() for p in params) / 1e6,
                             **({"sim_neg": SIM_NEG, "sim_neg_families": {f: [s["name"] for s in v] for f, v in neg_splits.items()},
                                 "sim_present_splits": len(splits), "speaker_embeddings": SPK_EMB} if SIM_NEG else {}),
                             **({"asr_replay_config": ASR_REPLAY_CONFIG, "asr_replay_records": len(asr_set),
                                 "asr_replay_loss": "token KL(11.17 || student), one more batch per step, weight kl_weight"} if asr_set else {}),
                             **({"meeting_plan": MEETING_PLAN, "meeting_loss": "target-text CE, weight 1", "meeting_weights": M_WEIGHTS,
                                 "meeting_windows": {c: len(meetings.windows[c]) for c in meetings.corpora},
                                 "meeting_neg_prob": NEG_PROB, "meeting_sim_temp": SIM_TEMP, "meeting_near_prob": NEAR_PROB,
                                 "meeting_crop_prob": CROP_PROB, "meeting_crop_pad": CROP_PAD, "meeting_noise_prob": NOISE_PROB,
                                 "meeting_noise_snr_db": NOISE_SNR, "meeting_enroll_snr_min_db": ENROLL_SNR_MIN,
                                 "meeting_noise": "@".join(NOISE[:2]) + ":" + NOISE[2]} if meetings else {})})
    print(f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable; speakers {n_spk}; replay {len(replay_set)}; {run.url}", flush=True)
model.train(); teacher_tower.eval(); t0 = time.time(); it = iter(loader); rit = replay_batches()
ait = replay_batches(asr_set, asr_sampler, int(os.environ.get("ASR_REPLAY_WORKERS", "4"))) if asr_set else None
SIM_FAMS = ("present", "swap") + NEG_FAMILIES
s_facts = []  # (family, hard cosine or nan) of the simulated samples since the last log, this rank
task_sum = torch.zeros(len(TASKS), device=dev); task_n = torch.zeros(len(TASKS), device=dev)
for step in range(1, steps + 1):
    batch, sinfo = next(it); irms = [x[0] for x in sinfo]; s_facts += [x[1:] for x in sinfo]
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
    if mloader is not None:
        mbatch, facts = next(mloader); m_facts += facts
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce_m = model(**to_dev(mbatch), use_cache=False).loss
            ce_m.backward()
        for v in cap.values(): v.clear()
    rb = to_dev(next(rit)); tasks = rb["catalog_task"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        kl, per = replay_kl(rb)
        (kl_w * kl).backward()
    for v in cap.values(): v.clear()
    if ait is not None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            kl_a, _ = replay_kl(to_dev(next(ait)))
            (kl_w * kl_a).backward()
        for v in cap.values(): v.clear()
    task_sum.index_add_(0, tasks, per); task_n.index_add_(0, tasks, torch.ones_like(per))
    if ddp:
        allreduce(trainable)
    gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 20 == 0:
        stats = torch.stack([task_sum, task_n])
        if ddp: dist.all_reduce(stats)
        mlog = {}
        if mloader is not None:  # what the meeting stream actually sent since the last log, over all ranks
            f = np.array([(n, n and s >= 0.4, nr, nz, cr) for n, s, nr, nz, cr in m_facts], dtype=np.float64)
            fs = torch.tensor(np.concatenate([f.sum(0), [len(f)]]), device=dev)
            if ddp: dist.all_reduce(fs)
            fs = (fs[:5] / fs[5].clamp_min(1)).tolist(); m_facts.clear()
            mlog = {"train/ce_meeting": ce_m.item(), "train/meeting_neg_frac": fs[0], "train/meeting_neg_sim_ge_0.4_frac": fs[1],
                    "train/meeting_near_frac": fs[2], "train/meeting_noise_frac": fs[3], "train/meeting_crop_frac": fs[4]}
            if rank == 0 and (step % 200 == 0 or os.environ.get("SMOKE")):
                print("meeting stream", {k.split("/")[1]: round(v, 3) for k, v in mlog.items()}, flush=True)
        if SIM_NEG:  # what the simulated stream actually sent since the last log, over all ranks
            hc = [h for _, h in s_facts if h == h]
            fs = torch.tensor([sum(f == k for f, _ in s_facts) for k in SIM_FAMS] + [len(hc), len(s_facts), sum(hc)],
                              device=dev, dtype=torch.float64)
            if ddp: dist.all_reduce(fs)
            hard_cos = (fs[-1] / fs[-3].clamp_min(1)).item(); fs = (fs[:-2] / fs[-2].clamp_min(1)).tolist(); s_facts.clear()
            mlog.update({f"train/sim_{k}_frac": fs[j] for j, k in enumerate(SIM_FAMS)})
            mlog.update({"train/sim_hard_frac": fs[-1], "train/sim_hard_cosine_mean": hard_cos})
            if rank == 0 and (step % 200 == 0 or os.environ.get("SMOKE")):
                print("sim stream", {k: round(fs[j], 3) for j, k in enumerate(SIM_FAMS + ("hard",))}, "hard cosine", round(hard_cos, 3), flush=True)
        else:
            s_facts.clear()
        if ait is not None:
            mlog["train/replay_kl_asr_stream"] = kl_a.item()
        if rank == 0:
            run.log({**mlog, "train/ce_sim": ce.item(), "train/aux_mask_mse": aux.item(), "train/replay_kl": kl.item(), "train/grad_norm": gn.item(),
                     "train/lr": sched.get_last_lr()[0], **{f"train/aux_mask_mse_l{li + 1}": v.item() for li, v in per_layer.items()},
                     **{f"train/replay_kl_{name}": (stats[0, k] / stats[1, k]).item() for k, name in enumerate(TASKS) if stats[1, k] > 0}},
                    step=step)
        task_sum.zero_(); task_n.zero_()
    if rank == 0 and (step % 200 == 0 or step <= 3 or (os.environ.get("SMOKE") and step % 10 == 0)):
        print(f"step {step} ce {ce.item():.4f} {'ce_m %.4f ' % ce_m.item() if mloader is not None else ''}aux {aux.item():.4f} kl {kl.item():.5f} {'kl_a %.5f ' % kl_a.item() if ait is not None else ''}gn {gn.item():.2f} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G {time.time() - t0:.0f}s", flush=True)
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
