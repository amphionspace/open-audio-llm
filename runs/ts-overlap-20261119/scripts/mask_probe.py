"""tasks/22 diagnostic: how well does a model's audio-tower layer 12 predict the target's mel ideal ratio mask, and does
applying the predicted mask to the mixture's mel input help recognition?
1. Freeze MODEL (11.17, or 11.17 + a ft_e2e.py state), train a linear head layer-12 -> 128-band mask per mixture token
   (as the training aux of ft_e2e.py) on on-the-fly scenes (train-split speakers), MSE to the area-pooled mel IRM.
2. Dev MSE on the fixed task-19 dev scenes, with two references: the mean mask and an all-ones (pass-through) mask.
3. On the 615 heavy-overlap samples: upsample the predicted token masks to mel frames, multiply the mixture mel power,
   re-apply the log-mel transform, and decode with 11.17 and with MODEL (greedy HF); also the unmasked decode of MODEL.
usage: mask_probe.py MODEL(base|STATE.pt) OUT_DIR STEPS SHARD NUM_SHARDS   (shard 0 trains; others wait for its head)"""
import json, math, os, random, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tse"))
import tse_data
from swift.model import get_model_processor
from swift.template import get_template
import open_audio_llm.integrations.ms_swift.register_qwen3_asr  # noqa: F401
from open_audio_llm.data.qwen3_asr import prepare_ts_clips
from open_audio_llm.tsasr.sep_token import conv_token_count
mpath, out, steps, shard, shards = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
CK = "/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365"
SUB = Path("/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/ca9aa6d0-4ecc-4cfc-b526-5322bf6c9de5/scratchpad/ovdiag/subset")
DEV = Path("/workspace/data/ts-overlap-20261119/task19-tse/dev")
SYSTEM = "Transcribe only the enrolled speaker. Output nothing if they do not speak."
def load(state):
    m, proc = get_model_processor(CK, model_type="amphion_asr_1.7b", torch_dtype=torch.bfloat16, attn_impl="eager", device_map="cuda:0")
    m.eval(); m.thinker.audio_tower.config._attn_implementation = "sdpa"
    if state != "base":
        own = dict(m.named_parameters())
        for n, v in torch.load(state, map_location="cpu")["state"].items(): own[n].data.copy_(v.to(own[n].dtype))
    for p in m.parameters(): p.requires_grad_(False)
    return m, proc
model, processor = load(mpath)
tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor; fe = processor.feature_extractor
MELF = torch.from_numpy(fe.mel_filters).float(); WIN = torch.hann_window(fe.n_fft)
template = get_template(processor, max_length=None, template_type="amphion_asr_1.7b")
def power(w):
    s = torch.stft(torch.from_numpy(w).float(), fe.n_fft, fe.hop_length, window=WIN, return_complex=True)
    return MELF.T @ (s[..., :-1].abs() ** 2)
def logmel(p):
    l = torch.clamp(p, min=1e-10).log10(); l = torch.maximum(l, l.max() - 8.0); return (l + 4.0) / 4.0
cap = []
model.thinker.audio_tower.layers[11].register_forward_hook(lambda mod, inp, o: cap.append(o[0] if isinstance(o, tuple) else o))
def encode(e, m, content=None):
    e1, m1 = prepare_ts_clips([e, m], 16000)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "<audio>"}]
    if content is not None: msgs.append({"role": "assistant", "content": content})
    template.set_mode("train" if content is not None else "transformers")
    return template.data_collator([template.encode({"messages": msgs, "audios": [e1], "mix_wav": m1})]), m1
@torch.no_grad()
def layer12(batch):
    cap.clear(); ef = int(batch["enroll_n_frames"][0]); fl = int(batch["feature_attention_mask"][0].sum())
    model.thinker.get_audio_features(batch["input_features"].to("cuda:0", torch.bfloat16), feature_attention_mask=batch["feature_attention_mask"].to("cuda:0"),
                                     enroll_n_frames=batch["enroll_n_frames"].to("cuda:0"))
    h = cap[0].reshape(-1, cap[0].shape[-1]).float(); ne, nm = conv_token_count(ef), conv_token_count(fl - ef)
    assert h.shape[0] == ne + 1 + nm
    return h[ne + 1:], ef, fl
def irm_tokens(m1, t, ef, fl, nm):
    tt = np.pad(t[:len(m1)], (0, max(0, len(m1) - len(t)))).astype(np.float32)
    pt, pr = power(tt), power(m1 - tt); irm = (pt / (pt + pr).clamp_min(1e-10))[:, : fl - ef]
    return torch.nn.functional.interpolate(irm[None], size=nm, mode="area")[0].T.cuda(), irm
head = nn.Linear(1024, 128).cuda(); hp = out / "head.pt"
if shard == 0:
    recipe, pools, splits, _ = tse_data.setup("train"); rng = random.Random("mask-probe"); opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    t0 = time.time()
    for step in range(1, steps + 1):
        _, e, m, t = tse_data.scene(rng, pools, splits)
        batch, m1 = encode(e, m); h, ef, fl = layer12(batch); tgt, _ = irm_tokens(m1, t, ef, fl, h.shape[0])
        loss = ((torch.sigmoid(head(h)) - tgt) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        for g in opt.param_groups: g["lr"] = 1e-3 * 0.5 * (1 + math.cos(math.pi * step / steps))
        if step % 500 == 0: print(f"step {step} mse {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    torch.save(head.state_dict(), hp)
    rows = [json.loads(l) for l in open(DEV / "manifest.jsonl")]; acc = {"probe": [], "mean": [], "ones": []}; allt = []
    with torch.no_grad():
        for r in rows:
            e, m, t = (sf.read(DEV / "wav" / f"{r['key']}.{s}.wav", dtype="float32")[0] for s in ("enroll", "mix", "target"))
            batch, m1 = encode(e, m); h, ef, fl = layer12(batch); tgt, _ = irm_tokens(m1, t, ef, fl, h.shape[0]); allt.append(tgt)
            acc["probe"].append(((torch.sigmoid(head(h)) - tgt) ** 2).mean().item()); acc["ones"].append(((1 - tgt) ** 2).mean().item())
        mu = torch.cat(allt).mean(0)
        acc["mean"] = [((mu - x) ** 2).mean().item() for x in allt]
    json.dump({k: float(np.mean(v)) for k, v in acc.items()}, open(out / "dev_mse.json", "w")); print("dev mse", {k: round(float(np.mean(v)), 4) for k, v in acc.items()}, flush=True)
else:
    while not (out / "dev_mse.json").exists(): time.sleep(20)
head.load_state_dict(torch.load(hp)); head.eval()
base = model if mpath == "base" else load("base")[0]
stats = json.load(open(SUB / "oracle-stats.json")); rows = [json.loads(l) for l in open(SUB / "manifest.jsonl", encoding="utf-8")]
rows = [r for r in rows if r["id"] in stats][shard::shards]
def decode(mdl, batch):
    b = {k: (v.to("cuda:0") if torch.is_tensor(v) else v) for k, v in batch.items()}; b["input_features"] = b["input_features"].to(torch.bfloat16); b.pop("labels", None)
    with torch.inference_mode():
        o = mdl.generate(**b, max_new_tokens=256, do_sample=False)
    seq = o if torch.is_tensor(o) else (o.sequences if hasattr(o, "sequences") else o[0]); seq = seq if torch.is_tensor(seq) else seq.sequences
    gen = seq[0, b["input_ids"].shape[1]:] if seq.shape[1] > b["input_ids"].shape[1] else seq[0]
    raw = tok.decode(gen, skip_special_tokens=False).split("<|im_end|>")[0]; return raw.split("<asr_text>", 1)[1] if "<asr_text>" in raw else ""
with open(out / f"decode-{shard}.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        k = r["key"]; e = sf.read(SUB / "wav" / f"{k}.enroll.wav", dtype="float32")[0]; m = sf.read(SUB / "wav" / f"{k}.mix.wav", dtype="float32")[0]
        batch, m1 = encode(e, m)
        template.set_mode("transformers")
        batch = template.data_collator([template.encode({"messages": [{"role": "system", "content": r["context"]}, {"role": "user", "content": "<audio>"}],
                                                          "audios": [prepare_ts_clips([e, m], 16000)[0]], "mix_wav": m1})])
        with torch.no_grad():
            h, ef, fl = layer12(batch); mask_tok = torch.sigmoid(head(h)).T.cpu()  # (128, nm)
        mask = torch.nn.functional.interpolate(mask_tok[None], size=fl - ef, mode="linear", align_corners=False)[0]
        Pm = power(m1)[:, : fl - ef]
        masked = batch["input_features"].clone(); masked[0, :, ef:fl] = logmel(Pm * mask).to(masked.dtype)
        mb = dict(batch); mb["input_features"] = masked
        res = {"id": r["id"], "self_plain": decode(model, batch), "self_masked": decode(model, mb), "base_masked": decode(base, mb)}
        f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
print("done", len(rows), flush=True)
