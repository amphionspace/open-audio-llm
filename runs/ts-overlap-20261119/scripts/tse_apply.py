"""Apply the trained mask estimator to [enrollment, mixture] of a test set: writes OUT_DIR/{key}.tse.wav, OUT_DIR/{key}.remix.wav (extracted + 0.15 x mixture) and a
symlink OUT_DIR/{key}.mix.wav to the mixture (so one ASR pass decodes both), and SI-SDR (mixture / extracted vs the
target stem) when STEMS_DIR has {key}.target.wav. usage: tse_apply.py CKPT SET_DIR OUT_DIR [STEMS_DIR]"""
import json, os, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tse_model import TSE
from spk_embed import SpeakerEmbed
ck, sub, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); stems = Path(sys.argv[4]) if len(sys.argv) > 4 else None
out.mkdir(parents=True, exist_ok=True)
state = torch.load(ck, map_location="cpu"); mf2 = state.get("arch") == "mf2"; v2 = mf2 or any(k.startswith("ext.") for k in state["state"])
if mf2:
    from tse_mf2 import MF2TSE
    model = MF2TSE(pretrained=False, deep=bool(state.get("deep"))).cuda()
else:
    model = TSE(ext_emb=192 if v2 else 0).cuda()
model.load_state_dict(state["state"]); model.eval(); spk = SpeakerEmbed().cuda() if v2 else None
print("checkpoint step", state["step"], "dev", round(state["dev"], 2), flush=True)
def sisdr(est, ref):
    n = min(len(est), len(ref)); est, ref = est[:n], ref[:n]  # stems can be one sample shorter than the stored mixture
    ref = ref - ref.mean(); est = est - est.mean(); a = (est @ ref) / max(ref @ ref, 1e-10)
    return float(10 * np.log10(((a * ref) ** 2).sum() / max(((est - a * ref) ** 2).sum(), 1e-10)))
res = {}
for l in open(sub / "manifest.jsonl", encoding="utf-8"):
    r = json.loads(l); k = r["key"]
    e = sf.read(sub / "wav" / f"{k}.enroll.wav", dtype="float32")[0]; m = sf.read(sub / "wav" / f"{k}.mix.wav", dtype="float32")[0]
    et = torch.from_numpy(e)[None].cuda()
    mt = torch.from_numpy(m)[None].cuda()
    s = (model.extract(mt, spk(et)) if mf2 else model.extract(et, mt, spk=spk(et) if spk is not None else None))[0].cpu().numpy()
    sf.write(out / f"{k}.tse.wav", s.astype(np.float32), 16000, subtype="FLOAT")
    # remix: the extracted target plus the mixture at 0.15 (about -16 dB), a common way to soften extraction artifacts
    sf.write(out / f"{k}.remix.wav", (s + 0.15 * m).astype(np.float32), 16000, subtype="FLOAT")
    link = out / f"{k}.mix.wav"
    if not link.exists(): os.symlink(sub / "wav" / f"{k}.mix.wav", link)
    if stems and (stems / f"{k}.target.wav").exists():
        t = sf.read(stems / f"{k}.target.wav", dtype="float32")[0]
        res[r["id"]] = {"mix": sisdr(m, t), "tse": sisdr(s, t)}
json.dump({"step": state["step"], "sisdr": res}, open(out / "sisdr.json", "w"))
if res:
    print("SI-SDR mixture", round(np.mean([v["mix"] for v in res.values()]), 2), "extracted", round(np.mean([v["tse"] for v in res.values()]), 2), "n", len(res))
