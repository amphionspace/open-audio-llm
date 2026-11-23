"""Reference-free gate for the extractor: frozen CAM++ cosine of (extracted, enrollment) vs (mixture, enrollment) for
every test item; the gated output uses the extracted (or remix) transcript when the extracted audio is closer to the
enrollment than the mixture is and its transcript is non-empty, otherwise the mixture transcript.
usage: tse_gate.py TAG SUBSET_DIR SAFETY_DIR W_DIR OUT_JSON   (oal-train python; writes cosines, scoring is separate)"""
import json, sys
from pathlib import Path
import soundfile as sf, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spk_embed import SpeakerEmbed
tag, sub, saf, w, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5]
m = SpeakerEmbed().cuda(); res = {}
def emb(p): return torch.nn.functional.normalize(m(torch.from_numpy(sf.read(p, dtype="float32")[0])[None].cuda()), dim=-1)[0]
for base, cand in ((sub, sub / f"tse-{tag}"), (saf, saf / f"tse-{tag}")):
    for l in open(base / "manifest.jsonl", encoding="utf-8"):
        r = json.loads(l); k = r["key"]
        if not (cand / f"{k}.tse.wav").exists(): continue
        e = emb(base / "wav" / f"{k}.enroll.wav")
        res[r["id"]] = {"cos_mix": float(e @ emb(base / "wav" / f"{k}.mix.wav")), "cos_tse": float(e @ emb(cand / f"{k}.tse.wav"))}
json.dump(res, open(out, "w")); print(tag, len(res), "items")
