"""单个干扰人（目标缺席、无混合）和低电平源音频的拒识/漏转写探测。"""
import json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import probe_ts as P

rng = random.Random(1)
rs = P.load_jsonl("test_1spk")
rs = rng.sample(rs, 300)
all1 = P.load_jsonl("test_1spk")
cuts = {s["ref"]["cut_id"] for r in rs for s in r["audio_slots"]}
byspk = {}
for r in all1:
    byspk.setdefault(r["labels"]["target_spk"], []).append(r)
other_rec = {}
for r in rs:
    spk = r["labels"]["target_spk"]
    o = rng.choice([x for k, v in byspk.items() if k != spk for x in v[:5]])
    other_rec[r["id"]] = o
    cuts.add(o["audio_slots"][0]["ref"]["cut_id"])
paths = P.load_index(cuts)

def enr(rec):
    w = P.read(paths[rec["audio_slots"][0]["ref"]["cut_id"]])[: 3 * P.SR]
    return np.pad(w, (0, 3 * P.SR - len(w)))

def job(a):
    r, cond = a
    src = P.read(paths[r["audio_slots"][1]["ref"]["cut_id"]])
    wrong = cond.startswith("neg")
    e = enr(other_rec[r["id"]] if wrong else r)
    g = -12 if cond.endswith("-12") else 0
    src = src * 10 ** (g / 20)
    hyp = P.ask("http://127.0.0.1:8011", "amphionasr-v3-tsasr", np.concatenate([e, src]).astype(np.float32))
    return {"cond": cond, "id": r["id"], "hyp": hyp, "ref": r["target"]}

jobs = [(r, c) for r in rs for c in ["pos_g0", "pos_g-12", "neg_g0", "neg_g-12"]]
with ThreadPoolExecutor(24) as ex:
    out = list(ex.map(job, jobs))
Path(sys.argv[1]).write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out))
for c in ["pos_g0", "pos_g-12", "neg_g0", "neg_g-12"]:
    xs = [o for o in out if o["cond"] == c]
    empty = np.mean([P.norm(o["hyp"]) == "" for o in xs]) * 100
    if c.startswith("pos"):
        err = sum(P.edit(P.norm(o["hyp"]), P.norm(o["ref"])) for o in xs); ln = sum(len(P.norm(o["ref"])) for o in xs)
        print(f"{c:10s} n={len(xs)} CER={err/ln*100:5.2f}  empty(漏转写)={empty:5.1f}%")
    else:
        print(f"{c:10s} n={len(xs)} reject={empty:5.1f}%  误输出={100-empty:5.1f}%")
