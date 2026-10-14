"""合成混合里的发言间隙是数字静音（精确的 0）。加上噪声底后模型的输出会变吗？配对对照。"""
import gzip, json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
import probe_cases as C
import probe_ts as P

ROOT = Path("/workspace/data/datasets/ts_asr_benchmark")
VER = "v1-20261013"
FLOORS = [float(x) for x in sys.argv[2:]] or [-50.0]
SPLITS = ["enroll-clean-pos", "enroll-clean-neg", "dyn-solo", "dyn-full-k1", "dyn-burst-k2", "dyn-short-k1", "abs-k1", "abs-k2", "abs-short", "dom-studio-pos", "nonspeech-noise"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
idx = {}
for l in gzip.open(ROOT / "versions" / VER / "audio-index.jsonl.gz", "rt"):
    r = json.loads(l); idx[r["cut_id"]] = ROOT / r["relative_path"]
orig = {}
for l in open(sys.argv[0].replace("probe_floor.py", "tsb/run2/predictions.jsonl")):
    r = json.loads(l); orig[r["id"]] = r["prediction"]
recs = []
for sp in SPLITS:
    for i, l in enumerate(gzip.open(ROOT / "versions" / VER / "records" / f"{sp}.jsonl.gz", "rt")):
        if i >= N: break
        recs.append(json.loads(l))
SR = 16000
def run(job):
    r, floor = job
    enr, mix = (idx[s["ref"]["cut_id"]] for s in r["audio_slots"])
    e, _ = sf.read(enr, dtype="float32"); e = e[:3 * SR]; e = np.pad(e, (0, 3 * SR - len(e)))
    m, _ = sf.read(mix, dtype="float32")
    g = np.random.default_rng(0)
    amp = 10 ** (floor / 20)
    e2 = e + g.standard_normal(len(e)).astype(np.float32) * amp
    m2 = m + g.standard_normal(len(m)).astype(np.float32) * amp
    text, _ = C.ask(np.concatenate([e2, m2]).astype(np.float32), max_tokens=512)
    return r["id"], floor, text
jobs = [(r, f) for r in recs for f in FLOORS]
with ThreadPoolExecutor(24) as ex:
    out = list(ex.map(run, jobs))
res = {(i, f): t for i, f, t in out}
rows = []
for sp in SPLITS:
    rs = [r for r in recs if r["labels"]["scenario"] == sp]
    for f in FLOORS:
        if rs[0]["labels"]["target_present"]:
            def cer(h): return sum(P.edit(P.norm(h(r)), P.norm(r["target"])) for r in rs) / max(sum(len(P.norm(r["target"])) for r in rs), 1)
            a, b = cer(lambda r: orig[r["id"]]), cer(lambda r: res[(r["id"], f)])
            loops = lambda h: sum(1 for r in rs if len(P.norm(h(r))) > 3 * len(P.norm(r["target"])) + 30)
            keep = [r for r in rs if len(P.norm(res[(r["id"], f)])) <= 3 * len(P.norm(r["target"])) + 30 and len(P.norm(orig[r["id"]])) <= 3 * len(P.norm(r["target"])) + 30]
            def cerk(h): return sum(P.edit(P.norm(h(r)), P.norm(r["target"])) for r in keep) / max(sum(len(P.norm(r["target"])) for r in keep), 1)
            print(f"  [复读 {loops(lambda r: orig[r['id']])} -> {loops(lambda r: res[(r['id'], f)])} 条；去掉复读后 CER {100*cerk(lambda r: orig[r['id']]):.1f} -> {100*cerk(lambda r: res[(r['id'], f)]):.1f}]", end="")
            print(f"{sp:20s} floor {f:6.1f} dBFS  CER 无噪声底 {100*a:5.1f} -> 加噪声底 {100*b:5.1f}  miss {100*np.mean([not P.norm(orig[r['id']]) for r in rs]):.0f}% -> {100*np.mean([not P.norm(res[(r['id'], f)]) for r in rs]):.0f}%")
        else:
            a = np.mean([bool(P.norm(orig[r["id"]])) for r in rs]); b = np.mean([bool(P.norm(res[(r["id"], f)])) for r in rs])
            print(f"{sp:20s} floor {f:6.1f} dBFS  误输出 无噪声底 {100*a:5.1f}% -> 加噪声底 {100*b:5.1f}%  (n={len(rs)})")
