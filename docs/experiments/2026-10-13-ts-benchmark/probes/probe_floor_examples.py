import gzip, json, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
import probe_cases as C, probe_ts as P
ROOT = Path("/workspace/data/datasets/ts_asr_benchmark"); VER = "v1-20261013"; SR = 16000
idx = {}
for l in gzip.open(ROOT/"versions"/VER/"audio-index.jsonl.gz","rt"):
    r = json.loads(l); idx[r["cut_id"]] = ROOT/r["relative_path"]
orig = {json.loads(l)["id"]: json.loads(l)["prediction"] for l in open("tsb/run2/predictions.jsonl")}
rows = []
for l in list(gzip.open(ROOT/"versions"/VER/"records"/"dyn-solo.jsonl.gz","rt"))[:100]:
    r = json.loads(l); enr, mix = (idx[s["ref"]["cut_id"]] for s in r["audio_slots"])
    e,_ = sf.read(enr, dtype="float32"); e = np.pad(e[:3*SR],(0,max(0,3*SR-len(e[:3*SR])))); m,_ = sf.read(mix, dtype="float32")
    g = np.random.default_rng(0); amp = 10**(-50/20)
    t,_ = C.ask(np.concatenate([e+g.standard_normal(len(e)).astype(np.float32)*amp, m+g.standard_normal(len(m)).astype(np.float32)*amp]).astype(np.float32))
    d = P.edit(P.norm(t), P.norm(r["target"])) - P.edit(P.norm(orig[r["id"]]), P.norm(r["target"]))
    rows.append((d, r, t))
rows.sort(key=lambda x: -x[0])
for d, r, t in rows[:5]:
    print(f"+{d} edits [{r['labels']['duration_s']}s gap {r['labels'].get('target_gap_s')}] ref: {r['target']}\n   no floor: {orig[r['id']]}\n   floor   : {t[:160]}\n")
print('samples with +10 edits or more:', sum(1 for d,_,_ in rows if d>=10), 'of', len(rows))
