"""第二批有输出的片段：配 30 个随机错误说话人的注册音频，统计接受率。"""
import random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import probe_cases as C
import probe_ts as P
rng = random.Random(5)
SR = C.SR
recs = P.load_jsonl("test_1spk")
by = {}
for r in recs: by.setdefault(r["labels"]["target_spk"], []).append(r)
picked = [rng.choice(v) for v in by.values()] + [rng.choice(v) for v in by.values()]
paths = P.load_index({r["audio_slots"][0]["ref"]["cut_id"] for r in picked})
def enr(r):
    w = P.read(paths[r["audio_slots"][0]["ref"]["cut_id"]])[: 3 * SR]
    return np.pad(w, (0, 3 * SR - len(w)))
enrolls = [enr(r) for r in picked[:30]]
reg2 = C.read(C.ROOT / "case-08" / "registration-reference.wav")
rms = lambda x: float(np.sqrt((x ** 2).mean() + 1e-12))
for c in ["case-08", "case-09", "case-11", "case-10"]:
    src = C.read(C.ROOT / c / "source-audio.wav")
    acc = [bool(P.norm(C.ask(np.concatenate([e * (rms(reg2) / rms(e)), src]).astype(np.float32), max_tokens=64)[0])) for e in enrolls]
    print(f"{c} {len(src)/SR:5.2f}s: 随机错误说话人注册（电平对齐）接受率 {100*np.mean(acc):.0f}% ({sum(acc)}/{len(acc)})")
