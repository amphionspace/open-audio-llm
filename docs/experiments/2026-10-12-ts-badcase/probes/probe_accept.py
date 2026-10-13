"""badcase 的接受率对照：同一批源音频配随机"错误说话人"注册，对比测试人员的注册音频；以及电平归一化。"""
import json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import probe_cases as C
import probe_ts as P

rng = random.Random(3)
SR = P.SR
recs = P.load_jsonl("test_1spk")
by_spk = {}
for r in recs:
    by_spk.setdefault(r["labels"]["target_spk"], []).append(r)
picked = [x for v in by_spk.values() for x in rng.sample(v, min(5, len(v)))]  # 每个说话人最多 5 条
paths = P.load_index({r["audio_slots"][0]["ref"]["cut_id"] for r in picked})
def enr(r):
    w = P.read(paths[r["audio_slots"][0]["ref"]["cut_id"]])[: 3 * SR]
    return np.pad(w, (0, 3 * SR - len(w)))
enrolls = [enr(r) for r in picked]
rms = lambda x: float(np.sqrt((x ** 2).mean() + 1e-12))
train_rms = np.mean([rms(e) for e in enrolls])
tester_reg = C.read(C.ROOT / "case-02" / "registration-reference.wav")
print(f"AISHELLMix enrollment mean RMS {20*np.log10(train_rms):.1f} dB; tester registration {20*np.log10(rms(tester_reg)):.1f} dB")
cases = ["case-01", "case-02", "case-03", "case-05", "case-06", "case-07"]
srcs = {c: C.read(C.ROOT / c / "source-audio.wav") for c in cases}
gain = lambda e, target: e * (target / rms(e))

def run(j):
    c, kind, e = j
    t, _ = C.ask(np.concatenate([e, srcs[c]]).astype(np.float32), max_tokens=64)
    return c, kind, bool(P.norm(t))
jobs = []
for c in cases:
    for e in enrolls:
        jobs.append((c, "wrong_clean_asis", e))
        jobs.append((c, "wrong_at_tester_level", gain(e, rms(tester_reg))))
    jobs.append((c, "tester_reg_asis", tester_reg))
    jobs.append((c, "tester_reg_gain_to_train_level", np.clip(gain(tester_reg, train_rms), -1, 1)))
with ThreadPoolExecutor(16) as ex:
    out = list(ex.map(run, jobs))
from collections import defaultdict
agg = defaultdict(list)
for c, k, a in out:
    agg[(c, k)].append(a)
print(f"{'case':8s} {'wrong_clean%':>12s} {'wrong@tester_lvl%':>18s} {'tester_reg':>10s} {'tester_reg_gain→train':>22s}")
tot = defaultdict(list)
for c in cases:
    w = np.mean(agg[(c, 'wrong_clean_asis')]) * 100; l = np.mean(agg[(c, 'wrong_at_tester_level')]) * 100
    t = agg[(c, 'tester_reg_asis')][0]; g = agg[(c, 'tester_reg_gain_to_train_level')][0]
    print(f"{c:8s} {w:12.0f} {l:18.0f} {str(t):>10s} {str(g):>22s}")
    tot['w'] += agg[(c, 'wrong_clean_asis')]; tot['l'] += agg[(c, 'wrong_at_tester_level')]
print("mean accept wrong_clean %.1f%%, wrong@tester_level %.1f%% (n=%d)" % (np.mean(tot['w'])*100, np.mean(tot['l'])*100, len(tot['w'])))
