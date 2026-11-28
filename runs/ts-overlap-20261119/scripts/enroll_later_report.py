"""tasks/29 enrollment-swap causal test (README "判定细则"). For each split of ts_asr_real@meeting-enroll-later-20261128
(records keep their meeting-v5 ids), pair every record with the same id in the model's meeting-v5 run: miss (target
present, empty output) with the original enrollment vs the swapped one. The measured quantity is the extra misses of
the step-30k model over attempt 002, original vs swapped. G00000046 (RAMC 15-151) is excluded from the control verdict and
reported apart, as pre-registered. usage: enroll_later_report.py NAME=ORIG_RUN:VARIANT_RUN ...  (needs a002 and s30k)"""
import collections, json, sys

EXCLUDE = "ramc:G00000046"


def load(run):
    pred = {o["id"]: bool((o.get("prediction") or "").strip()) for o in map(json.loads, open(f"{run}/predictions.jsonl"))}
    rows = {}
    for s in map(json.loads, open(f"{run}/samples.jsonl")):
        if s["reference"].strip() and s["id"] in pred:
            rows[s["id"]] = (s["source"].split(":", 1)[1].split("/")[0], s["record"]["metadata"].get("target_speaker"), pred[s["id"]])
    return rows


M = {}
for arg in sys.argv[1:]:
    name, runs = arg.split("=", 1); orig, var = runs.split(":")
    M[name] = (load(orig), load(var))
var_ids = set.intersection(*(set(v) for _, v in M.values()))
groups = collections.defaultdict(list)
for i in sorted(var_ids):
    split, spk, _ = M["s30k"][1][i]
    if not all(i in o for o, _ in M.values()): continue
    g = split.replace("meeting-", "").replace("-pos", "")
    if g == "ramc-later-control" and spk == EXCLUDE: g = "ramc-later-control:G00000046"
    groups[g].append(i)


def misses(name, ids):
    o, v = M[name]; return sum(not o[i][2] for i in ids), sum(not v[i][2] for i in ids)


res = {}
print(f"{'group':34s} {'n':>5s}  " + "  ".join(f"{n} orig->swap" for n in M) + "   extra s30k-a002 orig -> swap")
order = ["aishell4-later", "ramc-later", "aishell4-later-control", "ramc-later-control", "ramc-later-control:G00000046"]
for g in order:
    ids = groups.get(g, [])
    if not ids: continue
    mm = {n: misses(n, ids) for n in M}
    ex = (mm["s30k"][0] - mm["a002"][0], mm["s30k"][1] - mm["a002"][1]); res[g] = ex
    print(f"{g:34s} {len(ids):5d}  " + "  ".join(f"{mm[n][0]:5d}->{mm[n][1]:<5d}" for n in M) + f"   {ex[0]:+5d} -> {ex[1]:+5d}")


def cut(keys):
    a = sum(res[k][0] for k in keys if k in res); b = sum(res[k][1] for k in keys if k in res)
    return a, b, (1 - b / a) if a > 0 else float("nan")


later, a4, ctrl = cut(["aishell4-later", "ramc-later"]), cut(["aishell4-later"]), cut(["aishell4-later-control", "ramc-later-control"])
print(f"\nlater (both corpora): extra {later[0]:+d} -> {later[1]:+d}, reduction {later[2]:.0%}")
print(f"AISHELL-4 later (main test): extra {a4[0]:+d} -> {a4[1]:+d}, reduction {a4[2]:.0%}")
print(f"control (excl. G00000046): extra {ctrl[0]:+d} -> {ctrl[1]:+d}, reduction {ctrl[2]:.0%}")
ok = later[2] > 0.5 and a4[2] > 0.5 and not ctrl[2] > 0.5
print("verdict:", "mechanism = start-of-recording enrollment" if ok else "not established by the pre-registered rule")
