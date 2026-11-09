import glob, gzip, json, sys, collections, re
from amphion_eval.normalize import tokenize_text
from rapidfuzz.distance import Levenshtein

def runs(model, evalset):
    out = []
    for d in sorted(glob.glob(f"/workspace/data/ae-runs/{model}/runs/*")):
        try:
            if evalset in open(d + "/run.json").read():
                out.append(d)
        except FileNotFoundError:
            pass
    return out[-1]

def load(model, evalset):
    d = runs(model, evalset)
    return {p["id"]: p for p in map(json.loads, open(d + "/predictions.jsonl")) if p.get("status") == "success"}

def identical(version):
    idx = {}
    for l in gzip.open(f"/workspace/data/datasets/ts_asr_crosslingual/versions/{version}/audio-index.jsonl.gz", "rt"):
        r = json.loads(l); idx[r["cut_id"]] = r["sha256"]
    return idx

def key(pid):  # ...:<split>/...:<cut> -> split, index
    m = re.search(r"((?:cv|tts)-[a-z0-9]+)-(same|cross):(\d{4})", pid)
    return m.groups() if m else None

for evalset, version in (("ts-asr-crosslingual@v1-20261109", "zh-v1-20261109"), ("ts-asr-crosslingual-en@v1-20261109", "en-v1-20261109")):
    idx = identical(version)
    print(f"\n######## {evalset} (mixture {version[:2]})")
    print(f"{'model':12s} {'scene':10s} | {'pairs':>5s} | same: CER  miss  FA  | cross: CER  miss  FA")
    for model, label in (("whitebox-merge-sum-full", "sum-full"), ("ts-robust-073-avg3", "sft073")):
        preds = load(model, evalset)
        by = collections.defaultdict(dict)
        for pid, p in preds.items():
            k = key(pid)
            if k: by[(k[0], k[2])][k[1]] = p
        agg = collections.defaultdict(lambda: {"n": 0, "same": [0, 0, 0, 0], "cross": [0, 0, 0, 0]})
        for (scene, i), pair in by.items():
            if len(pair) != 2: continue
            if idx.get(f"{scene}-same-{i}-mix") != idx.get(f"{scene}-cross-{i}-mix"): continue
            a = agg[scene]; a["n"] += 1
            for cond, p in pair.items():
                lang = p.get("language") or ""
                ref, hyp = tokenize_text(p["reference"], lang), tokenize_text(p.get("prediction", ""), lang)
                r = a[cond]; r[0] += Levenshtein.distance(ref, hyp); r[1] += len(ref)
                r[2] += bool(ref) and not hyp; r[3] += (not ref) and bool(hyp)
        for scene in sorted(agg, key=lambda s: (s.split("-")[0], ["solo", "k1", "k2", "abs1", "abs2"].index(s.split("-")[1]))):
            a = agg[scene]; n = a["n"]
            def fmt(r):
                cer = f"{100*r[0]/r[1]:5.1f}" if r[1] else "   - "
                return f"{cer} {100*r[2]/n:5.1f}% {100*r[3]/n:5.1f}%"
            print(f"{label:12s} {scene:10s} | {n:5d} | {fmt(a['same'])} | {fmt(a['cross'])}")
