"""tasks/29: scene-asr long dialogues (>= 30 s, 2+ speakers), plain ASR without enrollment: per model, the aggregate
from scene_asr_drops.measure, then per dialogue vs the first model: exact McNemar on "one speaker mostly dropped" and
a dialogue bootstrap (2,000, seed 0) of the change in dropped turns of non-first speakers. Same rules as measure().
usage: sceneasr_drop_pairs.py DROPS_DIR NAME=RUN_DIR ...   (DROPS_DIR holds scene_asr_drops.py, oal-ctx-grpo
feat/ctx-grpo-context-plan runs/zh-ctx-grpo-20261120/scripts)"""
import collections, difflib, json, math, random, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import scene_asr_drops as d


def per_dialogue(run):
    pred = {r["id"]: r.get("prediction") for r in map(json.loads, open(f"{run}/predictions.jsonl"))}
    out = {}
    for s in map(json.loads, open(f"{run}/samples.jsonl")):
        secs = s["record"]["audio_slots"][0]["ref"]["duration"]; segs = s["record"].get("metadata", {}).get("segments") or []
        if secs < 30 or len({x["speaker"] for x in segs}) < 2: continue
        ref, owner = "", []
        for k, x in enumerate(segs): ref += d.chars(x["text"]); owner += [k] * len(d.chars(x["text"]))
        got = [0] * len(ref)
        for b in difflib.SequenceMatcher(None, ref, d.chars(pred[s["id"]]), autojunk=False).get_matching_blocks():
            got[b.a:b.a + b.size] = [1] * b.size
        per = collections.defaultdict(lambda: [0, 0]); drop_other = 0
        for k, x in enumerate(segs):
            idx = [i for i, o in enumerate(owner) if o == k]; share = sum(got[i] for i in idx) / max(1, len(idx))
            per[x["speaker"]][0] += sum(got[i] for i in idx); per[x["speaker"]][1] += len(idx)
            drop_other += share < 0.3 and x["speaker"] != segs[0]["speaker"]
        shares = sorted(a / b for a, b in per.values())
        out[s["id"]] = (shares[0] < 0.5 and shares[-1] > 0.85, drop_other)
    return out


runs = [a.split("=", 1) for a in sys.argv[2:]]
for n, r in runs: print(n, *d.measure(Path(r)))
P = {n: per_dialogue(r) for n, r in runs}; base = runs[0][0]; ids = sorted(P[base]); rng = random.Random(0)
for n, _ in runs[1:]:
    b = sum(P[n][i][0] and not P[base][i][0] for i in ids); c = sum(P[base][i][0] and not P[n][i][0] for i in ids)
    p = min(1.0, 2 * sum(math.comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0
    diffs = [P[n][i][1] - P[base][i][1] for i in ids]
    boots = sorted(sum(diffs[rng.randrange(len(ids))] for _ in ids) for _ in range(2000))
    print(f"{n} vs {base}: mostly-dropped new {b}, fixed {c}, exact McNemar p={p:.3f}; "
          f"other-speaker dropped turns {sum(diffs):+d} [{boots[50]:+d}, {boots[1949]:+d}] (dialogue bootstrap 95%)")
