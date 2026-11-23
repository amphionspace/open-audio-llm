"""Score enroll_tile_probe.py: target-present scenes: error (S+D+I over reference chars, empty = all deleted) and
whole-utterance misses; target-absent scenes: false alarms (non-empty output). Paired bootstrap (2,000): tileL - padL
and tileL - native5. usage: tile_report.py TILE_JSONL OUT_JSON   (amphion-eval python)"""
import collections, json, random, sys
from amphion_eval.integrations.open_audio_llm import scoring
def edits(r, h):
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1): cur[j] = min(prev[j-1] + (a != b), prev[j] + 1, cur[j-1] + 1)
        prev = cur
    return prev[-1]
d = collections.defaultdict(dict); ref = {}
for l in open(sys.argv[1], encoding="utf-8"):
    o = json.loads(l); d[o["id"]][o["cond"]] = (o["text"] or "").strip(); ref[o["id"]] = o["reference"]
conds = ["native5", "pad1", "tile1", "pad2", "tile2", "pad3", "tile3"]
pos = [i for i in d if scoring.tokens(ref[i] or "", "zh")]; neg = [i for i in d if i not in pos]
E = {i: {c: edits(scoring.tokens(ref[i], "zh"), scoring.tokens(d[i][c], "zh")) for c in conds} for i in pos}
N = {i: len(scoring.tokens(ref[i], "zh")) for i in pos}
res = {"present": {"scenes": len(pos)}, "absent": {"scenes": len(neg)}}
for c in conds:
    res["present"][c] = {"error": round(100 * sum(E[i][c] for i in pos) / sum(N.values()), 2), "misses": sum(1 for i in pos if not d[i][c])}
    res["absent"][c] = {"false_alarms": sum(1 for i in neg if d[i][c])}
rng = random.Random(0)
def boot(a, b):
    xs = []
    for _ in range(2000):
        s = [pos[rng.randrange(len(pos))] for _ in pos]; xs.append(100 * sum(E[i][a] - E[i][b] for i in s) / sum(N[i] for i in s))
    xs.sort(); return [round(100 * sum(E[i][a] - E[i][b] for i in pos) / sum(N.values()), 2), round(xs[50], 2), round(xs[1949], 2)]
res["paired_error"] = {f"tile{L}-pad{L}": boot(f"tile{L}", f"pad{L}") for L in (1, 2, 3)}
res["paired_error"].update({f"tile{L}-native5": boot(f"tile{L}", "native5") for L in (1, 2, 3)})
res["paired_error"].update({f"pad{L}-native5": boot(f"pad{L}", "native5") for L in (1, 2, 3)})
json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
for c in conds: print(c, res["present"][c], res["absent"][c])
for k, v in res["paired_error"].items(): print(k, v)
