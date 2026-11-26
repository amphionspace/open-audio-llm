"""Meeting benchmark present-target misses (empty output) by record label bins, paired across models on the same ids,
plus the recordings and target speakers that hold the extra misses of the last model vs the first.
usage: meeting_miss_bins.py SPLIT NAME=RUN_DIR NAME=RUN_DIR ...   (e.g. meeting-aishell4-pos; labels from the last run's samples.jsonl)"""
import collections, json, sys

split, runs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
names = [n for n, _ in runs]; first, last = names[0], names[-1]
lab = {o["id"]: (o["record"]["labels"], o["reference"]) for o in map(json.loads, open(runs[-1][1] + "/samples.jsonl"))
       if f":{split}/" in o["source"]}
pred = {n: {o["id"]: (o.get("prediction") or "").strip() for o in map(json.loads, open(r + "/predictions.jsonl")) if o["id"] in lab}
        for n, r in runs}
ids = [i for i in lab if lab[i][1].strip() and all(i in pred[n] for n in names)]
print(split, "pairs", len(ids))


def binned(name, key, edges):
    rows = collections.defaultdict(collections.Counter)
    for i in ids:
        v = lab[i][0].get(key)
        b = "na" if v is None else next((f"<{e}" for e in edges if v < e), f">={edges[-1]}")
        rows[b]["n"] += 1
        for n in names: rows[b][n] += not pred[n][i]
    print(f"-- {name}")
    order = [f"<{e}" for e in edges] + [f">={edges[-1]}", "na"]
    for b in [b for b in order if b in rows]:
        c = rows[b]; print(f"  {b:>8} n {c['n']:5d}  " + "  ".join(f"{n} {100 * c[n] / c['n']:5.1f}" for n in names))


binned("enroll_own_sim", "enroll_own_sim", [0.5, 0.6, 0.7, 0.8])
binned("target_seconds", "target_seconds", [1, 2, 3, 5, 10])
binned("target_overlap_ratio", "target_overlap_ratio", [0.01, 0.3, 0.6, 0.9])
binned("speakers_in_window", "speakers_in_window", [2, 3, 4])
binned("mixture_rms_dbfs", "mixture_rms_dbfs", [-45, -40, -35, -30, -25])
for unit, key in (("recordings", lambda l: l["recording"]), ("target speakers", lambda l: (l["recording"], l["segment_speaker"]))):
    acc = collections.defaultdict(collections.Counter)
    for i in ids:
        c = acc[key(lab[i][0])]; c["n"] += 1
        for n in names: c[n] += not pred[n][i]
    d = sorted(((c[last] - c[first], k, c) for k, c in acc.items()), key=lambda t: -t[0])
    print(f"-- {unit} {len(acc)}: extra misses {last}-{first} {sum(x for x, _, _ in d)}, top 10 hold {sum(x for x, _, _ in d[:10])}")
    for x, k, c in d[:10]: print(f"  {k} n {c['n']:4d}  " + "  ".join(f"{n} {c[n]:3d}" for n in names))
