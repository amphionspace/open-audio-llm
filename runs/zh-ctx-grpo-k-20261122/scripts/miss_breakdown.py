"""README "差值合并诊断" static check: where a model's TS misses (target present, empty output) come from, on one
delivery suite, against the first model given. Reads the delivery runs of runs/zh-ctx-grpo-20261120
(ae-runs/ctx-grpo-20261120-delivery-<model>, latest complete run of the suite) and prints misses and false alarms per
model, misses by the suite's labels (cell, corpus, first speaker, enrollment domain), the last model's new misses
against the first, and misses by reference length (CJK characters, else words).
usage: python3 miss_breakdown.py SUITE_KEY MODEL [MODEL ...]   (e.g. ts-asr-first-other r1117 startj k pairmerge)
"""
import collections
import glob
import json
import re
import sys

RUNS = "/workspace/data/ae-runs/ctx-grpo-20261120-delivery-{}/runs"
LABELS = ("cell", "corpus", "first_speaker", "enroll_domain")
LENGTH_BINS = ((0, 10), (10, 20), (20, 40), (40, None))


def find(model, key):
    found = [r for r in sorted(glob.glob(RUNS.format(model) + "/run-*"))
             if key in (json.load(open(r + "/run.json"))["config"].get("eval_set") or "")
             and json.load(open(r + "/coverage.json")).get("complete")]
    return found[-1]


def load(run):
    samples = {s["id"]: s for s in map(json.loads, open(run + "/samples.jsonl"))}
    preds = {p["id"]: (p.get("prediction") or "").strip() for p in map(json.loads, open(run + "/predictions.jsonl"))}
    return samples, preds


def units(text):
    return len(re.sub(r"[^一-鿿]", "", text)) or len(text.split())


def main():
    key, models = sys.argv[1], sys.argv[2:]
    data = {m: load(find(m, key)) for m in models}
    samples = data[models[0]][0]
    pos = [i for i, s in samples.items() if s["reference"].strip()]
    neg = [i for i, s in samples.items() if not s["reference"].strip()]
    missed = {m: {i for i in pos if not data[m][1][i]} for m in models}
    print(f"{key}: {len(pos)} with the target, {len(neg)} without")
    for m in models:
        fa = sum(bool(data[m][1][i]) for i in neg)
        print(f"  {m}: misses {len(missed[m])} ({100 * len(missed[m]) / len(pos):.2f}%), false alarms {fa}")
    for label in LABELS:
        cells = collections.defaultdict(lambda: [0] * (len(models) + 1))
        for i in pos:
            c = cells[samples[i]["record"].get("labels", {}).get(label)]
            c[0] += 1
            for j, m in enumerate(models):
                c[j + 1] += i in missed[m]
        if len(cells) > 1:
            print(f"  by {label} [n, misses of {', '.join(models)}]:", dict(sorted(cells.items(), key=lambda x: str(x[0]))))
    a, b = models[0], models[-1]
    new = missed[b] - missed[a]
    print(f"  new misses of {b} against {a}: {len(new)}; misses of {a} that {b} recovers: {len(missed[a] - missed[b])}")
    where = collections.Counter((samples[i]["record"].get("labels", {}).get("cell"), samples[i]["record"].get("labels", {}).get("corpus"))
                                for i in new)
    print("  new misses by (cell, corpus):", where.most_common(10))
    for i in sorted(new)[:5]:
        print(f"    {i}: ref {samples[i]['reference'][:40]} | {a}: {data[a][1][i][:40]}")
    print("  misses by reference length:")
    for lo, hi in LENGTH_BINS:
        ids = [i for i in pos if lo <= units(samples[i]["reference"]) and (hi is None or units(samples[i]["reference"]) < hi)]
        print(f"    [{lo}, {hi or ''}) n={len(ids)} " + " ".join(f"{m}={len(missed[m] & set(ids))}" for m in models))


if __name__ == "__main__":
    main()
