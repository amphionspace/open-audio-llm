"""README "续训本身的纠偏收益": where two context-correction v2 runs (think off) differ on the real recordings.
Over the Emilia mined items (context-correction-scores.jsonl of each run, labels from samples.jsonl), counts items wrong in
A and right in B ("new right") and the reverse ("new wrong"), split by category (explicit / implicit / consistency),
train_overlap (seen / unseen) and A's kind of miss: ``alternative`` = A wrote the labelled homophone (the common
spelling), ``other`` = any other wrong output. Writes tasks/09-datasplit/flips-<a>-<b>.json and prints the table.
Run names are ctx-grpo-k-20261122-<name> (or the 11-21 runs of runs/zh-ctx-clue-target-20261117 for template/h/j, as in
compare_datasplit.sh).
usage: python3 ctx_flip_breakdown.py A:B [A:B ...]
"""
import collections
import glob
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tasks/09-datasplit"
OLD = {"template": "template", "h": "template-no-synth-ts", "j": "template-synth-ts-twins"}


def run(name):
    d = f"/workspace/data/ae-runs/ctx-clue-target-20261117-{OLD[name]}" if name in OLD else \
        f"/workspace/data/ae-runs/ctx-grpo-k-20261122-{name}"
    return sorted(glob.glob(d + "/runs/run-*"))[-1]


def items(name):
    r = run(name)
    labels = {s["id"]: s["record"]["labels"]["context_correction"] for s in map(json.loads, open(r + "/samples.jsonl"))
              if "emilia" in s["id"]}
    return {x["id"]: (x, labels[x["id"]]) for x in map(json.loads, open(r + "/context-correction-scores.jsonl"))
            if x["id"] in labels}


def breakdown(a, b):
    A, B = items(a), items(b)
    assert A.keys() == B.keys(), (len(A), len(B))
    out = collections.Counter()
    for i, (x, lab) in A.items():
        y = B[i][0]
        if x["target_ok"] == y["target_ok"]:
            continue
        flip = "new_right" if y["target_ok"] else "new_wrong"
        # the miss that was fixed (new right) or introduced (new wrong)
        miss = x if flip == "new_right" else y
        kind = "alternative" if miss["target_alternative"] else "other"
        for key in ("all", f"category={lab['category']}", f"overlap={lab['train_overlap']}", f"miss={kind}",
                    f"overlap={lab['train_overlap']},miss={kind}"):
            out[f"{key}|{flip}"] += 1
    n = len(A)
    acc = (sum(x["target_ok"] for x, _ in A.values()) / n, sum(y["target_ok"] for y, _ in B.values()) / n)
    return {"n": n, "accuracy": acc, "flips": dict(sorted(out.items()))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for pair in sys.argv[1:]:
        a, b = pair.split(":")
        res = breakdown(a, b)
        (OUT / f"flips-{a}-{b}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
        keys = sorted({k.split("|")[0] for k in res["flips"]}, key=lambda k: (k != "all", k))
        print(f"{a} -> {b}: Emilia {100 * res['accuracy'][0]:.2f} -> {100 * res['accuracy'][1]:.2f} (n {res['n']})")
        for k in keys:
            r, w = res["flips"].get(f"{k}|new_right", 0), res["flips"].get(f"{k}|new_wrong", 0)
            print(f"  {k:34s} new right {r:4d}  new wrong {w:4d}  net {r - w:+d}")


if __name__ == "__main__":
    main()
