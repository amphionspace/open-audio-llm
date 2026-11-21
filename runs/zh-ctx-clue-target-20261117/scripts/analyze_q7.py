"""Question 7 follow-up: where J's keep-rate shortfall comes from (README "结果：第 7 问").

Reads the latest context-correction v2 runs (think off) of template, J and H, the scene-TS runs, and the ECAPA
cosines of the TS-mode items (ts_similarity.py output in tasks/02-eval/compare/ctxv2-ts-ecapa.json). Prints:
1. keep items (explicit / consistency confirm) J lost or won against template, by mode and cause (empty output,
   over-correction, other), and the keep rate on the items both answered;
2. TS-mode items left empty, by enrollment-clone cosine bin;
3. scene-TS false alarms by scene group.
usage: python3 analyze_q7.py
"""
import json
from collections import Counter
from pathlib import Path

RUNS = Path("/workspace/data/ae-runs")
SIM = Path(__file__).resolve().parents[1] / "tasks/02-eval/compare/ctxv2-ts-ecapa.json"
ARMS = {"template": "template", "J": "template-synth-ts-twins", "H": "template-no-synth-ts", "I": "template-synth-ts-mix"}
BINS = (-1.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01)
SCENES = (("extreme-overlap", "极度重叠"), ("heavy-overlap", "重度重叠"), ("light-overlap", "轻度重叠"), ("light-sep", "轻度分开讲话"))


def latest(name):
    return sorted((RUNS / f"ctx-clue-target-20261117-{name}/runs").glob("run-*"))[-1]


def load(name):
    run = latest(name)
    scores = {r["id"]: r for r in map(json.loads, open(run / "context-correction-scores.jsonl", encoding="utf-8"))}
    preds = {r["id"]: r["prediction"] for r in map(json.loads, open(run / "predictions.jsonl", encoding="utf-8"))}
    return scores, preds


def main():
    runs = {k: load(v) for k, v in ARMS.items()}
    (ts, tp), (js, jp) = runs["template"], runs["J"]
    keep = [i for i, r in ts.items() if r["category"] in ("explicit", "consistency") and r["kind"] == "confirm"]
    cause = lambda i, s, p: "empty" if not p[i].strip() else "over-correct" if s[i]["target_alternative"] else "other"
    lost = [i for i in keep if ts[i]["target_ok"] and not js[i]["target_ok"]]
    won = [i for i in keep if not ts[i]["target_ok"] and js[i]["target_ok"]]
    print(f"1. keep items {len(keep)}: J lost {len(lost)}, won {len(won)}")
    print("   lost (mode, J's output):", dict(Counter((js[i]["mode"], cause(i, js, jp)) for i in lost)))
    print("   won (mode, template's output):", dict(Counter((ts[i]["mode"], cause(i, ts, tp)) for i in won)))
    both = [i for i in keep if tp[i].strip() and jp[i].strip()]
    rate = lambda s: 100 * sum(s[i]["target_ok"] for i in both) / len(both)
    print(f"   both answered {len(both)}: keep rate template {rate(ts):.2f}, J {rate(js):.2f}")
    sim = json.loads(SIM.read_text())
    empty = {k: {i for i in sim if not p[i].strip()} for k, (_, p) in runs.items()}
    print("2. TS-mode items left empty by enrollment-clone ECAPA cosine:",
          {k: len(v) for k, v in empty.items()}, f"J only {len(empty['J'] - empty['template'])}")
    for lo, hi in zip(BINS, BINS[1:]):
        ids = [i for i, s in sim.items() if lo <= s < hi]
        print(f"   [{lo:.1f}, {min(hi, 1.0):.1f}) {len(ids):5d} items:",
              "  ".join(f"{k} {100 * sum(i in e for i in ids) / len(ids):5.1f}%" for k, e in empty.items()))
    print("3. scene-TS false alarms (target-absent items with output):")
    fa = {}
    for k in ("template", "J", "H"):
        name = "scene-" + ARMS[k]
        counts = Counter()
        for r in map(json.loads, open(latest(name) / "predictions.jsonl", encoding="utf-8")):
            if not r["reference"].strip():
                scene = next(label for key, label in SCENES if key in r["source"])
                counts[scene, "items"] += 1
                counts[scene, "fa"] += bool(r["prediction"].strip())
        fa[k] = counts
    for _, label in SCENES:
        print(f"   {label}: {fa['template'][label, 'items']} items,",
              ", ".join(f"{k} {fa[k][label, 'fa']}" for k in fa))


if __name__ == "__main__":
    main()
