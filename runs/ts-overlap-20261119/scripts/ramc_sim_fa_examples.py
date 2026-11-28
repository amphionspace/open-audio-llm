"""RAMC same-recording negatives with impostor_sim >= 0.4 (benchmark sim_ge_0.4): false-alarm rate per model, paired
(1117 FA, new FA) counts, and the false-alarm outputs.  Runs are fixed below (11.17, task 28 attempt 001 / 002 at 30k)."""
import json, collections
R = "/workspace/data/ae-runs"
RUNS = {"1117": f"{R}/whitebox-merge-v18a-es116inc/runs/run-20261125T030434Z-79f66f30",
        "m30k": f"{R}/ts-overlap-meeting001-s30000/runs/run-20261125T221645Z-605a6fb1",
        "q30k": f"{R}/ts-overlap-meeting002-s30000/runs/run-20261126T165219Z-b2494082"}
lab = {o["id"]: o["record"] for o in map(json.loads, open(RUNS["q30k"] + "/samples.jsonl")) if ":meeting-ramc-neg/" in o["source"]}
first = next(iter(lab.values())); print(json.dumps(first["labels"], ensure_ascii=False)); print(json.dumps({k: v for k, v in first.get("metadata", {}).items() if k != "enrollment_pieces"}, ensure_ascii=False)[:800])
pred = {n: {o["id"]: (o.get("prediction") or "").strip() for o in map(json.loads, open(r + "/predictions.jsonl")) if o["id"] in lab} for n, r in RUNS.items()}
hi = [i for i, r in lab.items() if (r["labels"].get("impostor_sim") or 0) >= 0.4]
print("sim>=0.4", len(hi), {n: round(100 * sum(bool(pred[n][i]) for i in hi) / len(hi), 2) for n in RUNS})
both = collections.Counter((bool(pred["1117"][i]), bool(pred["q30k"][i])) for i in hi); print("(1117 FA, q30k FA)", dict(both))
seg = lambda r: r["labels"].get("segment_text") or r.get("target") or ""
print("examples q30k FA (sim, dur, pred):")
for i in hi[:400]:
    if pred["q30k"][i]:
        l = lab[i]["labels"]; print(f"  {l.get('impostor_sim'):.2f} {l.get('duration_s')}s  1117={pred['1117'][i][:30]!r}  q30k={pred['q30k'][i][:40]!r}")
