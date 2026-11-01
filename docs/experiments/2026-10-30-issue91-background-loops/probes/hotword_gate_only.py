"""Gate-only counterfactual on hotword-bench: baseline per-sample scores with only the rejected rows rescored as empty.

Writes OUT/base/summary.json and OUT/cf/summary.json for compare_hotword.py.

    python hotword_gate_only.py BASELINE_RUN_DIR GATE_RUN_DIR OUT
"""
import json, os, sys, time
from amphion_eval.integrations.open_audio_llm import hotword_bench
from amphion_eval.legacy.open_audio_llm.qwen3_asr import normalize as presence_text
NO_SPEECH = "language None<asr_text>"
base_dir, gate_dir, out = sys.argv[1:4]
t0 = time.time()
scores = [json.loads(l) for l in open(f"{base_dir}/hotword-scores.jsonl")]
pos = {(s["id"], s["condition"]): k for k, s in enumerate(scores)}
gate = {r["id"]: r for r in map(json.loads, open(f"{gate_dir}/predictions.jsonl"))}
base = {r["id"]: r for r in map(json.loads, open(f"{base_dir}/predictions.jsonl"))}
todo = [r for i, r in base.items() if (gate[i].get("raw_text") or "").startswith(NO_SPEECH) and presence_text(r["prediction"])]
need = {r["id"] for r in todo}
records = {}
for line in open(f"{base_dir}/samples.jsonl"):
    s = json.loads(line)
    if s["id"] in need:
        records[s["id"]] = s["record"]
cf = list(scores)
for r in todo:
    k = pos[r["sample_id"], r["condition"]]
    cf[k] = {**hotword_bench.score_sample(records[r["id"]], r["condition"], r["candidates"], ""), "label": r["label"]}
print("rescored", len(todo), f"{time.time() - t0:.0f}s", flush=True)
for name, rows in (("base", scores), ("cf", cf)):
    os.makedirs(f"{out}/{name}", exist_ok=True)
    json.dump({"hotword_bench": hotword_bench.summarize(rows)}, open(f"{out}/{name}/summary.json", "w"))
    print(name, f"{time.time() - t0:.0f}s", flush=True)
