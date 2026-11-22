"""Side-effect check set from the 11.17 v12 run, main benchmark (ts_asr_benchmark@v4) Chinese TS samples:
200 target-present with no overlap, 200 target-present with 0 < overlap < 0.6, 200 target-absent. Same export as
sep_export_subset.py (5 s enrollment, mixture, manifest with the eval's greedy output). usage: export_safety.py RUN OUT"""
import json, random, sys
from pathlib import Path
import numpy as np, soundfile as sf
from amphion_eval.integrations.open_audio_llm import scoring
from amphion_eval.integrations.open_audio_llm.catalog import EvaluationDataset, read_config
run, out = Path(sys.argv[1]), Path(sys.argv[2]); (out / "wav").mkdir(parents=True, exist_ok=True)
enroll_seconds = float(json.loads((run / "settings.json").read_text())["effective"]["enroll_seconds"])
samples = {s["id"]: s for s in map(json.loads, open(run / "samples.jsonl", encoding="utf-8"))}
preds = {p["id"]: p for p in map(json.loads, open(run / "predictions.jsonl", encoding="utf-8"))}
index = {r["id"]: r for r in map(json.loads, open(run / "requests.jsonl", encoding="utf-8")) if r["task"] == "ts_asr"}
groups = {"present_ov0": [], "present_ov_lt0.6": [], "absent": []}
for sid, p in sorted(preds.items()):
    s = samples.get(sid)
    if not s or p.get("task") != "ts_asr" or p.get("language") != "zh" or sid not in index: continue
    if not s["source"].startswith("ts_asr_benchmark@v4-20261022:"): continue
    lab = s["record"].get("labels") or {}; ref = scoring.tokens(p["reference"] or "", "zh"); ov = lab.get("target_overlap_ratio")
    if not ref:
        if lab.get("kind") == "neg": groups["absent"].append(sid)
    elif ov == 0: groups["present_ov0"].append(sid)
    elif ov is not None and 0 < ov < 0.6: groups["present_ov_lt0.6"].append(sid)
rng = random.Random(20261122); chosen = []
for g, ids in groups.items():
    rng.shuffle(ids); chosen += [(g, i) for i in ids[:200]]
dataset = EvaluationDataset(read_config(run / "data-config.yaml"), "evaluation"); sr = dataset.sampling_rate
with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
    for g, sid in chosen:
        req = index[sid]; assert dataset.records[req["index"]].record.id == sid
        audio = dataset.audio(req["index"]); cnt = int(round(enroll_seconds * sr))
        e = audio["enrollment"][:cnt]; e = np.pad(e, (0, cnt - len(e)))
        key = sid.replace(":", "_").replace("/", "_")
        sf.write(out / "wav" / f"{key}.enroll.wav", e.astype(np.float32), sr, subtype="FLOAT")
        sf.write(out / "wav" / f"{key}.mix.wav", audio["mixture"].astype(np.float32), sr, subtype="FLOAT")
        lab = samples[sid]["record"].get("labels") or {}
        f.write(json.dumps({"id": sid, "key": key, "group": g, "context": req["context"], "reference": preds[sid]["reference"],
                            "greedy": preds[sid].get("prediction") or "", "n_interferers": lab.get("n_interferers"),
                            "overlap": lab.get("target_overlap_ratio")}, ensure_ascii=False) + "\n")
print({g: min(200, len(v)) for g, v in groups.items()}, "available", {g: len(v) for g, v in groups.items()})
