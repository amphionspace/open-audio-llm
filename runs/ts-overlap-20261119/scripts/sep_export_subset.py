"""Export v12 zh target-present, overlap >= 0.9 samples (N per interferer count, fixed seed): enrollment cut/padded to the
run's enroll_seconds, mixture, as 16 kHz float wav; manifest with reference, 11.17 greedy prediction and labels."""
import json, random, sys
from pathlib import Path
import numpy as np, soundfile as sf
from amphion_eval.integrations.open_audio_llm import scoring
from amphion_eval.integrations.open_audio_llm.catalog import EvaluationDataset, read_config
run, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
(out / "wav").mkdir(parents=True, exist_ok=True)
enroll_seconds = float(json.loads((run / "settings.json").read_text())["effective"]["enroll_seconds"])
samples = {s["id"]: s for s in map(json.loads, open(run / "samples.jsonl", encoding="utf-8"))}
preds = {p["id"]: p for p in map(json.loads, open(run / "predictions.jsonl", encoding="utf-8"))}
index = {r["id"]: r for r in map(json.loads, open(run / "requests.jsonl", encoding="utf-8")) if r["task"] == "ts_asr"}
groups = {}
for sid, p in sorted(preds.items()):
    lab = (samples[sid]["record"].get("labels") or {}) if sid in samples else {}
    if p.get("task") != "ts_asr" or p.get("language") != "zh" or not scoring.tokens(p["reference"] or "", "zh"): continue
    if (lab.get("target_overlap_ratio") or 0) < 0.9 or sid not in index: continue
    groups.setdefault(lab.get("n_interferers"), []).append(sid)
rng = random.Random(20261122)
chosen = []
for k in (1, 2):
    ids = groups[k]; rng.shuffle(ids); chosen += ids[:n]
dataset = EvaluationDataset(read_config(run / "data-config.yaml"), "evaluation")
sr = dataset.sampling_rate
with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
    for sid in chosen:
        req = index[sid]; assert dataset.records[req["index"]].record.id == sid
        audio = dataset.audio(req["index"])
        cnt = int(round(enroll_seconds * sr)); e = audio["enrollment"][:cnt]; e = np.pad(e, (0, cnt - len(e)))
        key = sid.replace(":", "_").replace("/", "_")
        sf.write(out / "wav" / f"{key}.enroll.wav", e.astype(np.float32), sr, subtype="FLOAT")
        sf.write(out / "wav" / f"{key}.mix.wav", audio["mixture"].astype(np.float32), sr, subtype="FLOAT")
        lab = samples[sid]["record"].get("labels") or {}
        f.write(json.dumps({"id": sid, "key": key, "context": req["context"], "reference": preds[sid]["reference"],
                            "greedy": preds[sid].get("prediction") or "", "n_interferers": lab.get("n_interferers"),
                            "interferer_gain_db": lab.get("interferer_gain_db"), "overlap": lab.get("target_overlap_ratio"),
                            "sr": sr}, ensure_ascii=False) + "\n")
print(len(chosen), "exported, sr", sr, "enroll_seconds", enroll_seconds)
