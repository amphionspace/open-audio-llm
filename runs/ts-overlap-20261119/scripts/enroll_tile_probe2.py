"""tasks/24 attempt 002: short-enrollment fills (enroll_fill.py) on 2,000 target-present and 2,000 target-absent v12
Chinese TS samples (random, seed 20261123) whose stored enrollment is at least 6 s, so the native 5 s needs no padding.
Per sample: native5 + {pad, noisepad, tile, tilexf, tilegap} x L in {1, 2}. 11.17 TS on [enrollment | mixture], greedy as
the eval. Also writes the variants' enrollments to WAV_DIR for the speaker-similarity check.
usage: enroll_tile_probe2.py RUN_DIR OUT_JSONL WAV_DIR URL[,URL...]   (amphion-eval python)"""
import base64, json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, requests, soundfile as sf
from amphion_eval.integrations.open_audio_llm import scoring
from amphion_eval.integrations.open_audio_llm.catalog import EvaluationDataset, read_config
from amphion_eval.integrations.open_audio_llm.native import encode_for_server
from amphion_eval.integrations.open_audio_llm.qwen3_output import parse_asr_output
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gen")); sys.path.insert(0, str(HERE))
import mixer
from enroll_fill import fills, SR, N5
run, out, wav_dir, urls = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4].split(",")
wav_dir.mkdir(parents=True, exist_ok=True)
samples = {s["id"]: s for s in map(json.loads, open(run / "samples.jsonl", encoding="utf-8"))}
preds = {p["id"]: p for p in map(json.loads, open(run / "predictions.jsonl", encoding="utf-8"))}
index = {r["id"]: r for r in map(json.loads, open(run / "requests.jsonl", encoding="utf-8")) if r["task"] == "ts_asr"}
pos, neg = [], []
for sid, s in samples.items():
    if sid not in index or s.get("language") != "zh": continue
    dur = next((x["ref"].get("duration", 0) for x in s["record"]["audio_slots"] if x["purpose"] == "enrollment"), 0)
    if dur < 6: continue
    (pos if scoring.tokens(s["reference"] or "", "zh") else neg).append(sid)
rng = random.Random(20261123); rng.shuffle(pos); rng.shuffle(neg); ids = sorted(pos[:2000]) + sorted(neg[:2000])
dataset = EvaluationDataset(read_config(run / "data-config.yaml"), "evaluation")
def build(sid):
    audio = dataset.audio(index[sid]["index"]); e = audio["enrollment"].astype(np.float32)
    v = {"native5": e[:N5]}; voiced = mixer.trim_leading_silence(e)
    for L in (1, 2):
        for k, w in fills(e, voiced[: L * SR], seed=hash((sid, L)) % 2**32).items(): v[f"{k}{L}"] = w
    return audio["mixture"].astype(np.float32), v
session = requests.Session(); lock_i = [0]
def one(sid):
    m, v = build(sid); key = sid.replace(":", "_").replace("/", "_"); rows = []
    for name, e in v.items():
        sf.write(wav_dir / f"{key}.{name}.wav", e, SR, subtype="FLOAT")
        data = base64.b64encode(encode_for_server(np.concatenate([e, m]), SR)).decode("ascii")
        body = {"model": "m1117", "temperature": 0.0, "max_tokens": 256,
                "messages": [{"role": "system", "content": index[sid]["context"]}, {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}]}]}
        url = urls[hash((sid, name)) % len(urls)]
        for attempt in range(3):
            try:
                r = session.post(url + "/v1/chat/completions", json=body, timeout=600); r.raise_for_status(); break
            except Exception:
                if attempt == 2: raise
        lab = samples[sid]["record"].get("labels") or {}
        rows.append({"id": sid, "key": key, "cond": name, "reference": samples[sid]["reference"], "text": parse_asr_output(r.json()["choices"][0]["message"]["content"] or "")[1],
                     "split": sid.rsplit(":", 1)[0], "overlap": lab.get("target_overlap_ratio"), "n_interferers": lab.get("n_interferers"), "enroll_condition": lab.get("enroll_condition")})
    return rows
with ThreadPoolExecutor(24) as pool, open(out, "w", encoding="utf-8") as f:
    for rows in pool.map(one, ids):
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(len(ids), "samples (", min(2000, len(pos)), "present,", min(2000, len(neg)), "absent of", len(pos), "/", len(neg), "eligible )")
