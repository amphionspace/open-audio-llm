"""Short enrollment: zero-padded vs repeated to 5 s, against the native 5 s, on the v12 enrollment-condition scenes
(enroll-clean-pos / -neg: one scene per index, a 10 s stored enrollment). Variants of the same enrollment utterance:
native5 = first 5 s; padL / tileL (L = 1, 2, 3) = the first L s after trimming leading silence (generator's
trim_leading_silence), then zero-padded or repeated to 5 s. 11.17 TS on [variant | mixture], greedy as the eval.
usage: enroll_tile_probe.py RUN_DIR OUT_JSONL URL[,URL...]   (amphion-eval python)"""
import base64, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, requests
from amphion_eval.integrations.open_audio_llm.catalog import EvaluationDataset, read_config
from amphion_eval.integrations.open_audio_llm.native import encode_for_server
from amphion_eval.integrations.open_audio_llm.qwen3_output import parse_asr_output
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gen"))
import mixer
run, out, urls = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
SR = 16000; N5 = 5 * SR
samples = {s["id"]: s for s in map(json.loads, open(run / "samples.jsonl", encoding="utf-8"))}
index = {r["id"]: r for r in map(json.loads, open(run / "requests.jsonl", encoding="utf-8")) if r["task"] == "ts_asr"}
ids = sorted(i for i in index if i.startswith(("enroll-clean-pos:", "enroll-clean-neg:")))
dataset = EvaluationDataset(read_config(run / "data-config.yaml"), "evaluation")
def variants(e):
    v = {"native5": np.pad(e[:N5], (0, max(0, N5 - len(e))))}
    voiced = mixer.trim_leading_silence(e)
    for L in (1, 2, 3):
        cut = voiced[: L * SR]
        v[f"pad{L}"] = np.pad(cut, (0, N5 - len(cut)))
        v[f"tile{L}"] = np.tile(cut, -(-N5 // len(cut)))[:N5]
    return v
jobs = []
for sid in ids:
    req = index[sid]; audio = dataset.audio(req["index"])
    for name, e in variants(audio["enrollment"].astype(np.float32)).items():
        jobs.append((sid, name, req["context"], e.astype(np.float32), audio["mixture"].astype(np.float32)))
session = requests.Session()
def one(i):
    sid, name, ctx, e, m = jobs[i]
    data = base64.b64encode(encode_for_server(np.concatenate([e, m]), SR)).decode("ascii")
    body = {"model": "m1117", "temperature": 0.0, "max_tokens": 256,
            "messages": [{"role": "system", "content": ctx}, {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}]}]}
    r = session.post(urls[i % len(urls)] + "/v1/chat/completions", json=body, timeout=600); r.raise_for_status()
    return {"id": sid, "cond": name, "reference": samples[sid]["reference"], "text": parse_asr_output(r.json()["choices"][0]["message"]["content"] or "")[1]}
with ThreadPoolExecutor(32) as pool, open(out, "w", encoding="utf-8") as f:
    for row in pool.map(one, range(len(jobs))): f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(len(ids), "scenes,", len(jobs), "requests")
