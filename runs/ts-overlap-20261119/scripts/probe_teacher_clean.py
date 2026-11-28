"""11.17 TS (greedy, as the eval) on [5 s enrollment | clean] of every generated scene, for the punctuation of the
fine-tune labels. usage: teacher_clean.py MANIFEST WAV_DIR OUT_JSONL URL[,URL...] SHARD NUM_SHARDS   (amphion-eval python)"""
import base64, json, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests, soundfile as sf
from amphion_eval.integrations.open_audio_llm.native import encode_for_server
from amphion_eval.integrations.open_audio_llm.qwen3_output import parse_asr_output
manifest, wav, out, urls = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
shard, shards = int(sys.argv[5]), int(sys.argv[6])
SYSTEM = "Transcribe only the enrolled speaker. Output nothing if they do not speak."
rows = [json.loads(l) for l in open(manifest, encoding="utf-8")][shard::shards]
session = requests.Session()
def one(i):
    r = rows[i]
    e = sf.read(f"{wav}/{r['key']}.enroll.wav", dtype="float32")[0]; c = sf.read(f"{wav}/{r['key']}.clean.wav", dtype="float32")[0]
    audio = base64.b64encode(encode_for_server(np.concatenate([e, c]), 16000)).decode("ascii")
    body = {"model": "m1117", "temperature": 0.0, "max_tokens": 256,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}}]}]}
    for attempt in range(3):
        try:
            resp = session.post(urls[i % len(urls)] + "/v1/chat/completions", json=body, timeout=600); resp.raise_for_status(); break
        except Exception:
            if attempt == 2: return {"key": r["key"], "teacher": None}
    return {"key": r["key"], "teacher": parse_asr_output(resp.json()["choices"][0]["message"]["content"] or "")[1]}
with ThreadPoolExecutor(48) as pool, open(out, "w", encoding="utf-8") as f:
    for row in pool.map(one, range(len(rows))): f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("done", len(rows))
