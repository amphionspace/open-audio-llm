"""11.17 TS mode (greedy, temperature 0, max_tokens 256 as the eval) on [5 s enrollment | candidate] for every exported
sample: candidate 'orig' = the original mixture (reproduces the eval), the rest = separated streams. Records text and
mean token logprob. usage: asr_probe.py SUBSET_DIR OUT_JSONL URL[,URL...]   (amphion-eval python)"""
import base64, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, requests, soundfile as sf
from amphion_eval.integrations.open_audio_llm.native import encode_for_server
from amphion_eval.integrations.open_audio_llm.qwen3_output import parse_asr_output
sub, out, urls = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
rows = [json.loads(l) for l in open(sub / "manifest.jsonl", encoding="utf-8")]
jobs = []
for r in rows:
    jobs.append((r, "orig", sub / "wav" / f"{r['key']}.mix.wav"))
    for p in sorted((sub / "sep").glob(f"{r['key']}.*.wav")):
        jobs.append((r, p.name[len(r["key"]) + 1:-4], p))
session = requests.Session()
def one(i):
    r, name, path = jobs[i]
    enroll, _ = sf.read(sub / "wav" / f"{r['key']}.enroll.wav", dtype="float32")
    cand, _ = sf.read(path, dtype="float32")
    audio = base64.b64encode(encode_for_server(np.concatenate([enroll, cand]), 16000)).decode("ascii")
    body = {"model": "m1117", "temperature": 0.0, "max_tokens": 256, "logprobs": True,
            "messages": [{"role": "system", "content": r["context"]},
                         {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}}]}]}
    resp = session.post(urls[i % len(urls)].rstrip("/") + "/v1/chat/completions", json=body, timeout=900)
    resp.raise_for_status()
    c = resp.json()["choices"][0]
    lps = [t["logprob"] for t in ((c.get("logprobs") or {}).get("content") or [])]
    return {"id": r["id"], "cand": name, "text": parse_asr_output(c["message"]["content"] or "")[1],
            "mean_logprob": float(np.mean(lps)) if lps else None, "n_tok": len(lps)}
with ThreadPoolExecutor(32) as pool, open(out, "w", encoding="utf-8") as f:
    for row in pool.map(one, range(len(jobs))):
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(len(jobs), "requests written to", out)
