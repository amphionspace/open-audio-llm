"""Send prebuilt TS-ASR request WAVs (REQ_DIR/<variant>/<case id>.wav) to the 11.17 TS service exactly as
speaker-enroll does (system prompt TS_SYSTEM, temperature 0, max_tokens 512) and write <variant>.jsonl. stdlib only.
usage: send.py REQ_DIR URL VARIANT[,VARIANT...] JOBS"""
import base64, json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
PROMPT = "Transcribe only the enrolled speaker. Output nothing if they do not speak."
req_dir, url, variants, jobs = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(","), int(sys.argv[4])
def ask(p):
    body = {"model": "amphionasr-1.7b", "temperature": 0, "max_tokens": 512, "repetition_penalty": 1.0,
            "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": [{"type": "input_audio",
                "input_audio": {"data": base64.b64encode(p.read_bytes()).decode(), "format": "wav"}}]}]}
    r = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        raw = json.load(resp)["choices"][0]["message"]["content"] or ""
    _, sep, text = raw.partition("<asr_text>")
    return {"id": p.stem, "output": (text if sep else raw).strip()}
for v in variants:
    files = sorted((req_dir / v).glob("*.wav"))
    with ThreadPoolExecutor(jobs) as pool:
        rows = list(pool.map(ask, files))
    (req_dir / f"{v}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(v, len(rows), "non-empty", sum(bool(r["output"]) for r in rows), flush=True)
