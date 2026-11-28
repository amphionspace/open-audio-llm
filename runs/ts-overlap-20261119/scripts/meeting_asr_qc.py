"""tasks/27: plain-ASR transcripts of the base audio of a ts_asr_real meeting version, for the annotation text check
(meeting_quarantine.py's rule: a run of transcript characters the annotation lacks).

Every far-field / phone segment (``-mix``) and every stored enrollment (``-enroll``, first 5 s as the TS client sends
it) goes to AmphionASR services in plain-ASR mode (empty system prompt, temperature 0), round-robin over the given base
URLs; headset and noise copies are skipped (same annotation). One JSON row per file; resumes from OUT.
The scratch original is /workspace/workspace/mingdong/scratch/ts-meeting/asr_qc.py (meeting-v1-20261117).

    python meeting_asr_qc.py VERSION OUT URL [URL ...]
"""
import base64
import gzip
import io
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

ROOT = Path("/workspace/data/datasets/ts_asr_real")
SR, CROP = 16000, 5


def main():
    version, out, urls = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    rows = [json.loads(line) for line in gzip.open(ROOT / "versions" / version / "audio-index.jsonl.gz", "rt")]
    rows = [r for r in rows if r["cut_id"].endswith(("-mix", "-enroll"))]
    done = {json.loads(line)["cut_id"] for line in open(out)} if out.exists() else set()
    jobs = [r for r in rows if r["cut_id"] not in done]
    models = {u: requests.get(u + "/v1/models", timeout=10).json()["data"][0]["id"] for u in urls}
    print(f"{len(rows)} files, {len(jobs)} pending", flush=True)
    lock = threading.Lock()

    def run(job):
        number, row = job
        url = urls[number % len(urls)]
        try:
            audio, sr = sf.read(ROOT / row["relative_path"], dtype="float32")
            assert sr == SR
            if row["cut_id"].endswith("-enroll"):
                audio = audio[: CROP * SR]
            buf = io.BytesIO()
            sf.write(buf, np.clip(audio, -1, 1), SR, format="WAV", subtype="PCM_16")
            body = {"model": models[url], "temperature": 0, "max_tokens": 512,
                    "messages": [{"role": "system", "content": ""},
                                 {"role": "user", "content": [{"type": "input_audio", "input_audio": {
                                     "data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}}]}]}
            raw = requests.post(url + "/v1/chat/completions", json=body, timeout=600).json()["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001 - logged; a rerun resumes
            print(f"failed {row['cut_id']}: {exc!r}", flush=True)
            return
        text = raw.split("<asr_text>", 1)[1] if "<asr_text>" in raw else raw
        with lock, open(out, "a") as f:
            f.write(json.dumps({"cut_id": row["cut_id"], "text": text.strip(), "raw": raw}, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(48 * len(urls)) as pool:
        for i, _ in enumerate(pool.map(run, enumerate(jobs)), 1):
            if i % 1000 == 0:
                print(f"{i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
