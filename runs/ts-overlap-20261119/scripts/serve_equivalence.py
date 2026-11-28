"""tasks/11 serving check: send the same TS-ASR requests of an AmphionEval run to several served models, one request
at a time (no batching differences), greedy, and report how often their outputs are identical.

Requests are built like the eval (the run's data-config.yaml through EvaluationDataset, prepare_audio with the run's
enroll_seconds, encode_for_server); every STEP-th TS request of the run is used.
usage: serve_equivalence.py RUN_DIR STEP OUT_JSONL NAME=URL=MODEL [NAME=URL=MODEL ...]
Run with the AmphionEval python.
"""
import base64
import itertools
import json
import sys
from pathlib import Path

import requests

from amphion_eval.integrations.open_audio_llm.catalog import EvaluationDataset, read_config
from amphion_eval.integrations.open_audio_llm.native import encode_for_server, prepare_audio


def main():
    run, step, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    servers = [spec.split("=", 2) for spec in sys.argv[4:]]
    reqs = [r for r in map(json.loads, open(run / "requests.jsonl", encoding="utf-8")) if r["task"] == "ts_asr"][::step]
    dataset = EvaluationDataset(read_config(run / "data-config.yaml"), "evaluation")
    # The run's own enrollment split (prepare_audio's default is the 3-second legacy one).
    enroll_seconds = float(json.loads((run / "settings.json").read_text())["effective"]["enroll_seconds"])
    session = requests.Session()
    with open(out, "w", encoding="utf-8") as f:
        for req in reqs:
            waveform, _ = prepare_audio(dataset, req["index"], enroll_seconds=enroll_seconds)
            audio = base64.b64encode(encode_for_server(waveform, dataset.sampling_rate)).decode("ascii")
            row = {"id": req["id"], "outputs": {}}
            for name, url, model in servers:
                body = {"model": model, "temperature": 0.0, "max_tokens": 512, "seed": 42,
                        "messages": [{"role": "system", "content": req["context"]},
                                     {"role": "user", "content": [{"type": "input_audio",
                                                                   "input_audio": {"data": audio, "format": "wav"}}]}]}
                response = session.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=600)
                response.raise_for_status()
                row["outputs"][name] = response.json()["choices"][0]["message"]["content"] or ""
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    names = [name for name, _, _ in servers]
    summary = {"requests": len(rows), "identical": {}}
    for a, b in itertools.combinations(names, 2):
        same = sum(r["outputs"][a] == r["outputs"][b] for r in rows)
        summary["identical"][f"{a}|{b}"] = same
        print(f"{a} vs {b}: identical {same}/{len(rows)}")
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
