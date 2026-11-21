"""tasks/10: GRPO headroom on heavy-overlap Chinese TS (v12), sampling only, no training.

For every target-present zh TS sample of an AmphionEval v12 run with target overlap >= MIN_OVERLAP, draw N samples at
temperature 1 from a served model and score each with the retrospective's in-utterance error (decompose_errors.py:
AmphionEval zh tokens, (sub + del + ins) / reference length; an empty output counts as all deleted). Greedy is the
run's own prediction. A group whose samples all score the same gives GRPO nothing to learn from; a greedy error that
no sample beats cannot be reinforced (the context-correction probe found 58% of its failures there).
Requests are built like the eval (prepare_audio with the run's enroll_seconds, encode_for_server, parse_asr_output).
usage: probe_ts_sampling.py RUN_DIR URL MODEL OUT_JSONL [N] [MIN_OVERLAP] [SHARD] [NUM_SHARDS]
Run with the AmphionEval python.
"""
import base64
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from amphion_eval.integrations.open_audio_llm import scoring
from amphion_eval.integrations.open_audio_llm.catalog import EvaluationDataset, read_config
from amphion_eval.integrations.open_audio_llm.native import encode_for_server, prepare_audio
from amphion_eval.integrations.open_audio_llm.qwen3_output import parse_asr_output


def edits(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j - 1] + (r != h), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]


def error(ref_tokens, text):
    return edits(ref_tokens, scoring.tokens(text or "", "zh")) / len(ref_tokens)


def main():
    run, url, model, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    min_overlap = float(sys.argv[6]) if len(sys.argv) > 6 else 0.9
    shard, shards = (int(sys.argv[7]), int(sys.argv[8])) if len(sys.argv) > 8 else (0, 1)
    enroll_seconds = float(json.loads((run / "settings.json").read_text())["effective"]["enroll_seconds"])
    samples = {s["id"]: s for s in map(json.loads, open(run / "samples.jsonl", encoding="utf-8"))}
    preds = {p["id"]: p for p in map(json.loads, open(run / "predictions.jsonl", encoding="utf-8"))}
    index = {r["id"]: r for r in map(json.loads, open(run / "requests.jsonl", encoding="utf-8")) if r["task"] == "ts_asr"}
    wanted = []
    for sid, p in sorted(preds.items()):
        labels = (samples[sid]["record"].get("labels") or {}) if sid in samples else {}
        ratio = labels.get("target_overlap_ratio")
        if p.get("task") == "ts_asr" and p.get("language") == "zh" and scoring.tokens(p["reference"] or "", "zh") \
                and ratio is not None and ratio >= min_overlap and sid in index:
            wanted.append(sid)
    wanted = wanted[shard::shards]
    dataset = EvaluationDataset(read_config(run / "data-config.yaml"), "evaluation")
    session = requests.Session()

    def one(sid):
        req = index[sid]
        assert dataset.records[req["index"]].record.id == sid, sid
        waveform, _ = prepare_audio(dataset, req["index"], enroll_seconds=enroll_seconds)
        audio = base64.b64encode(encode_for_server(waveform, dataset.sampling_rate)).decode("ascii")
        body = {"model": model, "n": n, "temperature": 1.0, "max_tokens": 512,
                "messages": [{"role": "system", "content": req["context"]},
                             {"role": "user", "content": [{"type": "input_audio",
                                                           "input_audio": {"data": audio, "format": "wav"}}]}]}
        response = session.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=900)
        response.raise_for_status()
        texts = [parse_asr_output(c["message"]["content"] or "")[1] for c in response.json()["choices"]]
        ref = scoring.tokens(preds[sid]["reference"], "zh")
        return {"id": sid, "ref_len": len(ref), "greedy": round(error(ref, preds[sid].get("prediction")), 5),
                "samples": [round(error(ref, t), 5) for t in texts], "empty": sum(not (t or "").strip() for t in texts)}

    with ThreadPoolExecutor(16) as pool, open(out, "w", encoding="utf-8") as f:
        for row in pool.map(one, wanted):
            f.write(json.dumps(row) + "\n")
    print(f"{len(wanted)} samples written to {out}")


if __name__ == "__main__":
    main()
