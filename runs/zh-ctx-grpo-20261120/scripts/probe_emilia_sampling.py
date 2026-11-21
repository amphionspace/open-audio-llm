"""Sampling probe on the Emilia "需要纠正 · explicit" items of the context-correction eval v2 (README "结果", failure
analysis): how often, in N samples at temperature 1, does a served model write every target as the context says?

A model that never gets an item right under sampling gives GRPO nothing to reinforce there; one that sometimes does
could be pushed by RL. The prompt, audio and scoring are AmphionEval's own (the run's data-config.yaml through
EvaluationDataset, prepare_audio, encode_for_server, parse_asr_output, context_correction.score_sample), so the only
difference from the eval is sampling. Run with the AmphionEval 0.14 python and the !158 source on PYTHONPATH.
usage: probe_emilia_sampling.py RUN_DIR SERVER_URL MODEL OUT_JSONL [N]   (RUN_DIR: the model's think-off ctx-v2 run)
"""
import base64
import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from amphion_eval.integrations.open_audio_llm import context_correction
from amphion_eval.integrations.open_audio_llm.catalog import EvaluationDataset, read_config
from amphion_eval.integrations.open_audio_llm.native import encode_for_server, prepare_audio
from amphion_eval.integrations.open_audio_llm.qwen3_output import parse_asr_output

MINED = "/workspace/data/datasets/context_correction_zh/versions/v2-20261116/records/mined_real.jsonl.gz"


def main():
    run, url, model, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    wanted = set()
    for line in gzip.open(MINED, "rt", encoding="utf-8"):
        r = json.loads(line)
        if (not r["metadata"]["corpus_transcript_correct"] and r["labels"]["context_correction"]["category"] == "explicit"
                and "emilia_zh" in r["id"]):
            wanted.add(r["id"])
    index = {r["id"]: r for r in map(json.loads, open(run / "requests.jsonl", encoding="utf-8")) if r["id"] in wanted}
    records = {s["id"]: s["record"] for s in map(json.loads, open(run / "samples.jsonl", encoding="utf-8")) if s["id"] in index}
    greedy = {r["id"]: r for r in map(json.loads, open(run / "context-correction-scores.jsonl", encoding="utf-8")) if r["id"] in index}
    dataset = EvaluationDataset(read_config(run / "data-config.yaml"), "evaluation")
    session = requests.Session()

    def one(item_id):
        req = index[item_id]
        assert dataset.records[req["index"]].record.id == item_id, item_id
        waveform, _ = prepare_audio(dataset, req["index"])
        audio = base64.b64encode(encode_for_server(waveform, dataset.sampling_rate)).decode("ascii")
        body = {"model": model, "n": n, "temperature": 1.0, "max_tokens": 512,
                "messages": [{"role": "system", "content": req["context"]},
                             {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}}]}]}
        response = session.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=600)
        response.raise_for_status()
        texts = [parse_asr_output(c["message"]["content"] or "")[1] for c in response.json()["choices"]]
        ok = [bool(context_correction.score_sample(records[item_id], t, task="asr")["target_ok"]) for t in texts]
        return {"id": item_id, "greedy_ok": greedy[item_id]["target_ok"], "samples_ok": sum(ok), "n": n}

    with ThreadPoolExecutor(16) as pool, open(out, "w", encoding="utf-8") as f:
        for row in pool.map(one, sorted(index)):
            f.write(json.dumps(row) + "\n")
    rows = [json.loads(l) for l in open(out)]
    for g in (False, True):
        part = [r for r in rows if r["greedy_ok"] == g]
        never = sum(r["samples_ok"] == 0 for r in part)
        some = sum(0 < r["samples_ok"] < n for r in part)
        print(f"greedy {'right' if g else 'wrong'}: {len(part)} items, never right {never}, sometimes {some}, "
              f"always {len(part) - never - some}; mean right {sum(r['samples_ok'] for r in part) / max(len(part), 1):.2f}/{n}")


if __name__ == "__main__":
    main()
