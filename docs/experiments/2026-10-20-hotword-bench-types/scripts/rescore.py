"""Re-score a hotword-bench run's predictions against another version's records.

v2 and v3 send the same requests (same targets and lists), so a v2 run scored
against v3 records gives v3's per-type and per-domain breakdowns. Every row's
reference and list are checked against the records, and the re-scored
``metrics`` are compared with the run's own summary.

    python rescore.py <run dir> v3-20261020 <out.json>
"""

import glob
import gzip
import json
import os
import sys

from amphion_eval.integrations.open_audio_llm import hotword_bench as hb

ROOT = "/workspace/data/datasets/hotword_bench/versions"


def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(run, version, out):
    records = {}
    for path in glob.glob(f"{ROOT}/{version}/records/*.jsonl.gz"):
        for record in read_jsonl(path):
            records[record["id"].split(":", 1)[1]] = record  # id without the version
    scored = []
    for row in read_jsonl(os.path.join(run, "predictions.jsonl")):
        record = records[row["sample_id"].split(":", 1)[1]]
        candidates = (
            row["candidates"]
            if isinstance(row["candidates"], list)
            else json.loads(row["candidates"])
        )
        assert record["target"] == row["reference"], row["id"]
        if row["condition"] != "none":
            assert candidates == record["metadata"]["bias_lists"][row["condition"]], (
                row["id"]
            )
        scored.append(
            {
                **hb.score_sample(
                    record, row["condition"], candidates, row["prediction"]
                ),
                "label": row["label"],
            }
        )
    summary = hb.summarize(scored)
    with open(out, "w") as stream:
        json.dump(summary, stream, ensure_ascii=False)
    with open(os.path.join(run, "summary.json")) as stream:
        old = json.load(stream)["hotword_bench"]["metrics"]
    same = set(old) == set(summary["metrics"]) and all(
        old[k] == summary["metrics"][k] for k in old
    )
    print(f"{len(scored)} rows; metrics identical to the run summary: {same}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
