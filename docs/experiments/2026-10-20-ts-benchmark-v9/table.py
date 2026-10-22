"""Per-column and per-group tables of ts-asr-benchmark@v9-20261020 runs, one run per model.

Columns: CER, miss rate, false-alarm rate (from summary.json, same numbers the board shows) plus the
wrong-speaker rate of target-present columns (the output is closer to an interferer's text than to the
target's, same rule as audio-data-contract scripts/ts_benchmark/analyze.py). Group values are macro
averages over the group's columns, as on the board.

    python table.py --eval-set eval-set.yaml --run NAME=RUN_DIR ... --output table.json
"""

import argparse
import gzip
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml
from rapidfuzz.distance import Levenshtein

ROOTS = json.loads(Path("/workspace/data/registry/roots.json").read_text())


def normalize(text):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text or "").lower())


def cer(a, b):
    return Levenshtein.distance(list(a), list(b)) / max(len(b), 1)


def load_records(entries):
    records = {}
    for e in entries:
        key = f"{e['dataset_id']}@{e['version']}"
        path = (
            Path(ROOTS[e["dataset_id"]])
            / "versions"
            / e["version"]
            / "records"
            / f"{e['split']}.jsonl.gz"
        )
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                r = json.loads(line)
                records[(key, r["id"])] = r
    return records


def interferer_texts(record):
    meta = record.get("metadata") or {}
    interferers = meta.get("interferers")
    if interferers is None and "window_utts" in meta:
        by_speaker = defaultdict(list)
        for u in meta["window_utts"]:
            if u["speaker"] != meta["target_speaker"]:
                by_speaker[u["speaker"]].append(u)
        interferers = [{"utts": utts} for utts in by_speaker.values()]
    if not interferers:
        return []
    whole = [normalize("".join(u["text"] for u in t["utts"])) for t in interferers]
    parts = [normalize(u["text"]) for t in interferers for u in t["utts"]]
    return [t for t in whole + parts if t]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.eval_set.read_text())
    entries = config["evaluation"]
    by_label = {e["label"]: e for e in entries}
    records = load_records(entries)
    out = {"models": {}, "columns": {}, "groups": []}
    for item in args.run:
        name, run = item.split("=", 1)
        run = Path(run)
        info = json.loads((run / "run.json").read_text())
        metrics = json.loads((run / "summary.json").read_text())["metrics"]
        wrong = defaultdict(lambda: [0, 0])
        failures = 0
        for line in (run / "predictions.jsonl").open():
            p = json.loads(line)
            failures += p["status"] != "success"
            dataset = p["source"].split(":", 1)[0]
            record = records[(dataset, p["id"])]
            if not record["labels"]["target_present"]:
                continue
            hyp, ref = normalize(p.get("prediction")), normalize(record["target"])
            others = interferer_texts(record)
            split = p["source"].split(":", 1)[1].split("/", 1)[0]
            if hyp and others:
                wrong[(dataset, split)][0] += min(cer(hyp, o) for o in others) < cer(
                    hyp, ref
                )
                wrong[(dataset, split)][1] += 1
        columns = {}
        for label, e in by_label.items():
            dataset = f"{e['dataset_id']}@{e['version']}"
            m = metrics[f"{dataset}:{e['split']}/ts_asr/zh"]
            k, n = wrong.get((dataset, e["split"]), (0, 0))
            columns[label] = {
                "n": m["utterances"],
                "errors": m.get("errors"),
                "units": m.get("reference_units"),
                "cer": None if m.get("error_rate") is None else 100 * m["error_rate"],
                "miss": None if m.get("miss_rate") is None else 100 * m["miss_rate"],
                "fa": None
                if m.get("false_alarm_rate") is None
                else 100 * m["false_alarm_rate"],
                "wrong": 100 * k / n if n else None,
            }
        out["models"][name] = {
            "run_id": info["run_id"],
            "wandb": info.get("wandb_url"),
            "state": info.get("state"),
            "failures": failures,
            "samples": sum(c["n"] for c in columns.values()),
        }
        for label, cell in columns.items():
            out["columns"].setdefault(label, {})[name] = cell
    for group in config["groups"]:
        row = {
            "section": group.get("section"),
            "name": group["name"],
            "labels": group["labels"],
            "values": {},
        }
        for name in out["models"]:
            cells = [out["columns"][label][name] for label in group["labels"]]
            row["values"][name] = {
                metric: (sum(vals) / len(vals) if vals else None)
                for metric in ("cer", "miss", "fa", "wrong")
                for vals in [[c[metric] for c in cells if c[metric] is not None]]
            }
        out["groups"].append(row)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["models"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
