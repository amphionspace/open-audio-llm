"""Dialogue statistics of synthesized training windows, next to the real meetings whose timelines they reuse.

Draws a fixed random sample of window records of one split, plans each one the way training does
(CatalogSwiftDataset._plan_meeting; no audio), and summarizes the targets with label_stats (speakers, overlap,
turn length, back-channels per minute, fillers...). Adds embedded turns per minute: turns of more than 2 units that
lie entirely inside another speaker's turn (it starts before and ends after) -- 54% of the real heavily overlapped
dev turns, 0% of the old synthesizer's (../zh-diar-sim-real-gap-20261031, section 4c)."""
import argparse
import json
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_stats as ls  # noqa: E402


def embedded_per_minute(target, duration):
    turns = []
    for line in target.splitlines():
        m = ls.TURN.match(line.strip())
        if m:
            turns.append((m[1], float(m[2]), float(m[3]), "".join(ls.UNIT.findall(m[4]))))
    inside = sum(1 for s, a, b, text in turns if len(text) > 2
                 and any(o != s and oa < a and ob > b for o, oa, ob, _ in turns))
    return inside / (duration / 60)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--split", default="train_120s")
    p.add_argument("--sample", type=int, default=400)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--real", action="append", default=[], metavar="NAME=RECORDS.jsonl.gz",
                   help="real timestamped meeting records summarized the same way")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from open_audio_llm.data.catalog_dataset import CatalogSwiftDataset

    data = yaml.safe_load(args.data_config.read_text())
    online = data["online_meeting"]
    source = {"dataset_id": online["material"]["dataset_id"], "version": online["material"]["version"],
              "split": args.split, "online_meeting": online}
    dataset = CatalogSwiftDataset({"catalog": data["catalog"], "roots": data["roots"], "seed": 1,
                                   "sampling_rate": 16000, "train": [source]}, message_format="qwen3_asr")
    rng = random.Random(args.seed)
    rows, embedded = {}, {}
    for index in rng.sample(range(len(dataset.records)), args.sample):
        record = dataset.records[index].record
        corpus = record.labels["template_corpus"]
        duration = record.slot("mixture").ref.duration
        planned = dataset._plan_meeting(record, random.Random(f"{args.seed}:{index}"))
        row = ls.window(planned.target, duration)
        if row:
            rows.setdefault(corpus, []).append(row)
            embedded.setdefault(corpus, []).append(embedded_per_minute(planned.target, duration))
    report = {f"synthetic/{corpus}": {**ls.summarize(r),
                                      "embedded_per_min": round(sum(embedded[corpus]) / len(embedded[corpus]), 2)}
              for corpus, r in sorted(rows.items())}
    import gzip

    for item in args.real:
        name, path = item.split("=", 1)
        records = [json.loads(line) for line in gzip.open(path, "rt") if line.strip()]
        real, inside = [], []
        for record in random.Random(args.seed).sample(records, min(args.sample, len(records))):
            duration = record["audio_slots"][0]["ref"]["duration"]
            target = ls.target_of(record)
            row = ls.window(target, duration)
            if row:
                real.append(row)
                inside.append(embedded_per_minute(target, duration))
        report[f"real/{name}"] = {**ls.summarize(real), "embedded_per_min": round(sum(inside) / len(inside), 2)}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
