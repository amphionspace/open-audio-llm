"""Compare AmphionEval v12 runs against the 10.28 baseline: aggregates plus per-item presence flips.

Aggregates come from each run's summary.json (scoring protocol v2): CER pooled over target-present
splits, miss rate over target-present utterances, false-alarm rate over target-absent utterances.
Presence flips count items whose prediction switches between empty and non-empty versus the
baseline; the noise reference is a rerun of the same weights on the delivery-pack server.
"""
import argparse
import json
from pathlib import Path


def aggregate(run):
    summary = json.loads((run / "summary.json").read_text())
    if summary["protocol"].get("version") != 2:
        raise ValueError(f"{run}: expected scoring protocol v2")
    errors = units = pos = missed = neg = alarms = 0.0
    for row in summary["metrics"].values():
        if row.get("reference_units"):
            errors += row["errors"]
            units += row["reference_units"]
        if row.get("miss_rate") is not None:
            count = row["utterances"] - (row.get("negative_utterances") or 0)
            pos += count
            missed += row["miss_rate"] * count
        if row.get("false_alarm_rate") is not None:
            neg += row["negative_utterances"]
            alarms += row["false_alarm_rate"] * row["negative_utterances"]
    return {"cer": errors / units, "miss_rate": missed / pos, "false_alarm_rate": alarms / neg,
            "present_utterances": int(pos), "absent_utterances": int(neg)}


def presence(run):
    out = {}
    for line in open(run / "predictions.jsonl"):
        row = json.loads(line)
        out[row["id"]] = bool((row.get("prediction") or "").strip())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True, help="same weights, different server run")
    parser.add_argument("--candidate", action="append", default=[], help="name=run_dir")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = presence(args.baseline)
    rows = {"baseline": {**aggregate(args.baseline), "run": str(args.baseline)}}
    for name, run in [("noise-rerun", str(args.noise))] + [c.split("=", 1) for c in args.candidate]:
        run = Path(run)
        other = presence(run)
        shared = base.keys() & other.keys()
        rows[name] = {**aggregate(run), "run": str(run), "items": len(shared),
                      "presence_flips": sum(base[i] != other[i] for i in shared),
                      "to_empty": sum(base[i] and not other[i] for i in shared),
                      "to_text": sum(other[i] and not base[i] for i in shared)}
    args.output.write_text(json.dumps(rows, indent=2))
    for name, row in rows.items():
        print(f"{name:14s} CER {row['cer']*100:6.2f}  miss {row['miss_rate']*100:5.2f}  FA {row['false_alarm_rate']*100:5.2f}"
              + (f"  flips {row['presence_flips']} (->empty {row['to_empty']}, ->text {row['to_text']})" if 'presence_flips' in row else ""))


if __name__ == "__main__":
    main()
