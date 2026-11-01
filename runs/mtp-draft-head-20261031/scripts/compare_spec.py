"""Pair spec_check outputs request by request and summarise speed and acceptance.

For each mode and concurrency, every variant's raw outputs are compared with the reference
variant's at the same concurrency and at concurrency 1; the reference against itself across
concurrencies is the batch-nondeterminism floor. Writes ``compare.json`` next to the inputs.
"""
import argparse
import json
from pathlib import Path


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def read(path):
    return {r["id"]: r for r in map(json.loads, path.open())} if path.exists() else None


def pair(left, right):
    common = sorted(set(left) & set(right))
    differ = [i for i in common if left[i]["raw"] != right[i]["raw"]]
    chars = sum(len(left[i]["raw"]) for i in common)
    edits = sum(edit_distance(left[i]["raw"], right[i]["raw"]) for i in differ)
    return {"requests": len(common), "identical": len(common) - len(differ), "different": len(differ),
            "char_edits": edits, "chars": chars,
            "examples": [{"id": i, "reference": left[i]["raw"], "variant": right[i]["raw"]}
                         for i in differ[:5]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--reference", default="base")
    args = parser.parse_args()
    variants = sorted(d.name for d in args.artifacts.iterdir() if d.is_dir())
    result = {"identity": {}, "speed": {}}
    for variant in variants:
        for rows_path in sorted((args.artifacts / variant).glob("rows-*.json")):
            for row in json.loads(rows_path.read_text()):
                if "error" in row:
                    result["speed"].setdefault(variant, []).append(row)
                    continue
                spec = row["spec"]
                result["speed"].setdefault(variant, []).append({
                    **{k: row[k] for k in ("mode", "concurrency", "requests", "wall_s", "completion_tokens",
                                           "output_tokens_per_s", "audio_seconds_per_wall_second",
                                           "latency_ms_p50", "latency_ms_p95", "latency_ms_mean")},
                    "mean_acceptance_length": spec.get("mean_acceptance_length"),
                    "draft_acceptance_rate": spec.get("draft_acceptance_rate"),
                    "acceptance_rate_per_position": spec.get("acceptance_rate_per_position"),
                })
                mode, concurrency = row["mode"], row["concurrency"]
                outputs = read(args.artifacts / variant / f"requests-{mode}-c{concurrency}.jsonl")
                same = read(args.artifacts / args.reference / f"requests-{mode}-c{concurrency}.jsonl")
                single = read(args.artifacts / args.reference / f"requests-{mode}-c1.jsonl")
                key = f"{variant}/{mode}/c{concurrency}"
                result["identity"][key] = {
                    "vs_reference_same_concurrency": pair(same, outputs) if same else None,
                    "vs_reference_c1": pair(single, outputs) if single else None,
                }
    (args.artifacts / "compare.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    for key, value in result["identity"].items():
        same, single = value["vs_reference_same_concurrency"], value["vs_reference_c1"]
        print(key, "same-c:", same and f"{same['different']}/{same['requests']} differ, {same['char_edits']} edits",
              "| vs c1:", single and f"{single['different']}/{single['requests']} differ, {single['char_edits']} edits")
    for variant, rows in result["speed"].items():
        for row in rows:
            print(variant, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
                                      ensure_ascii=False))


if __name__ == "__main__":
    main()
