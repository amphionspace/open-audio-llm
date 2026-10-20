#!/usr/bin/env python3
"""Build the README comparison table from AmphionEval summary.json files (per-set and pooled CER)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RUNS = {
    "A0 34479 全局": "/workspace/data/ae-runs/checkpoint-34479/runs/run-20261015T164311Z-bd22c9ff",
    "A1 34479 1s": "/workspace/data/ae-runs/streaming-s0-34479-win1s/runs/{a1}",
    "B0 Qwen3 8s": "/workspace/data/ae-runs/qwen3-asr-1.7b/runs/run-20261015T164311Z-19553ee4",
    "B1 Qwen3 1s": "/workspace/data/ae-runs/streaming-s0-qwen3-asr-win1s/runs/{b1}",
    "X2 流式": "/workspace/data/ae-runs/x2streaming-asr-4b-1009/runs/{x2}",
}
POLICE_GROUPS = {"指令识别", "术语与业务", "口音合成"}


def source_prefixes():
    """Map each kept label to the summary-key prefix of its source, split into public zh and police."""
    cfg = yaml.safe_load((HERE / "x2streaming" / "asr-benchmark-zh.yaml").read_text(encoding="utf-8"))
    police = {label for g in cfg["groups"] if g["name"] in POLICE_GROUPS for label in g["labels"]}
    return [
        (e["label"], f'{e["dataset_id"]}@{e["version"]}:{e["split"]}/', e["label"] in police)
        for e in cfg["evaluation"]
    ]


def main() -> None:
    a1, b1, x2 = sys.argv[1:4]
    runs = {name: Path(path.format(a1=a1, b1=b1, x2=x2)) for name, path in RUNS.items()}
    metrics = {name: json.loads((path / "summary.json").read_text())["metrics"] for name, path in runs.items()}
    prefixes = source_prefixes()
    names = list(runs)
    print("| 测试集 | n | " + " | ".join(names) + " |")
    print("|---|---:|" + "---:|" * len(names))
    pooled = {name: {"public": [0, 0], "police": [0, 0]} for name in names}
    for label, prefix, is_police in prefixes:
        keys = sorted(k for k in metrics[names[0]] if k.startswith(prefix))
        for key in keys:
            row = []
            for name in names:
                m = metrics[name].get(key)
                if m is None:
                    row.append("—")
                    continue
                bucket = pooled[name]["police" if is_police else "public"]
                bucket[0] += m["errors"]
                bucket[1] += m["reference_units"]
                row.append(f'{m["error_rate"] * 100:.2f}')
            n = metrics[names[0]][key]["utterances"]
            print(f"| {key.split('/asr/')[0]} {key.split('/asr/')[1]} | {n} | " + " | ".join(row) + " |")
    for title, parts in (("公开中文合并", ["public"]), ("警务合并", ["police"]), ("中文+警务合并", ["public", "police"])):
        cells = []
        for name in names:
            errors = sum(pooled[name][p][0] for p in parts)
            units = sum(pooled[name][p][1] for p in parts)
            cells.append(f"**{errors / units * 100:.2f}**" if units else "—")
        print(f"| **{title}** | | " + " | ".join(cells) + " |")
    for name in names:
        print(f"{name}: records_sha256={json.loads((runs[name] / 'summary.json').read_text())['protocol']['records_sha256'][:8]}", file=sys.stderr)


if __name__ == "__main__":
    main()
