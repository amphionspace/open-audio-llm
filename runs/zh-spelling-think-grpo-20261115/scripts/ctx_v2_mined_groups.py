"""09-ctx-v2-eval: real-recording accuracy split by whether the corpus transcript needed a correction, from ae's per-item scores.

usage: ctx_v2_mined_groups.py <label>=<run dir> [...]   (first run is the reference for new right / new wrong)
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

VERSIONS = {"v1-20261018": "/workspace/data/datasets/context_correction_zh/versions/v1-20261018/records/mined_real.jsonl.gz",
            "v2-20261116": "/workspace/data/datasets/context_correction_zh/versions/v2-20261116/records/mined_real.jsonl.gz"}
meta = {}
for path in VERSIONS.values():
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        cc = r["labels"]["context_correction"]
        meta[r["id"]] = {"need": not r["metadata"]["corpus_transcript_correct"], "category": cc["category"],
                         "overlap": cc.get("train_overlap")}
runs = []
for arg in sys.argv[1:]:
    label, run = arg.split("=", 1)
    ok = {}
    for line in open(Path(run) / "context-correction-scores.jsonl"):
        s = json.loads(line)
        if s["kind"] == "mined":
            ok[s["id"]] = (s["source"], s["target_ok"])
    runs.append((label, ok))
groups = defaultdict(list)
for item, (source, _) in runs[0][1].items():
    m = meta[item]
    name = source.split(":")[1]
    for key in [(name, "all"), (name, "需要纠正" if m["need"] else "原转写已对"),
                (name, ("需要纠正" if m["need"] else "原转写已对") + " · " + m["category"])]:
        groups[key].append(item)
    if m["overlap"]:
        groups[(name, ("需要纠正" if m["need"] else "原转写已对") + " · " + m["overlap"])].append(item)
ref_label, ref = runs[0]
print("| 来源 | 组 | 条数 | " + " | ".join(label for label, _ in runs) + " | " +
      " | ".join(f"{label} 比 {ref_label}：新对 / 新错" for label, _ in runs[1:]) + " |")
print("|---|---|---|" + "---|" * (2 * len(runs) - 1))
for (name, group), items in sorted(groups.items()):
    cells = [f"{100 * sum(ok[i][1] for i in items) / len(items):.1f}" for _, ok in runs]
    flips = [f"{sum(ok[i][1] and not ref[i][1] for i in items)} / {sum(ref[i][1] and not ok[i][1] for i in items)}"
             for _, ok in runs[1:]]
    print(f"| {name} | {group} | {len(items)} | " + " | ".join(cells) + " | " + " | ".join(flips) + " |")
