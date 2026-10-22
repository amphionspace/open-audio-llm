"""Markdown tables for the v9 README from the JSON written by table.py.

Cells are "model1 / model2 / ..." in the order the runs were given. Groups are macro averages over
their columns, as on the board, except 业务回归, whose columns hold 1-28 samples each: those are
pooled over samples (counts given).
"""

import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
models = list(data["models"])
METRICS = ("cer", "miss", "wrong", "fa")


def fmt(values):
    if all(v is None for v in values):
        return ""
    return " / ".join("–" if v is None else f"{v:.0f}" for v in values)


def pooled(labels):
    """Rates pooled over the samples of the given columns (CER = all errors / all reference characters)."""
    out = {}
    for m in models:
        cells = [data["columns"][label][m] for label in labels]
        row = {}
        for k in METRICS:
            pairs = [(c[k], c["n"]) for c in cells if c[k] is not None]
            n = sum(w for _, w in pairs)
            row[k] = sum(v * w for v, w in pairs) / n if n else None
        units = sum(c["units"] or 0 for c in cells if c["cer"] is not None)
        row["cer"] = (
            100 * sum(c["errors"] for c in cells if c["cer"] is not None) / units
            if units
            else None
        )
        out[m] = row
    return out


print("| 子集 | 分组 | 字错率 | 漏识 | 转写了别人 | 误输出 |")
print("| --- | --- | --- | --- | --- | --- |")
section = None
for g in data["groups"]:
    values = g["values"]
    name = g["name"]
    if g["section"] == "业务回归":
        values = pooled(g["labels"])
        n = sum(data["columns"][label][models[0]]["n"] for label in g["labels"])
        name += f"（共 {n} 条，按条合计）"
    cells = [fmt([values[m][k] for m in models]) for k in METRICS]
    shown = g["section"] if g["section"] != section else ""
    section = g["section"]
    print(f"| {shown} | {name} | " + " | ".join(cells) + " |")

print()
print("新增 10 列：")
print()
print("| 列 | 条数 | 字错率 | 漏识 | 转写了别人 | 误输出 |")
print("| --- | --- | --- | --- | --- | --- |")
for label in [
    "abs-rand-k1",
    "abs-hard-k1",
    "realreg3-neg",
    "realreg3-hard-neg",
    "realreg3-music",
    "realreg3-mid1-neg",
    "realreg3-mid2-neg",
    "handover-same-open",
    "handover-rand",
    "handover-rev-same-open",
]:
    cells = data["columns"][label]
    print(
        f"| `{label}` | {cells[models[0]]['n']} | "
        + " | ".join(fmt([cells[m][k] for m in models]) for k in METRICS)
        + " |"
    )

print()
print("| 模型 | 运行号 | 条数 | 失败 | W&B |")
print("| --- | --- | --- | --- | --- |")
for m, info in data["models"].items():
    print(
        f"| {m} | `{info['run_id']}` | {info['samples']:,} | {info['failures']} | [{info['run_id'][-8:]}]({info['wandb']}) |"
    )
