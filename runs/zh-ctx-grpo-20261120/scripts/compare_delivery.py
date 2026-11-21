"""Delivery evaluation (README "交付评估"): avg3 against 11.17 on the 27 suites of make_delivery_configs.py, both run on
the same node and client. ts-asr suites: paired bootstrap (ae open-audio-llm compare, AmphionEval !158), every overall
metric with its 95% CI; an error-type metric is worse when its CI low > 0, a hit rate when its CI high < 0. Hotword
suites (not covered by compare): pooled CER and recall as the delivery packages report them (08's hotword_pool.py).
Writes tasks/04-delivery/compare/cmp-<suite>.json and tasks/04-delivery/delivery.md.
usage: python3 compare_delivery.py
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_delivery_configs import SUITES, run_dir  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "tasks/04-delivery"
RUNS = Path("/workspace/data/ae-runs")
PY = "/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python"
CMP_SRC = "/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src"
MODELS = ("r1117", "avg3")  # baseline, candidate
HIGHER_BETTER = {"term_hit_rate", "recall"}


def identity(config):
    return config.get("eval_set") or str(config.get("data_config"))


def latest(model, ident):
    found = []
    for run in sorted((RUNS / f"ctx-grpo-20261120-delivery-{model}/runs").glob("run-*")):
        c = json.loads((run / "run.json").read_text())["config"]
        cov = json.loads((run / "coverage.json").read_text()) if (run / "coverage.json").exists() else {}
        if identity(c) == ident and cov.get("complete") and cov.get("successful") == cov.get("total"):
            found.append(run)
    if not found:
        raise SystemExit(f"no complete run of {model} for {ident}")
    return found[-1]


def pooled(run):
    metrics = json.loads((run / "summary.json").read_text())["hotword_bench"]["metrics"]
    e = sum(g.get("errors") or 0 for g in metrics.values()); u = sum(g.get("units") or 0 for g in metrics.values())
    listed = [g for k, g in metrics.items() if not k.endswith("/none")]
    h = sum(g.get("hits") or 0 for g in listed); t = sum(g.get("gold_total") or 0 for g in listed)
    return e / u, (h / t if t else None)


def one(item):
    key, suffix, _ = item
    base = json.loads((run_dir(suffix) / "run.json").read_text())["config"]
    a, b = (latest(m, identity(base)) for m in MODELS)
    if base.get("evaluator") == "hotwords":
        (ca, ra), (cb, rb) = pooled(a), pooled(b)
        cells = [f"CER {100 * ca:.2f} → {100 * cb:.2f}（{100 * (cb - ca):+.2f}）"]
        if ra is not None:
            cells.append(f"召回 {100 * ra:.1f} → {100 * rb:.1f}（{100 * (rb - ra):+.1f}）")
        return key, cells, None
    out = OUT / "compare" / f"cmp-{key}.json"
    with open(out.with_suffix(".log"), "w") as log:
        subprocess.run([PY, "-m", "amphion_eval.cli", "open-audio-llm", "compare", str(a), str(b), "--output", str(out)],
                       env={"PYTHONPATH": CMP_SRC, "PATH": "/usr/bin:/bin"}, stdout=subprocess.DEVNULL, stderr=log, check=True)
    overall = json.loads(out.read_text())["overall"]
    cells, worse = [], []
    for group, metrics in overall.items():
        for name, x in metrics.items():
            if x.get("a") is None:
                continue
            bad = x["ci"][1] < 0 if name in HIGHER_BETTER else x["ci"][0] > 0
            cells.append(f"{group} {name} {100 * x['a']:.2f} → {100 * x['b']:.2f}，{100 * x['diff']:+.2f} "
                         f"[{100 * x['ci'][0]:+.2f}, {100 * x['ci'][1]:+.2f}]{'*' if x['significant'] else ''}{' **更差**' if bad else ''}")
            if bad:
                worse.append(f"{group} {name}")
    return key, cells, worse


def main():
    (OUT / "compare").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(one, SUITES))
    lines = ["# 交付评估：avg3 对 11.17（同一节点、同一推理栈，差 = avg3 − 11.17）", "",
             "| 项目 | 结果 |", "|---|---|"]
    worse_all = []
    for key, cells, worse in results:
        lines.append(f"| {key} | " + "<br>".join(cells) + " |")
        if worse:
            worse_all.append(f"{key}: {', '.join(worse)}")
    lines += ["", "显著更差的项：" + ("；".join(worse_all) if worse_all else "无（热词类无 CI，单独看）")]
    (OUT / "delivery.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
