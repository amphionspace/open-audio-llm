"""Pooled hotword-bench numbers as in the 11.06 package's validation/results/results.md (compare does not cover them).

CER = errors / units over all condition groups; recall = hits / gold over the groups that pass a hotword list (not
``/none``). Reproduces the package's ts-asr-hotwords, badcase and hotword-bench cells from the release runs.
usage: hotword_pool.py <run dir> [...]
"""
import json
import sys


def pooled(groups):
    e, u = sum(g.get("errors") or 0 for g in groups), sum(g.get("units") or 0 for g in groups)
    h, t = sum(g.get("hits") or 0 for g in groups), sum(g.get("gold_total") or 0 for g in groups)
    return e / u, h / t


for run in sys.argv[1:]:
    metrics = json.load(open(f"{run}/summary.json"))["hotword_bench"]["metrics"]
    cer, _ = pooled(metrics.values())
    _, recall = pooled([g for k, g in metrics.items() if not k.endswith("/none")])
    print(f"{run.rstrip('/').rsplit('/', 1)[-1]}\tCER {100 * cer:.2f}\trecall {100 * recall:.1f}")
