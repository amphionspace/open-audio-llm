"""Compare AmphionEval runs of the same eval set from their summary.json.

Per (task, language): error rate = sum(errors) / sum(reference units) over all suites, false-alarm and miss counts
(rates times utterances). For the Chinese TS set v12 the ('ts_asr', 'zh') row is the gate (CER / FA / miss).
With few suites (plain ASR, customer sets) it also prints each suite's error rate.
Usage: summary_compare.py NAME,NAME,... RUN_ID [RUN_ID ...]   (runs are looked up under /workspace/data/ae-runs/*/runs/)
"""
import collections
import glob
import json
import sys

SUITE_ROWS_MAX = 60  # print per-suite rows only for small summaries (plain ASR has 34 suites, v12 has 270)


def load(run):
    # A .json path is a summary written aside (rescore_v5.py); otherwise an AmphionEval run ID.
    path = run if run.endswith(".json") else glob.glob(f"/workspace/data/ae-runs/*/runs/{run}")[0] + "/summary.json"
    return json.load(open(path))["metrics"]


def aggregate(metrics):
    total = collections.defaultdict(collections.Counter)
    for key, m in metrics.items():
        _, task, language = key.split("/")
        t = total[(task, language)]
        t["err"] += m.get("errors") or 0
        t["ref"] += m.get("reference_units") or 0
        negative = m.get("negative_utterances") or 0
        t["fa"] += round(negative * (m.get("false_alarm_rate") or 0))
        t["neg"] += negative
        t["miss"] += round((m["utterances"] - negative) * (m.get("miss_rate") or 0))
        t["utt"] += m["utterances"]
    return total


names = sys.argv[1].split(",")
runs = [load(r) for r in sys.argv[2:]]
totals = [aggregate(m) for m in runs]
for key in sorted(set().union(*totals)):
    print(key, "  ".join(f"{n}: {100 * t[key]['err'] / max(t[key]['ref'], 1):.2f} FA {t[key]['fa']}/{t[key]['neg']} "
                         f"miss {t[key]['miss']} n {t[key]['utt']}" for n, t in zip(names, totals)))
if len(runs[0]) <= SUITE_ROWS_MAX:
    print(f"  {'suite':70s}", "  ".join(f"{n:>7s}" for n in names))
    for key in runs[0]:
        print(f"  {key:70s}", "  ".join(f"{100 * (r[key].get('error_rate') or 0):7.2f}" if key in r else "      -" for r in runs))
