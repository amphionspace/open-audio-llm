"""Compare hotword-bench style AmphionEval runs (summary.json hotword_bench.metrics): per condition and overall
(error rate = sum errors / sum units, recall = sum hits / sum gold words).
usage: hotword_compare.py NAME=RUN_DIR [NAME=RUN_DIR ...]
"""
import json
import sys


def load(run):
    return json.load(open(f"{run}/summary.json"))["hotword_bench"]["metrics"]


def main():
    runs = [a.split("=", 1) for a in sys.argv[1:]]
    data = {n: load(r) for n, r in runs}
    conds = sorted(set.intersection(*(set(d) for d in data.values())))
    print(f"{'condition':28s} " + "  ".join(f"{n:>20s}" for n, _ in runs) + "   (error % / recall %)")
    tot = {n: [0, 0, 0, 0] for n, _ in runs}
    for c in conds:
        cells = []
        for n, _ in runs:
            m = data[n][c]
            cells.append(f"{100 * m['error_rate']:7.2f} / {100 * (m['recall'] or 0):6.1f}")
            t = tot[n]; t[0] += m["errors"]; t[1] += m["units"]; t[2] += m.get("hits") or 0; t[3] += m.get("gold_total") or 0
        print(f"{c:28s} " + "  ".join(f"{x:>20s}" for x in cells))
    print(f"{'ALL':28s} " + "  ".join(f"{100 * t[0] / t[1]:7.2f} / {100 * t[2] / max(t[3], 1):6.1f}".rjust(20) for t in tot.values()))


if __name__ == "__main__":
    main()
