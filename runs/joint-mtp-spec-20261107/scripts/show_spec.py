"""One line per variant/mode/concurrency from spec_check lane logs. Usage: show_spec.py ATTEMPT_DIR"""
import json
import sys
from pathlib import Path

rows = []
for path in sorted(Path(sys.argv[1]).glob("logs/lane*.log")):
    for line in path.read_text().splitlines():
        if line.startswith('{"variant"'):
            rows.append(json.loads(line))
rows.sort(key=lambda d: (d.get("mode", ""), d.get("concurrency", 0), d["variant"]))
for d in rows:
    if "error" in d:
        print(d)
        continue
    per = d["per_position"]
    print(f'{d["mode"]:6} c{d["concurrency"]:<3} {d["variant"]:10} p50 {d["latency_ms_p50"]:5.0f} p95 {d["latency_ms_p95"]:5.0f} '
          f'mean {d["latency_ms_mean"]:5.0f} tok/s {d["output_tokens_per_s"]:6.0f} '
          f'MAL {d["mean_acceptance_length"] or 0:.3f} {[round(x, 3) for x in per] if per else ""}')
