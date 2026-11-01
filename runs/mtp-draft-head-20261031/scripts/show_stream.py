"""Compact stream_bench table from summary-table.json. Usage: show_stream.py ARTIFACTS_DIR"""
import json
import sys
from pathlib import Path

for r in sorted(json.loads((Path(sys.argv[1]) / "summary-table.json").read_text()),
                key=lambda r: (r["concurrency"], r["variant"])):
    print(f'c{r["concurrency"]:<3} {r["variant"]:18} segs {r["segments"]:4} lag {r["preview_lag_ms_p50"]:5.0f}/{r["preview_lag_ms_p95"]:5.0f} '
          f'final {r["final_latency_ms_p50"]:5.0f}/{r["final_latency_ms_p95"]:5.0f} stop2final {r["stop_to_final_ms_p50"]:5.0f}/{r["stop_to_final_ms_p95"]:5.0f} '
          f'MAL {r["mean_acceptance_length"] or 0:.2f} aud/s {r["audio_seconds_per_wall_second"]:.1f} cer {r["final_cer"]:.4f}')
