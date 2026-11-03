"""One table row per (variant, concurrency) from stream_bench lane outputs (``*/rows.json``)."""
import json
import sys
from pathlib import Path

rows = []
for path in sorted(Path(sys.argv[1]).glob("lane*/*/rows.json")):
    variant = path.parent.name
    for row in json.loads(path.read_text()):
        if row["phase"] != "stream":
            continue
        m = row["metrics"]
        key = (variant, row["concurrency"])
        entry = next((r for r in rows if (r["variant"], r["concurrency"]) == key), None)
        if entry is None:
            entry = {"variant": variant, "concurrency": row["concurrency"]}
            rows.append(entry)
        if row["prompt"] == "server":
            drafts = m.get("vllm:spec_decode_num_drafts_total") or m.get("vllm:spec_decode_num_drafts")
            accepted = m.get("vllm:spec_decode_num_accepted_tokens_total") or m.get("vllm:spec_decode_num_accepted_tokens")
            entry["acceptance_rate"] = m.get("spec_acceptance_rate")
            entry["mean_acceptance_length"] = 1 + accepted / drafts if drafts else None
        else:
            entry.update({k: m[k] for k in (
                "segments", "requests", "preview_lag_ms_p50", "preview_lag_ms_p95", "stop_to_final_ms_p50",
                "stop_to_final_ms_p95", "final_latency_ms_p50", "final_latency_ms_p95", "preview_latency_ms_p50",
                "preview_latency_ms_p95", "final_cer", "negative_false_output_rate", "audio_seconds_per_wall_second")})
out = Path(sys.argv[1]) / "summary-table.json"
out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
for r in sorted(rows, key=lambda r: (r["concurrency"], r["variant"])):
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False))
