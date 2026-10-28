"""Same-panel diagnostics for MOSS-Transcribe-Diarize outputs: convert MOSS's ``[start][Sxx]text[end]`` predictions
(AmphionEval moss_transcribe_diarize parser) into this round's ``[Sxx][start-end] text`` lines and run the selector's
own turn_diagnostics on them, next to any of our evaluation runs of the same panel.

    amphion-eval/bin/python moss_turn_diagnostics.py OUT_JSON LABEL SIMILARITY NAME=PREDICTIONS[:moss] ...
"""
import json
import sys
from pathlib import Path

from amphion_eval.parsers import ParseError, parse_moss_transcribe_diarize

sys.path.insert(0, str(Path(__file__).parent))
from select_checkpoints import turn_diagnostics  # noqa: E402

out, label, similarity, *runs = sys.argv[1:]
result = {}
for spec in runs:
    name, path = spec.split("=", 1)
    moss = path.endswith(":moss")
    path = Path(path.removesuffix(":moss"))
    if moss:
        converted = Path(out).with_name(f"predictions-{name}.jsonl")
        with converted.open("w") as stream:
            for row in map(json.loads, path.open()):
                try:
                    segments = parse_moss_transcribe_diarize(row.get("prediction") or "")
                except ParseError:  # refusals ("I'm sorry, I can't assist..."), all on NOTSOFAR windows: scored as silent
                    segments = []
                row["prediction"] = "\n".join(f"[{s['speaker']}][{float(s['start']):.2f}-{float(s['end']):.2f}] {s['text'].strip()}"
                                              for s in segments)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        path = converted
    result[name] = turn_diagnostics(path, label, similarity)
Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
keys = sorted({k for v in result.values() for k in v if isinstance(v[k], (int, float))})
print(f"{'metric':45s}" + "".join(f"{n:>14s}" for n in result))
for k in keys:
    print(f"{k:45s}" + "".join(f"{result[n].get(k, float('nan')):14.4f}" for n in result))
