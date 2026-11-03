"""Score saved meeting predictions with AmphionEval's own meeting scorer (score_sample + aggregate), so outputs that did
not come through `ae open-audio-llm meeting` (MOSS Pro via amphiondata) get the same cpER / DER / speaker count as the
panel runs. Each predictions row needs id, label, language, reference (timestamped target), duration and the raw output.

    amphion-eval/bin/python -I meeting_rescore.py PREDICTIONS OUTPUT_FORMAT [LABEL]
"""
import json
import sys
from collections import defaultdict

from amphion_eval.integrations.open_audio_llm.meeting import aggregate, score_sample

path, fmt = sys.argv[1:3]
only = sys.argv[3] if len(sys.argv) > 3 else None
scored = defaultdict(list)
for row in map(json.loads, open(path)):
    if only and row["label"] != only:
        continue
    record = {"id": row["id"], "language": row["language"], "target": row["reference"]}
    scored[row["label"]].append(score_sample(record, row.get("raw_text") or row.get("prediction") or "", fmt, row["duration"]))
for label, rows in scored.items():
    print(label, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in aggregate(rows).items()}))
