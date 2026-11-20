"""tasks/02: paired comparison of TS runs that differ only in enrollment length, on samples whose stored enrollment is
at least MIN_ENROLL seconds (so no run pads the enrollment with silence).
Presence: false alarms / whole-utterance misses on that subset. Transcription: in-utterance error by overlap bin on
target-present samples where every run output text (strictly paired).
Usage: enroll_len_paired.py MIN_ENROLL SAMPLES.jsonl NAME=RUN_DIR [NAME=RUN_DIR ...] > table.txt
"""
import collections
import json
import sys

from amphion_eval.integrations.open_audio_llm import scoring

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ts_gate_breakdown import edit_ops, enrollment_seconds, overlap_bin, read_jsonl  # noqa: E402

min_enroll, samples_path, runs = float(sys.argv[1]), sys.argv[2], [a.split("=", 1) for a in sys.argv[3:]]
samples = {s["id"]: s["record"] for s in read_jsonl(samples_path)}
keep = {i for i, r in samples.items() if r.get("task") == "ts_asr" and r.get("language") == "zh"
        and (enrollment_seconds(r) or 0) >= min_enroll}
preds = {name: {p["id"]: p for p in read_jsonl(f"{run}/predictions.jsonl") if p["id"] in keep} for name, run in runs}
ids = sorted(set.intersection(*(set(p) for p in preds.values())))
presence = {name: collections.Counter() for name, _ in runs}
paired = collections.defaultdict(lambda: {name: collections.Counter() for name, _ in runs})
for i in ids:
    ref = scoring.tokens(preds[runs[0][0]][i].get("reference") or "", "zh")
    hyps = {name: scoring.tokens(preds[name][i].get("prediction") or "", "zh") for name, _ in runs}
    for name, h in hyps.items():
        c = presence[name]
        if not ref:
            c["absent"] += 1; c["fa"] += bool(h)
        else:
            c["present"] += 1; c["miss"] += not h
    if ref and all(hyps.values()):
        b = overlap_bin((samples[i].get("labels") or {}).get("target_overlap_ratio"))
        for key in ("all", b):
            for name, h in hyps.items():
                c = paired[key][name]; c["ref"] += len(ref); c["err"] += sum(edit_ops(ref, h))
names = [n for n, _ in runs]
print(f"subset: enrollment >= {min_enroll} s, {len(ids)} samples")
print("presence      " + "  ".join(f"{n:>12s}" for n in names))
print("FA            " + "  ".join(f"{presence[n]['fa']:>5d}/{presence[n]['absent']:<6d}" for n in names))
print("miss          " + "  ".join(f"{presence[n]['miss']:>5d}/{presence[n]['present']:<6d}" for n in names))
print("in-utterance error % (paired: every run output text)")
for key in ("all", "0", "0-0.6", "0.6-0.9", ">=0.9"):
    if key in paired:
        row = paired[key]
        print(f"  {key:10s} ref {row[names[0]]['ref']:7d}  " + "  ".join(f"{n}={100 * row[n]['err'] / row[n]['ref']:.2f}" for n in names))
