"""Paired bootstrap of the in-utterance transcription error difference between two v12 runs, per target-overlap bin.

Samples: target-present zh TS samples where both runs (and, if given, every run in --pair-with) wrote text, the same
strict pairing as enroll_len_paired.py; error = AmphionEval zh-token edits / reference length, summed over the
resampled samples. Prints B minus A with a 95% interval from 2,000 resamples of samples.
usage: paired_bootstrap.py SAMPLES.jsonl A=RUN_DIR B=RUN_DIR [C=RUN_DIR ...]   (pairs: B-A, C-A, ...)
"""
import random
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from amphion_eval.integrations.open_audio_llm import scoring  # noqa: E402
from ts_gate_breakdown import edit_ops, overlap_bin, read_jsonl  # noqa: E402

RESAMPLES = 2000


def main():
    samples = {s["id"]: s["record"] for s in read_jsonl(sys.argv[1])}
    runs = [a.split("=", 1) for a in sys.argv[2:]]
    keep = {i for i, r in samples.items() if r.get("task") == "ts_asr" and r.get("language") == "zh"}
    preds = {name: {p["id"]: p for p in read_jsonl(f"{run}/predictions.jsonl") if p["id"] in keep} for name, run in runs}
    rows = {}
    for sid in keep:
        ps = [preds[name].get(sid) for name, _ in runs]
        if any(p is None for p in ps):
            continue
        ref = scoring.tokens(ps[0]["reference"] or "", "zh")
        hyps = [scoring.tokens(p.get("prediction") or "", "zh") for p in ps]
        if not ref or not all(hyps):
            continue
        labels = samples[sid].get("labels") or {}
        rows.setdefault(overlap_bin(labels.get("target_overlap_ratio")), []).append(
            (len(ref), [sum(edit_ops(ref, h)) for h in hyps]))
    rng = random.Random(0)
    names = [n for n, _ in runs]
    for b in sorted(rows, key=str):
        data = rows[b]
        n = len(data)
        point = {}
        for k in range(1, len(names)):
            diff = lambda d: 100 * (sum(e[k] for _, e in d) - sum(e[0] for _, e in d)) / sum(r for r, _ in d)
            boots = sorted(diff([data[rng.randrange(n)] for _ in range(n)]) for _ in range(RESAMPLES))
            point[names[k]] = (diff(data), boots[int(0.025 * RESAMPLES)], boots[int(0.975 * RESAMPLES)])
        base = 100 * sum(e[0] for _, e in data) / sum(r for r, _ in data)
        print(f"overlap {b!s:8s} n={n:5d} {names[0]}={base:.2f}  " + "  ".join(
            f"{k}-{names[0]} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]" for k, v in point.items()), flush=True)


if __name__ == "__main__":
    main()
