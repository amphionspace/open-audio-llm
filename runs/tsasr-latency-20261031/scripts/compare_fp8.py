"""Pair FP8 AmphionEval runs with their bf16 10.28 runs (same set, same request parameters).

Both sides are rescored from predictions.jsonl with the installed AmphionEval scorer (some bf16
runs predate scoring v2). Per set: pooled error rate (errors / reference units over the common
items), a paired bootstrap 95% interval of the pooled difference (utterances resampled), the
macro mean over sources, sources worse / better, and how many predictions changed at all.
Hotword runs add hotword recall and distractor false alarms; ts-asr runs add false-output rate on
target-absent items when the set has them.

    python compare_fp8.py OUT_JSON NAME=BF16_RUN:FP8_RUN [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from amphion_eval.integrations.open_audio_llm.scoring import (
    presence_text, summarize, summarize_hotwords, tokens, variety)
from rapidfuzz.distance import Levenshtein

BOOTSTRAP = 2000  # resamples for the 95% interval


def load(run):
    # Failed items have no output to score: the 4266-token audio refused by the 4096 encoder cache,
    # and the items a reference run did not reach.
    rows = map(json.loads, open(Path(run) / "predictions.jsonl"))
    return {r["id"]: r for r in rows if r.get("status", "success") == "success"}


def unit_errors(row):
    kind = variety(row["language"])
    if not presence_text(row["reference"]):
        return 0, 0
    ref = tokens(row["reference"], kind)
    return Levenshtein.distance(ref, tokens(row["prediction"], kind)), len(ref)


def interval(base, fp8, ids):
    """Paired bootstrap of the pooled error-rate difference fp8 - bf16, in percentage points."""
    b = np.array([unit_errors(base[i]) for i in ids], dtype=float)
    f = np.array([unit_errors(fp8[i])[0] for i in ids], dtype=float)
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, len(ids), len(ids))
        deltas.append(100 * (f[pick].sum() - b[pick, 0].sum()) / b[pick, 1].sum())
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def compare(name, base_run, fp8_run):
    evaluator = json.loads((Path(fp8_run) / "run.json").read_text())["config"]["evaluator"]
    base, fp8 = load(base_run), load(fp8_run)
    ids = sorted(base.keys() & fp8.keys())
    score = summarize_hotwords if evaluator == "hotwords" else summarize
    mb = score([base[i] for i in ids])
    mf = score([fp8[i] for i in ids])
    sources = [s for s in sorted(mb) if mb[s].get("error_rate") is not None]
    rows = {s: {"utterances": mb[s]["utterances"], "metric": mb[s]["metric"],
                "bf16": 100 * mb[s]["error_rate"], "fp8": 100 * mf[s]["error_rate"]} for s in sources}
    for s in sources:
        if evaluator == "hotwords" and mb[s]["hotword_total"]:
            rows[s].update(recall_bf16=100 * mb[s]["hotword_recall"], recall_fp8=100 * mf[s]["hotword_recall"],
                           distractor_hits_bf16=mb[s]["distractor_hits"], distractor_hits_fp8=mf[s]["distractor_hits"])
    absent = {s: (100 * mb[s]["false_alarm_rate"], 100 * mf[s]["false_alarm_rate"], mb[s]["negative_utterances"])
              for s in mb if mb[s].get("false_alarm_rate") is not None}
    units = sum(mb[s]["reference_units"] for s in sources)
    out = {
        "bf16_run": str(base_run), "fp8_run": str(fp8_run), "items": len(ids),
        "missing_in_fp8": len(base.keys() - fp8.keys()),
        "changed_predictions": sum(base[i]["prediction"] != fp8[i]["prediction"] for i in ids),
        "pooled_bf16": 100 * sum(mb[s]["errors"] for s in sources) / units,
        "pooled_fp8": 100 * sum(mf[s]["errors"] for s in sources) / units,
        "pooled_delta_95": interval(base, fp8, ids),
        "macro_bf16": float(np.mean([rows[s]["bf16"] for s in sources])),
        "macro_fp8": float(np.mean([rows[s]["fp8"] for s in sources])),
        "sources_worse": sum(rows[s]["fp8"] > rows[s]["bf16"] + 1e-9 for s in sources),
        "sources_better": sum(rows[s]["fp8"] < rows[s]["bf16"] - 1e-9 for s in sources),
        "by_source": rows,
        "false_output_on_absent": absent,
    }
    lo, hi = out["pooled_delta_95"]
    print(f"\n## {name}  n={len(ids)}  changed {out['changed_predictions']} ({100 * out['changed_predictions'] / len(ids):.1f}%)")
    print(f"   pooled {out['pooled_bf16']:.3f} -> {out['pooled_fp8']:.3f} ({out['pooled_fp8'] - out['pooled_bf16']:+.3f}, 95% [{lo:+.3f}, {hi:+.3f}])"
          f"  macro {out['macro_bf16']:.3f} -> {out['macro_fp8']:.3f}  worse {out['sources_worse']} / better {out['sources_better']} of {len(sources)}")
    for s, r in sorted(rows.items(), key=lambda kv: kv[1]["bf16"] - kv[1]["fp8"]):
        extra = f"  recall {r['recall_bf16']:.2f} -> {r['recall_fp8']:.2f}" if "recall_bf16" in r else ""
        print(f"     {s[:70]:70} {r['bf16']:7.3f} -> {r['fp8']:7.3f} ({r['fp8'] - r['bf16']:+.3f}){extra}")
    for s, (a, c, n) in absent.items():
        print(f"     {s[:70]:70} false output {a:.2f}% -> {c:.2f}% (n={n})")
    return out


def main():
    out_path, pairs = sys.argv[1], sys.argv[2:]
    result = {}
    for pair in pairs:
        name, runs = pair.split("=", 1)
        base_run, fp8_run = runs.split(":", 1)
        result[name] = compare(name, base_run, fp8_run)
    Path(out_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
