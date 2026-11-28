"""Split a TS-ASR gate run's character errors into presence errors and in-utterance transcription errors by overlap.

The gate (CER / false alarms / misses from summary.json) stays as it is. This adds, on the same predictions:
- presence: false-alarm utterances and characters (target absent, text output), whole-utterance misses;
- in-utterance error: (S + D + I) / reference characters on target-present samples where the model output text,
  per target overlap bin (labels.target_overlap_ratio);
- interference cost (with --asr-run): the same target utterances recorded alone, from the same model's plain-ASR run,
  matched by corpus + utterance id; TS error minus plain error on those samples.

Tokens and normalization come from AmphionEval's scoring (run with the amphion-eval interpreter).
Usage:
  ts_gate_breakdown.py --ts-run RUN_DIR [--asr-run RUN_DIR] [--samples SAMPLES.jsonl] [--min-enroll-seconds S]
                       [--language zh] --out OUT.json [--wandb]
--samples: a run that reused inference has no samples.jsonl; pass one from a run of the same eval set.
--wandb: add the numbers as breakdown/* summary keys to the TS run's own W&B run (id = AmphionEval run id).
"""
import argparse
import collections
import json
import os
import sys

from amphion_eval.integrations.open_audio_llm import scoring

# Overlap bins of the 2026-11-19 retrospective (docs/experiments/2026-11-19-ts-retro-overlap).
OVERLAP_BINS = ("0", "0-0.6", "0.6-0.9", ">=0.9")


def overlap_bin(ratio):
    if ratio is None:
        return None
    if ratio == 0:
        return "0"
    return "0-0.6" if ratio < 0.6 else "0.6-0.9" if ratio < 0.9 else ">=0.9"


def edit_ops(ref, hyp):
    """Substitutions, deletions, insertions of a minimum-edit alignment."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    i, j, s, dl, ins = n, m, 0, 0, 0
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            dl, i = dl + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return s, dl, ins


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def enrollment_seconds(record):
    for slot in record.get("audio_slots", []):
        if slot.get("purpose") == "enrollment":
            return slot["ref"].get("duration")
    return None


def rate(errors, ref):
    return round(100.0 * errors / ref, 2) if ref else None


def breakdown(ts_run, samples_path, language, min_enroll):
    samples = {s["id"]: s for s in read_jsonl(samples_path)}
    total, bins = collections.Counter(), collections.defaultdict(collections.Counter)
    kept = set()
    for p in read_jsonl(os.path.join(ts_run, "predictions.jsonl")):
        if p.get("task") != "ts_asr" or p.get("language") != language:
            continue
        if p["id"] not in samples:
            raise SystemExit(f"{p['id']} has no sample record; pass --samples from a run of the same eval set")
        record = samples[p["id"]]["record"]
        if min_enroll is not None and (enrollment_seconds(record) or 0) < min_enroll:
            continue
        kept.add(p["id"])
        ref = scoring.tokens(p.get("reference") or "", language)
        hyp = scoring.tokens(p.get("prediction") or "", language)
        c = collections.Counter(utt=1, ref=len(ref))
        if not ref:
            c["absent_utt"] = 1
            if hyp:
                c["fa_utt"], c["fa_chars"] = 1, len(hyp)
        elif not hyp:
            c["present_utt"], c["miss_utt"], c["miss_chars"] = 1, 1, len(ref)
        else:
            s, dl, ins = edit_ops(ref, hyp)
            c.update(present_utt=1, output_utt=1, output_ref=len(ref), sub=s, dele=dl, ins=ins)
        total.update(c)
        b = overlap_bin((record.get("labels") or {}).get("target_overlap_ratio"))
        if b is not None and ref:
            bins[b].update(c)
    return total, bins, kept


def interference_cost(ts_run, asr_run, samples_path, language, kept):
    """TS vs plain-ASR error on the same target utterances (plain-ASR ids end with the source utterance id)."""
    plain = {}
    for p in read_jsonl(os.path.join(asr_run, "predictions.jsonl")):
        plain[(p["id"].split("@")[0], p["id"].rsplit(":", 1)[-1])] = p
    samples = {s["id"]: s for s in read_jsonl(samples_path)}
    c = collections.Counter()
    for p in read_jsonl(os.path.join(ts_run, "predictions.jsonl")):
        if p.get("task") != "ts_asr" or p.get("language") != language or not p.get("reference") or p["id"] not in kept:
            continue
        utts = (samples[p["id"]]["record"].get("metadata") or {}).get("target_utts") or []
        matched = [plain.get((u["id"].split(":")[0], u["id"].split(":", 1)[1])) for u in utts]
        if not matched or any(m is None for m in matched):
            continue
        ref = scoring.tokens(p["reference"], language)
        hyp = scoring.tokens(p.get("prediction") or "", language)
        plain_ref = scoring.tokens("".join(m["reference"] for m in matched), language)
        plain_hyp = scoring.tokens("".join(m.get("prediction") or "" for m in matched), language)
        # A TS reference that keeps only part of the utterance would not be the same sentence.
        if abs(len(plain_ref) - len(ref)) > 2:
            continue
        c.update(samples=1, ref=len(ref), ts_errors=sum(edit_ops(ref, hyp)), plain_errors=sum(edit_ops(plain_ref, plain_hyp)))
    if not c["samples"]:
        return None
    return {"samples": c["samples"], "ref_chars": c["ref"], "plain_error": rate(c["plain_errors"], c["ref"]),
            "ts_error": rate(c["ts_errors"], c["ref"]), "cost": rate(c["ts_errors"] - c["plain_errors"], c["ref"])}


def summarize(total, bins):
    errors = total["fa_chars"] + total["miss_chars"] + total["sub"] + total["dele"] + total["ins"]
    presence = total["fa_chars"] + total["miss_chars"]
    out = {
        "utterances": total["utt"], "reference_chars": total["ref"],
        "false_alarm_utterances": total["fa_utt"], "absent_utterances": total["absent_utt"],
        "miss_utterances": total["miss_utt"], "present_utterances": total["present_utt"],
        "presence_error_share": round(100.0 * presence / errors, 1) if errors else None,
        "in_utterance_error": rate(total["sub"] + total["dele"] + total["ins"], total["output_ref"]),
        "overlap": {},
    }
    for name in OVERLAP_BINS:
        c = bins.get(name)
        if c:
            out["overlap"][name] = {"present_utterances": c["present_utt"], "miss_utterances": c["miss_utt"],
                                    "output_ref_chars": c["output_ref"],
                                    "in_utterance_error": rate(c["sub"] + c["dele"] + c["ins"], c["output_ref"])}
    return out


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}/{k}", v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = value


def log_wandb(ts_run, result):
    import wandb

    run = json.load(open(os.path.join(ts_run, "run.json")))
    entity, project = run["wandb_url"].split("wandb.ai/")[1].split("/")[:2]
    summary = {}
    flatten("breakdown", result, summary)
    w = wandb.init(entity=entity, project=project, id=run["run_id"], resume="must")
    w.summary.update(summary)
    w.finish()
    print(f"W&B {run['wandb_url']}: {len(summary)} breakdown keys", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts-run", required=True)
    ap.add_argument("--asr-run")
    ap.add_argument("--samples")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--min-enroll-seconds", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    samples = args.samples or os.path.join(args.ts_run, "samples.jsonl")
    total, bins, kept = breakdown(args.ts_run, samples, args.language, args.min_enroll_seconds)
    result = summarize(total, bins)
    result["filter"] = {"language": args.language, "min_enroll_seconds": args.min_enroll_seconds}
    result["source"] = {"ts_run": os.path.abspath(args.ts_run), "asr_run": args.asr_run and os.path.abspath(args.asr_run)}
    if args.asr_run:
        result["interference"] = interference_cost(args.ts_run, args.asr_run, samples, args.language, kept)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"present {result['present_utterances']}  miss {result['miss_utterances']}  "
          f"FA {result['false_alarm_utterances']}/{result['absent_utterances']}  "
          f"presence share {result['presence_error_share']}%  in-utt {result['in_utterance_error']}%")
    for name, v in result["overlap"].items():
        print(f"  overlap {name:8s} present {v['present_utterances']:6d}  miss {v['miss_utterances']:5d}  in-utt {v['in_utterance_error']}%")
    if result.get("interference"):
        print("  interference", result["interference"])
    if args.wandb:
        log_wandb(args.ts_run, result)


if __name__ == "__main__":
    sys.exit(main())
