"""tasks/27 meeting benchmark report: absent false alarms and present misses / CER on the ts_asr_real meeting splits
(meeting-v3: AISHELL-4; meeting-v4 and later: AliMeeting and MagicData-RAMC with controls) from AmphionEval TS runs.

Per model (one or more run directories of it, joined by record id):
- per split: requests, FA% (absent target by labels.target_present, any output token), miss% (present target, no
  output token), CER% (present target, edits over reference characters; a miss counts its reference as deletions);
  present requests whose reference normalizes to nothing (fillers only) are counted apart, not scored;
- base absent FA by labels.impostor_sim, duration_s, overlap_s and meeting_boxes, per corpus;
- each control against its own base request (metadata.base): absent FA, present miss and CER, with the difference and
  a 95% bootstrap interval over recordings (the cluster; 2,000 resamples);
- -xrec at matched similarity (|labels.xrec_sim_gap| <= MATCH): FA with the in-meeting enrollment vs. with the
  other-recording one on the same segments, overall and by similarity bin.
With several models, each later model is compared with the first on every split, record by record (clustered).

Tokens and normalization come from AmphionEval's scoring (run with the amphion-eval interpreter).
usage: meeting_fa_report.py --model NAME=RUN[,RUN...] [--model NAME=RUN[,RUN...]] [--records VERSION_DIR ...] --out OUT.json [--wandb]
--records: score only the records of these dataset version directories (a derived version keeps its source's record
  IDs, e.g. meeting-v5 drops meeting-v4 segments), for the splits they hold; other splits are kept whole.
--wandb: add the numbers as meeting/* summary keys to the W&B run of each model's first run directory.
"""
import argparse
import collections
import gzip
import json
import os

import numpy as np
from amphion_eval.integrations.open_audio_llm import scoring
from rapidfuzz.distance import Levenshtein

MATCH = 0.05  # |cosine gap| of a matched -xrec pair
BOOT = 2000
SIM_BINS = (-1.0, 0.2, 0.3, 0.4, 0.5, 1.01)
DUR_BINS = (0.0, 2.0, 4.0, 6.0, 10.0, 1e9)
OVL_BINS = (0.0, 1e-3, 0.5, 1e9)
BOX_BINS = (0, 3, 4, 5, 6, 100)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def kept_records(dirs):
    """{split: record ids} of the given dataset version directories."""
    keep = {}
    for d in dirs:
        for path in sorted(os.listdir(os.path.join(d, "records"))):
            with gzip.open(os.path.join(d, "records", path), "rt", encoding="utf-8") as f:
                keep[path.split(".")[0]] = {json.loads(line)["id"] for line in f if line.strip()}
    return keep


def load(runs, keep=None):
    """{record id: request} of the meeting splits in the run directories (only ``keep``'s records of its splits)."""
    out = {}
    for run in runs:
        samples = {s["id"]: s for s in read_jsonl(os.path.join(run, "samples.jsonl"))}
        for p in read_jsonl(os.path.join(run, "predictions.jsonl")):
            s = samples.get(p["id"])
            if s is None or ":meeting-" not in s["source"]:
                continue
            record = s["record"]
            labels, meta = record.get("labels") or {}, record.get("metadata") or {}
            split = s["source"].split(":", 1)[1].split("/")[0]
            if keep and split in keep and p["id"] not in keep[split]:
                continue
            ref = scoring.tokens(p.get("reference") or "", "zh")
            hyp = scoring.tokens(p.get("prediction") or "", "zh")
            base = meta.get("base")
            key = (split, labels["recording"], meta["segment_index"], meta["target_speaker"])
            out[p["id"]] = {
                "split": split, "labels": labels, "cluster": labels["recording"], "present": bool(labels["target_present"]),
                "out": bool(hyp), "ref": len(ref), "edits": Levenshtein.distance(ref, hyp) if ref else 0,
                "key": key, "base_key": None if base is None else
                (base["split"], base["recording"], base["segment_index"], base["target_speaker"]),
            }
    return out


def rates(items):
    absent = [x for x in items if not x["present"]]
    present = [x for x in items if x["present"] and x["ref"]]
    r = {"n": len(items)}
    if len(present) < sum(x["present"] for x in items):
        r["present_empty_ref"] = sum(x["present"] and not x["ref"] for x in items)
    if absent:
        r["fa"] = round(100 * sum(x["out"] for x in absent) / len(absent), 2)
    if present:
        r["miss"] = round(100 * sum(not x["out"] for x in present) / len(present), 2)
        r["cer"] = round(100 * sum(x["edits"] for x in present) / max(1, sum(x["ref"] for x in present)), 2)
    return r


def cluster_delta(rows, rng):
    """rows: (cluster, a, b, den). (sum b - sum a) / sum den with a 95% bootstrap interval over clusters."""
    sums = collections.defaultdict(lambda: np.zeros(3))
    for c, a, b, d in rows:
        sums[c] += (a, b, d)
    m = np.array(list(sums.values()))
    point = (m[:, 1].sum() - m[:, 0].sum()) / m[:, 2].sum()
    idx = rng.integers(0, len(m), size=(BOOT, len(m)))
    s = m[idx].sum(axis=1)
    boot = (s[:, 1] - s[:, 0]) / s[:, 2]
    return [round(100 * point, 2), round(100 * float(np.percentile(boot, 2.5)), 2),
            round(100 * float(np.percentile(boot, 97.5)), 2)]


def paired(pairs, rng):
    """Base vs. other on paired requests: FA (absent), miss and CER (present)."""
    out = {"pairs": len(pairs)}
    absent = [(a, b) for a, b in pairs if not a["present"]]
    present = [(a, b) for a, b in pairs if a["present"] and a["ref"]]
    if absent:
        out["fa"] = [round(100 * sum(a["out"] for a, _ in absent) / len(absent), 2),
                     round(100 * sum(b["out"] for _, b in absent) / len(absent), 2)]
        out["fa_delta"] = cluster_delta([(a["cluster"], a["out"], b["out"], 1) for a, b in absent], rng)
    if present:
        out["miss"] = [round(100 * sum(not a["out"] for a, _ in present) / len(present), 2),
                       round(100 * sum(not b["out"] for _, b in present) / len(present), 2)]
        out["miss_delta"] = cluster_delta([(a["cluster"], not a["out"], not b["out"], 1) for a, b in present], rng)
        ref = sum(a["ref"] for a, _ in present)
        out["cer"] = [round(100 * sum(a["edits"] for a, _ in present) / ref, 2),
                      round(100 * sum(b["edits"] for _, b in present) / ref, 2)]
        out["cer_delta"] = cluster_delta([(a["cluster"], a["edits"], b["edits"], a["ref"]) for a, b in present], rng)
    return out


def binned(items, field, edges):
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        xs = [x for x in items if x["labels"].get(field) is not None and lo <= x["labels"][field] < hi]
        if xs:
            out[f"{lo:g}-{hi:g}"] = rates(xs)
    return out


def corpus_of(split):
    return split.split("-")[1]


def report(requests, rng):
    by_split = collections.defaultdict(list)
    for x in requests.values():
        by_split[x["split"]].append(x)
    res = {"splits": {s: rates(xs) for s, xs in sorted(by_split.items())}, "curves": {}, "controls": {}, "xrec_matched": {}}
    base = {x["key"]: x for x in requests.values() if x["base_key"] is None}
    for corpus in sorted({corpus_of(s) for s in by_split}):
        neg = [x for s, xs in by_split.items() if corpus_of(s) == corpus and s.endswith("-neg") for x in xs]
        res["curves"][corpus] = {"impostor_sim": binned(neg, "impostor_sim", SIM_BINS),
                                 "duration_s": binned(neg, "duration_s", DUR_BINS),
                                 "overlap_s": binned(neg, "overlap_s", OVL_BINS),
                                 "meeting_boxes": binned(neg, "meeting_boxes", BOX_BINS)}
    for split, xs in sorted(by_split.items()):
        pairs = [(base[x["base_key"]], x) for x in xs if x["base_key"] is not None and x["base_key"] in base]
        if not pairs:
            continue
        res["controls"][split] = paired(pairs, rng)
        if split.endswith("-xrec"):
            matched = [(a, b) for a, b in pairs if abs(b["labels"]["xrec_sim_gap"]) <= MATCH]
            entry = {"match": MATCH, "coverage": round(len(matched) / len(pairs), 3), **paired(matched, rng), "by_sim": {}}
            for lo, hi in zip(SIM_BINS[:-1], SIM_BINS[1:]):
                sub = [(a, b) for a, b in matched if lo <= a["labels"]["impostor_sim"] < hi]
                if sub:
                    entry["by_sim"][f"{lo:g}-{hi:g}"] = paired(sub, rng)
            res["xrec_matched"][split] = entry
        if split.endswith("-snr"):
            res["controls"][split]["by_snr"] = {}
            for lo, hi in ((-1e9, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 1e9)):
                sub = [(a, b) for a, b in pairs if lo <= b["labels"]["snr_db"] < hi]
                if sub:
                    res["controls"][split]["by_snr"][f"{lo:g}-{hi:g}"] = paired(sub, rng)
    return res


def compare(ref, other, rng):
    """Each split of ``other`` against ``ref`` on the same record ids."""
    out = {}
    by_split = collections.defaultdict(list)
    for rid, x in other.items():
        if rid in ref:
            by_split[x["split"]].append((ref[rid], x))
    for split, pairs in sorted(by_split.items()):
        out[split] = paired(pairs, rng)
    return out


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}/{k}", v, out)
    elif isinstance(value, list):
        for name, v in zip(("point", "lo", "hi") if len(value) == 3 else ("a", "b"), value):
            out[f"{prefix}/{name}"] = v
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = value


def log_wandb(run_dir, result):
    import wandb

    run = json.load(open(os.path.join(run_dir, "run.json")))
    entity, project = run["wandb_url"].split("wandb.ai/")[1].split("/")[:2]
    summary = {}
    flatten("meeting", result, summary)
    w = wandb.init(entity=entity, project=project, id=run["run_id"], resume="must")
    w.summary.update(summary)
    w.finish()
    print(f"W&B {run['wandb_url']}: {len(summary)} meeting keys", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="NAME=RUN_DIR[,RUN_DIR...]")
    ap.add_argument("--records", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    models = [(m.split("=", 1)[0], m.split("=", 1)[1].split(",")) for m in args.model]
    keep = kept_records(args.records)
    loaded = {name: load(runs, keep) for name, runs in models}
    result = {name: report(loaded[name], rng) for name, _ in models}
    first = models[0][0]
    for name, _ in models[1:]:
        result[name]["vs_" + first] = compare(loaded[first], loaded[name], rng)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    for name, _ in models:
        print("==", name)
        for split, r in result[name]["splits"].items():
            print(" ", split, r)
        for split, r in result[name]["controls"].items():
            print("  control", split, {k: v for k, v in r.items() if k != "by_snr"})
        for split, r in result[name]["xrec_matched"].items():
            print("  matched", split, {k: v for k, v in r.items() if k != "by_sim"})
    if args.wandb:
        for name, runs in models:
            log_wandb(runs[0], result[name])


if __name__ == "__main__":
    main()
