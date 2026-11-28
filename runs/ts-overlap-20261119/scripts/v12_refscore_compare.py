"""tasks/06: compare ref_score.py outputs of several models on the same v12 target-present samples.

Per model and target-overlap bin (the 2026-11-19 retrospective's bins), on samples every model scored:
- ref_nll: teacher-forced -log p of the reference per non-punctuation token (acoustic/LM fit of the test set);
- r1 / r2-5 / r>20: share of reference tokens by rank (r>20: the model has practically lost the word);
- p_none: mean P(" None") at the language decision under the true enrollment;
- swap_gain: (log p of the reference with the true enrollment - with another speaker's) per token: how much the
  enrollment helps;
- sel_overlap / sel_interferer: 1 - cosine of the last audio-tower layer under the enrollment swap, at mixture frames
  where target and interferer overlap / only the interferer talks (encoder target selectivity, as in the retrospective);
- own_agree: HF greedy reproduces the served output (sanity check, should stay ~1).
Usage: v12_refscore_compare.py OUT.json NAME=SHARD_DIR [NAME=SHARD_DIR ...]
"""
import collections
import glob
import json
import sys

BINS = ("0", "0-0.6", "0.6-0.9", ">=0.9")


def overlap_bin(ratio):
    if ratio is None:
        return None
    return "0" if ratio == 0 else "0-0.6" if ratio < 0.6 else "0.6-0.9" if ratio < 0.9 else ">=0.9"


def load(shard_dir):
    return {r["id"]: r for path in sorted(glob.glob(f"{shard_dir}/shard-*.jsonl")) for r in map(json.loads, open(path))}


def main():
    out, specs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    models = {name: load(d) for name, d in specs}
    ids = sorted(set.intersection(*(set(m) for m in models.values())))
    result = {"samples": len(ids), "models": {}}
    print(f"paired samples: {len(ids)}")
    header = f"{'bin':8s} {'model':10s} {'n':>5s} {'ref_nll':>8s} {'r1%':>6s} {'r2-5%':>6s} {'r>20%':>6s} {'p_none':>7s} " \
             f"{'swap_gain':>9s} {'sel_ov':>7s} {'sel_int':>7s} {'agree':>6s}"
    print(header)
    for name, recs in models.items():
        result["models"][name] = {}
        for b in ("all", *BINS):
            c = collections.Counter()
            for i in ids:
                r = recs[i]
                if b != "all" and overlap_bin(r["labels"].get("target_overlap_ratio")) != b:
                    continue
                ref = r["ref"]
                hist = ref["rank_hist"]  # ranks 1, 2-5, 6-20, 21-100, >100
                c["n"] += 1
                c["tokens"] += ref["n_tokens"]
                c["nll"] -= ref["logp_sum"]
                c["r1"] += hist[0]
                c["r2_5"] += hist[1]
                c["r20"] += hist[3] + hist[4]
                c["p_none"] += 2.718281828459045 ** r["decision_true"]["none"]
                c["swap_gain"] += ref["logp_sum"] - r["swap"]["ref_logp_sum"]
                c["agree"] += r["own_agree"][0]
                c["agree_n"] += r["own_agree"][1]
                regions = r["swap"].get("regions") or {}
                for key, region in (("ov", "overlap"), ("int", "interferer_only")):
                    if region in regions:
                        c[f"sel_{key}"] += 1 - regions[region]["audio"][-1]
                        c[f"sel_{key}_n"] += 1
            if not c["n"]:
                continue
            t = c["tokens"]
            m = {"samples": c["n"], "ref_nll": round(c["nll"] / t, 4), "r1": round(100 * c["r1"] / t, 2),
                 "r2_5": round(100 * c["r2_5"] / t, 2), "r_gt20": round(100 * c["r20"] / t, 2),
                 "p_none": round(c["p_none"] / c["n"], 4), "swap_gain": round(c["swap_gain"] / t, 4),
                 "sel_overlap": round(c["sel_ov"] / c["sel_ov_n"], 5) if c["sel_ov_n"] else None,
                 "sel_interferer": round(c["sel_int"] / c["sel_int_n"], 5) if c["sel_int_n"] else None,
                 "own_agree": round(c["agree"] / c["agree_n"], 4)}
            result["models"][name][b] = m
    for b in ("all", *BINS):
        for name in models:
            m = result["models"][name].get(b)
            if m:
                print(f"{b:8s} {name:10s} {m['samples']:5d} {m['ref_nll']:8.4f} {m['r1']:6.2f} {m['r2_5']:6.2f} "
                      f"{m['r_gt20']:6.2f} {m['p_none']:7.4f} {m['swap_gain']:9.4f} {m['sel_overlap'] or 0:7.4f} "
                      f"{m['sel_interferer'] or 0:7.4f} {m['own_agree']:6.4f}")
    json.dump(result, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
