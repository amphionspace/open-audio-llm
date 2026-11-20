"""Aggregate ref_score.py shards: where the target's words are lost under overlap.

Usage: analyze_ref_score.py OUT.json SHARD.jsonl [...] --predictions AE_RUN/predictions.jsonl
Prints tables and writes OUT.json. Per-sample TS character errors come from the served predictions
(AmphionEval ``scoring.tokens``, punctuation removed), so the white-box numbers can be tied to real errors.
"""
import argparse
import collections
import json
import statistics

RANK_NAMES = ("r1", "r2-5", "r6-20", "r21-100", "r>100")
REGIONS = ("target_only", "overlap", "interferer_only", "silence")


def edit_distance(r, h):
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j - 1] + (r[i - 1] != h[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]


def overlap_bin(lab):
    ov = lab.get("target_overlap_ratio")
    if ov is None:
        return "na"
    return "0" if ov == 0 else "<0.3" if ov < .3 else "<0.6" if ov < .6 else "<0.9" if ov < .9 else ">=0.9"


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = k
        return r
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--predictions", required=True)
    args = ap.parse_args()
    from amphion_eval.integrations.open_audio_llm import scoring

    preds = {p["id"]: p for p in map(json.loads, open(args.predictions))}
    recs = [json.loads(line) for f in args.shards for line in open(f)]
    agree = [sum(r["own_agree"][0] for r in recs), sum(r["own_agree"][1] for r in recs)]
    result = {"samples": len(recs), "hf_vllm_token_agreement": agree[0] / agree[1], "bins": {}}
    print(f"samples {len(recs)}; HF greedy == vLLM token agreement {agree[0]}/{agree[1]} = {agree[0]/agree[1]:.4f}")

    groups = collections.defaultdict(list)
    for r in recs:
        lab = r["labels"]
        p = preds[r["id"]]
        ref = scoring.tokens(p["reference"] or "", "zh")
        hyp = scoring.tokens(p.get("prediction") or "", "zh")
        r["_err"], r["_ref"], r["_miss"] = edit_distance(ref, hyp), len(ref), not hyp
        for key in ("all", "overlap=" + overlap_bin(lab), "n_interf=" + str(lab.get("n_interferers")),
                    "group=" + str(lab.get("group"))):
            groups[key].append(r)

    print("\n[1] reference tokens under teacher forcing (served-output CER for the same samples)")
    print(f"{'bin':18s} {'n':>5s} {'CER%':>6s} {'miss':>4s} | " + " ".join(f"{n:>7s}" for n in RANK_NAMES)
          + " | wrong: near(2-5) far(>20) medLogp  eos  interf chance")
    for key in sorted(groups, key=lambda k: (k.split("=")[0] != "all", k)):
        rs = [r for r in groups[key] if not r["_miss"]]  # whole-utterance misses are a presence error, not a word error
        hist = [sum(r["ref"]["rank_hist"][i] for r in rs) for i in range(5)]
        tot = sum(hist) or 1
        wrong_ranks = [x for r in rs for x in r["ref"]["wrong_rank"]]
        wrong_lp = [x for r in rs for x in r["ref"]["wrong_logp"]]
        nw = len(wrong_ranks) or 1
        near = sum(1 for x in wrong_ranks if x <= 5) / nw
        far = sum(1 for x in wrong_ranks if x > 20) / nw
        eos = sum(r["ref"]["eos_top1"] for r in rs) / nw
        interf = sum(r["ref"]["competitor_in_interferer"] for r in rs) / nw
        chance = sum(r["ref"]["competitor_in_chance"] for r in rs) / nw
        cer = 100 * sum(r["_err"] for r in groups[key]) / max(sum(r["_ref"] for r in groups[key]), 1)
        miss = sum(r["_miss"] for r in groups[key])
        med = statistics.median(wrong_lp) if wrong_lp else float("nan")
        result["bins"][key] = {"n": len(groups[key]), "cer": cer, "miss": miss,
                               "rank_share": [h / tot for h in hist], "wrong_near": near, "wrong_far": far,
                               "wrong_median_logp": med, "wrong_eos_top1": eos,
                               "wrong_competitor_interferer": interf, "wrong_competitor_chance": chance}
        print(f"{key:18s} {len(groups[key]):5d} {cer:6.2f} {miss:4d} | "
              + " ".join(f"{100*h/tot:6.2f}%" for h in hist)
              + f" | {100*near:5.1f}% {100*far:5.1f}% {med:7.2f} {100*eos:4.1f}% {100*interf:5.1f}% {100*chance:5.1f}%")

    print("\n[2] enrollment swap: mean cosine (true vs other speaker's enrollment) of mixture-token states")
    n_audio = len(recs[0]["swap"]["audio_mix_cos"])
    n_llm = len(recs[0]["swap"]["llm_mix_cos"])
    audio_layers = [0, 4, 8, 12, 16, 20, n_audio - 1]
    llm_layers = [0, 4, 8, 12, 16, 20, 24, n_llm - 1]
    result["swap"] = {}
    for key in ("all", "overlap=0", "overlap=<0.9", "overlap=>=0.9"):
        rs = groups.get(key, [])
        if not rs:
            continue
        entry = {"audio_mix_cos": [statistics.fmean(r["swap"]["audio_mix_cos"][l] for r in rs) for l in range(n_audio)],
                 "llm_mix_cos": [statistics.fmean(r["swap"]["llm_mix_cos"][l] for r in rs) for l in range(n_llm)],
                 "llm_text_cos": [statistics.fmean(r["swap"]["llm_text_cos"][l] for r in rs) for l in range(n_llm)],
                 "regions": {}}
        for reg in REGIONS:
            have = [r["swap"]["regions"][reg] for r in rs if reg in r["swap"].get("regions", {})]
            if have:
                entry["regions"][reg] = {"samples": len(have),
                                         "audio": [statistics.fmean(h["audio"][l] for h in have) for l in range(n_audio)],
                                         "llm": [statistics.fmean(h["llm"][l] for h in have) for l in range(n_llm)]}
        result["swap"][key] = entry
        print(f"  {key}: audio layers {audio_layers} " + " ".join(f"{entry['audio_mix_cos'][l]:.3f}" for l in audio_layers)
              + f" | LLM layers {llm_layers} " + " ".join(f"{entry['llm_mix_cos'][l]:.3f}" for l in llm_layers))
        for reg, v in entry["regions"].items():
            print(f"     {reg:16s} ({v['samples']:5d}) audio " + " ".join(f"{v['audio'][l]:.3f}" for l in audio_layers)
                  + " | LLM " + " ".join(f"{v['llm'][l]:.3f}" for l in llm_layers))

    print("\n[3] selectivity vs errors (overlap >= 0.6, samples with all three regions)")
    sel_rows = []
    for r in recs:
        lab, reg = r["labels"], r["swap"].get("regions", {})
        ov = lab.get("target_overlap_ratio")
        if ov is None or ov < .6 or r["_miss"] or not all(k in reg for k in ("overlap", "interferer_only")):
            continue
        sel_rows.append(r)
    result["selectivity"] = {}
    for name, layer_key, layer in (("audio_last", "audio", n_audio - 1), ("llm_mid", "llm", 14), ("llm_last", "llm", n_llm - 1)):
        # how much more the representation moves where the target speaks than where only the interferer speaks
        sel = [(1 - r["swap"]["regions"]["overlap"][layer_key][layer]) - (1 - r["swap"]["regions"]["interferer_only"][layer_key][layer])
               for r in sel_rows]
        cer = [r["_err"] / max(r["_ref"], 1) for r in sel_rows]
        rho = spearman(sel, cer) if len(sel) >= 8 else None
        q = sorted(range(len(sel)), key=sel.__getitem__)
        lo, hi = q[: len(q) // 4], q[-(len(q) // 4):]
        cer_lo = 100 * sum(sel_rows[i]["_err"] for i in lo) / max(sum(sel_rows[i]["_ref"] for i in lo), 1)
        cer_hi = 100 * sum(sel_rows[i]["_err"] for i in hi) / max(sum(sel_rows[i]["_ref"] for i in hi), 1)
        if len(sel) < 8:
            print(f"  {name}: only {len(sel)} samples, skipped")
            continue
        result["selectivity"][name] = {"n": len(sel), "mean": statistics.fmean(sel) if sel else None, "spearman_vs_cer": rho,
                                       "cer_least_selective_quartile": cer_lo, "cer_most_selective_quartile": cer_hi}
        print(f"  {name:10s} n={len(sel)} mean selectivity {statistics.fmean(sel):+.4f}  spearman(sel, CER) {rho:+.3f}  "
              f"CER least-selective quartile {cer_lo:.2f}% vs most-selective {cer_hi:.2f}%")
    json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
