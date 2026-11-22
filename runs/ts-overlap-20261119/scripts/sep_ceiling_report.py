"""tasks/17 report: in-utterance error (S+D+I over reference chars, empty output = all deleted, AmphionEval zh tokens) on
the exported v12 overlap >= 0.9 subset, by interferer count, for the eval's greedy output, the reproduction on the
original mixture ('orig'), and separated candidates chosen
  oracle:    by the reference (upper bound of the separation front end);
  conf:      by 11.17's confidence (highest mean token logprob among non-empty TS outputs);
  conf+orig: as conf with the original mixture in the pool;
  either:    min(orig, oracle) per sample (upper bound of any orig/separated fusion).
Paired bootstrap (2,000) minus orig. usage: sep_ceiling_report.py SUBSET_DIR ASR_JSONL OUT_JSON  (amphion-eval python)"""
import collections, json, random, sys
from amphion_eval.integrations.open_audio_llm import scoring


def edits(r, h):
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1):
            cur[j] = min(prev[j - 1] + (a != b), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[-1]


def pick(cands):
    live = [c for c in cands if (c["text"] or "").strip()]
    return max(live, key=lambda c: c["mean_logprob"])["text"] if live else ""


def main():
    sub, asr, out = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{sub}/manifest.jsonl", encoding="utf-8")}
    outs = collections.defaultdict(dict)
    for l in open(asr, encoding="utf-8"):
        o = json.loads(l)
        outs[o["id"]][o["cand"]] = o
    per = collections.defaultdict(list)
    for sid, r in rows.items():
        ref = scoring.tokens(r["reference"], "zh")
        err = lambda t: edits(ref, scoring.tokens(t or "", "zh"))
        o = outs[sid]
        seps = [c for k, c in o.items() if k != "orig"]
        e = {"eval": err(r["greedy"]), "orig": err(o["orig"]["text"]), "oracle": min(err(c["text"]) for c in seps),
             "conf": err(pick(seps)), "conf+orig": err(pick(seps + [o["orig"]]))}
        e["either"] = min(e["orig"], e["oracle"])
        per[r["n_interferers"]].append((len(ref), e))
    rng, result = random.Random(0), {}
    for k, items in sorted(per.items()):
        n_ref = sum(n for n, _ in items)
        res = {"samples": len(items), "ref_chars": n_ref,
               "error": {m: round(100 * sum(e[m] for _, e in items) / n_ref, 2) for m in items[0][1]},
               "separated_better_samples": sum(e["oracle"] < e["orig"] for _, e in items),
               "separated_worse_samples": sum(e["oracle"] > e["orig"] for _, e in items), "minus_orig": {}}
        for m in ("conf", "conf+orig", "oracle"):
            diffs = []
            for _ in range(2000):
                s = [items[rng.randrange(len(items))] for _ in items]
                diffs.append(100 * (sum(e[m] for _, e in s) - sum(e["orig"] for _, e in s)) / sum(n for n, _ in s))
            diffs.sort()
            d = 100 * (sum(e[m] for _, e in items) - sum(e["orig"] for _, e in items)) / n_ref
            res["minus_orig"][m] = {"diff": round(d, 2), "lo": round(diffs[50], 2), "hi": round(diffs[1949], 2)}
        hard = [(n, e) for n, e in items if e["orig"] / n >= 0.3]
        hr = sum(n for n, _ in hard)
        res["orig_hard"] = {"samples": len(hard), "orig": round(100 * sum(e["orig"] for _, e in hard) / hr, 2),
                            "oracle": round(100 * sum(e["oracle"] for _, e in hard) / hr, 2)}
        result[f"interferers_{k}"] = res
        print(k, json.dumps(res, ensure_ascii=False))
    json.dump(result, open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
