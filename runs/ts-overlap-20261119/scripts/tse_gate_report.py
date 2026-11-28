"""Score the CAM++-gated outputs (tse_gate.py cosines): gated = extracted / remix transcript if cos(extracted, enrollment)
> cos(mixture, enrollment) and non-empty, else the mixture transcript. By set and group, paired bootstrap vs mixture.
usage: tse_gate_report.py SUBSET_DIR SAFETY_DIR W_DIR GATE_DIR OUT_JSON TAG [TAG ...]   (amphion-eval python)"""
import collections, json, random, sys
from amphion_eval.integrations.open_audio_llm import scoring
sub, saf, w, gd, out, tags = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:]
def edits(r, h):
    prev = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, b in enumerate(h, 1): cur[j] = min(prev[j-1] + (a != b), prev[j] + 1, cur[j-1] + 1)
        prev = cur
    return prev[-1]
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{sub}/manifest.jsonl", encoding="utf-8")}
st = json.load(open(f"{sub}/oracle-stats.json")); srows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{saf}/manifest.jsonl", encoding="utf-8")}
rng = random.Random(0); res = {}
for tag in tags:
    gate = json.load(open(f"{gd}/gate-{tag}.json"))
    for setname, rr, ids, path in (("heavy", rows, list(st), f"{w}/asr-heavy-{tag}.jsonl"), ("safety", srows, list(srows), f"{w}/asr-safety-{tag}.jsonl")):
        o = collections.defaultdict(dict)
        for l in open(path, encoding="utf-8"):
            x = json.loads(l); o[x["id"]][x["cand"]] = (x["text"] or "").strip()
        groups = collections.defaultdict(list); fa = collections.Counter()
        for i in ids:
            g = gate[i]; use = g["cos_tse"] > g["cos_mix"]
            gt = o[i]["tse"] if use and o[i]["tse"] else o[i]["mix"]; gr = o[i]["remix"] if use and o[i]["remix"] else o[i]["mix"]
            ref = scoring.tokens(rr[i]["reference"], "zh")
            if not ref:
                fa["n"] += 1; fa["mix"] += bool(o[i]["mix"]); fa["gate_tse"] += bool(gt); fa["gate_remix"] += bool(gr); continue
            x = {"n": len(ref), "use": use}
            for c, t in (("mix", o[i]["mix"]), ("gate_tse", gt), ("gate_remix", gr)): x[c] = edits(ref, scoring.tokens(t, "zh"))
            grp = f"interferers_{rr[i]['n_interferers']}" if setname == "heavy" else rr[i]["group"]
            groups[grp].append(x); groups[f"{setname}_all"].append(x)
        for grp, xs in sorted(groups.items()):
            n = sum(x["n"] for x in xs); r = {"samples": len(xs), "used_extraction": round(sum(x["use"] for x in xs) / len(xs), 3)}
            for c in ("mix", "gate_tse", "gate_remix"): r[c] = round(100 * sum(x[c] for x in xs) / n, 2)
            for c in ("gate_tse", "gate_remix"):
                d = []
                for _ in range(2000):
                    s = [xs[rng.randrange(len(xs))] for _ in xs]; d.append(100 * sum(x[c] - x["mix"] for x in s) / sum(x["n"] for x in s))
                d.sort(); r[f"{c}_minus_mix"] = [round(r[c] - r["mix"], 2), round(d[50], 2), round(d[1949], 2)]
            res[f"{tag}/{setname}/{grp}"] = r; print(tag, setname, grp, json.dumps(r))
        if fa["n"]:
            res[f"{tag}/{setname}/absent_fa"] = {k: round(100 * v / fa["n"], 1) for k, v in fa.items() if k != "n"}; print(tag, "absent FA%", res[f"{tag}/{setname}/absent_fa"])
json.dump(res, open(out, "w"), indent=1)
