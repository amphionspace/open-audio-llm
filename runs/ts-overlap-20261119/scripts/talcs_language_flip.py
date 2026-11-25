"""TALCS: utterances whose reference is mostly Chinese (>= 50% CJK chars among letters+CJK) but whose output has under
10% CJK, i.e. the Chinese speech came out in English; plus empty outputs and predicted_language counts. Per run."""
import json, re, sys, collections
cjk = lambda s: len(re.findall(r"[一-鿿]", s)); lat = lambda s: len(re.findall(r"[A-Za-z]", s))
for arg in sys.argv[1:]:
    nm, run = arg.split("=")
    c = collections.Counter()
    for o in map(json.loads, open(run + "/predictions.jsonl", encoding="utf-8")):
        if not o["source"].startswith("talcs@"): continue
        r, p = o["reference"], o.get("prediction") or ""
        c["n"] += 1; c["lang_" + str(o.get("predicted_language"))] += 1
        if not p.strip(): c["empty"] += 1; continue
        if cjk(r) / max(1, cjk(r) + lat(r) / 4) >= 0.5:  # ~4 letters per English word vs 1 CJK char per syllable
            c["zh_major"] += 1
            if cjk(p) / max(1, cjk(p) + lat(p) / 4) < 0.1: c["zh_major_out_english"] += 1
    print(nm, dict(sorted(c.items())))
