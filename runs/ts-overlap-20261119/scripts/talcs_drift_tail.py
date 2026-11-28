"""TALCS tails: worst and best per-utterance error changes vs base for each run, overlap of the worst-20 sets, and what
the worst outputs look like (length, CJK share, finish reason).  usage: BASE_RUN NAME=RUN NAME=RUN"""
import json, re, sys
from amphion_eval.integrations.open_audio_llm import scoring
from rapidfuzz.distance import Levenshtein
def load(r): return {o["id"]: o for o in map(json.loads, open(r + "/predictions.jsonl", encoding="utf-8")) if o["source"].startswith("talcs@")}
def err(o, ref): return Levenshtein.distance(ref, scoring.tokens(o.get("prediction") or "", "zh"))
B = load(sys.argv[1]); refs = {i: scoring.tokens(o["reference"], "zh") for i, o in B.items()}
eb = {i: err(o, refs[i]) for i, o in B.items()}; worst = {}
cjk = lambda s: len(re.findall(r"[一-鿿]", s)); lat = lambda s: len(re.findall(r"[A-Za-z]+", s))
for arg in sys.argv[2:]:
    nm, run = arg.split("="); C = load(run); d = sorted(((err(C[i], refs[i]) - eb[i], i) for i in B), reverse=True)
    worst[nm] = [i for _, i in d[:20]]
    print(nm, "worst10", sum(x for x, _ in d[:10]), "best10", sum(x for x, _ in d[-10:]), "net", sum(x for x, _ in d))
    for x, i in d[:8]:
        o = C[i]; p = o.get("prediction") or ""
        print(f"   +{x:5d} ref {len(refs[i]):4d} tok  hyp_cjk {cjk(p):4d} hyp_words {lat(p):4d}  base_err {eb[i]:4d} finish {o.get('finish_reason')} tokens {o.get('completion_tokens')}  {i}")
    for x, i in d[-3:]: print(f"   {x:6d} (better) ref {len(refs[i])} base_err {eb[i]} base_finish {B[i].get('finish_reason')} {i}")
a, b = list(worst)[:2]; print("worst-20 overlap", a, b, len(set(worst[a]) & set(worst[b])))
