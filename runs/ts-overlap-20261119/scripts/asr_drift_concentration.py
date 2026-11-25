"""Plain-ASR drift concentration: per suite, net extra errors vs base, the share from the top-k worst utterances, and
runaway outputs (hyp tokens > 2x ref tokens + 10, or finish_reason length) per run.  usage: BASE_RUN NAME=RUN ..."""
import json, sys, collections
from amphion_eval.integrations.open_audio_llm import scoring
from rapidfuzz.distance import Levenshtein
SUITES = ("talcs@", "wenetspeech@icefall-20260908:test_net", "wenetspeech@icefall-20260908:test_meeting")
def load(r): return {o["id"] + "|" + o["source"]: o for o in map(json.loads, open(r + "/predictions.jsonl", encoding="utf-8")) if o["source"].startswith(SUITES)}
def score(o):
    ref = scoring.tokens(o["reference"], "zh"); hyp = scoring.tokens(o.get("prediction") or "", "zh")
    return len(ref), Levenshtein.distance(ref, hyp), int(len(hyp) > 2 * len(ref) + 10 or o.get("finish_reason") == "length")
B = {k: score(o) for k, o in load(sys.argv[1]).items()}
print("base runaway", collections.Counter(k.split("|")[1].split("/")[0] for k, v in B.items() if v[2]))
for arg in sys.argv[2:]:
    nm, run = arg.split("="); C = {k: score(o) for k, o in load(run).items()}
    for s in SUITES:
        ks = [k for k in B if k.split("|")[1].startswith(s)]
        d = sorted((C[k][1] - B[k][1] for k in ks), reverse=True); ref = sum(B[k][0] for k in ks); net = sum(d)
        print(f"{nm:5s} {s[:40]:40s} n {len(ks):6d} net {net:+6d} ({100*net/ref:+.2f} pts)  top10 {sum(d[:10]):+6d}  top50 {sum(d[:50]):+6d}  "
              f"without top50 {100*(net-sum(d[:50]))/ref:+.2f} pts  runaway base {sum(B[k][2] for k in ks)} new {sum(C[k][2] for k in ks)}")
