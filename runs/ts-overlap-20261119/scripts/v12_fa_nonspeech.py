"""v12 false alarms on scenes with no interfering speech (labels.n_interferers == 0: non-speech, music, song), per model,
with the outputs: hallucinated text where nothing in the scene speaks.
Usage (amphion-eval python): v12_fa_nonspeech.py SAMPLES.jsonl base=BASE_RUN NAME=RUN [NAME=RUN ...]"""
import json, sys, collections
from amphion_eval.integrations.open_audio_llm import scoring
samples = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
runs = [a.split("=", 1) for a in sys.argv[2:]]
outs = {}
for name, run in runs:
    o = {}
    for l in open(f"{run}/predictions.jsonl"):
        p = json.loads(l)
        if p.get("task") == "ts_asr" and p.get("language") == "zh" and not scoring.tokens(p.get("reference") or "", "zh"):
            o[p["id"]] = (scoring.tokens(p.get("prediction") or "", "zh"), p.get("raw_text"), p.get("completion_tokens"))
    outs[name] = o
ids = [i for i in outs["base"] if (samples[i]["record"].get("labels") or {}).get("n_interferers") == 0]
c = collections.Counter(samples[i]["record"]["labels"]["scenario"] for i in ids)
print("n_interferers=0 scenarios:", dict(c))
for name in outs:
    fa = [i for i in ids if outs[name][i][0]]
    print(name, len(fa), collections.Counter(samples[i]["record"]["labels"]["scenario"] for i in fa).most_common())
print("\nexamples (a002 / t25 / a001 / base raw):")
for i in ids:
    if any(outs[n][i][0] for n in outs):
        print(f"{i:24s}", " | ".join(f"{n}:{''.join(outs[n][i][0])[:30]}" for n in outs))
