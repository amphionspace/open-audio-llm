"""Replay the request selection of ft_enc_replay.py Meetings on a meeting_train_plan.py plan, without audio: per corpus
and per target-speech-seconds bin of the sent audio, the share of requests that are negatives.
usage: meeting_plan_neg_share.py PLAN.json [N_REQUESTS]   (constants copied from ft_enc_replay.py)"""
import json, random, collections, sys
plan = json.load(open(sys.argv[1]))
W = plan["windows"]
M_WEIGHTS = {"alimeeting": 3.5, "aishell4": 3.8, "ramc": 1.7}
NEG_PROB, CROP_PROB, CROP_PAD, TALK_MARGIN = 0.5, 0.5, 0.5, 0.3
rng = random.Random(0); corpora = [c for c in M_WEIGHTS if W.get(c)]
EDGES = [1, 2, 3, 5, 10]
def b(v): return next((f"<{e}" for e in EDGES if v < e), f">={EDGES[-1]}")
cnt = collections.defaultdict(collections.Counter)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300000; done = 0
while done < N:
    c = rng.choices(corpora, [M_WEIGHTS[x] for x in corpora])[0]; w = rng.choice(W[c])
    lo, hi = w["start"], w["start"] + w["duration"]
    crop = rng.random() < CROP_PROB
    if crop:
        own = [u for u in w["utts"] if u[0] == w["target"]]; k = rng.randrange(len(own))
        us, ud = w["start"] + own[k][1], own[k][2]
        prev_end = max([w["start"] + u[1] + u[2] for u in own[:k]], default=lo)
        next_start = min([w["start"] + u[1] for u in own[k + 1:]], default=hi)
        lo = max(lo, prev_end, us - rng.uniform(0, CROP_PAD)); hi = min(hi, next_start, us + ud + rng.uniform(0, CROP_PAD))
        if hi - lo < 0.3: continue
        if not w["pieces"][k].strip("，。、；：！？,.;:!? "): continue
    talkers = sorted({u[0] for u in w["utts"] if w["start"] + u[1] < hi + TALK_MARGIN and w["start"] + u[1] + u[2] > lo - TALK_MARGIN})
    if w["target"] not in talkers: continue
    cand = [x for x in w["neg"] if x[0] not in talkers]
    neg = rng.random() < NEG_PROB and bool(cand)
    tsec = sum(max(0, min(hi, w["start"] + u[1] + u[2]) - max(lo, w["start"] + u[1])) for u in w["utts"] if u[0] == w["target"])
    done += 1
    for key in (c, "all"):
        cnt[(key, "crop" if crop else "window", b(tsec))]["neg" if neg else "pos"] += 1
        cnt[(key, "any", b(tsec))]["neg" if neg else "pos"] += 1
for key in corpora + ["all"]:
    print("==", key)
    for kind in ("any", "crop", "window"):
        row = []
        for e in [f"<{x}" for x in EDGES] + [f">={EDGES[-1]}"]:
            k = cnt[(key, kind, e)]; n = k["pos"] + k["neg"]
            row.append(f"{e}: n {n:6d} neg {100*k['neg']/max(1,n):4.1f}%")
        print(f"  {kind:6s} " + " | ".join(row))
