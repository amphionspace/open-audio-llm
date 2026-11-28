"""tasks/29 first-speaker causal test (README "因果验证：是不是最先听到的声音被当成目标"). Per model: the original
scene-asr run and its cut-first / cut-same runs (police_scene_bench@scene-asr-cut-20261128, same record ids). Speaker
roles come from the original dialogue: S1 = first speaker, S2 = first speaker different from S1, S3+ = the rest. A turn
is dropped when under 30% of its characters are matched in the output (difflib over the whole reference, as
scene_asr_drops.py). Only turns present in both the original and the variant are compared; variant dialogues with one
speaker left are skipped. usage: scene_cut_report.py NAME=ORIG_RUN,CUTFIRST_RUN,CUTSAME_RUN ...   (python3)"""
import collections, difflib, json, re, sys

chars = lambda s: "".join(re.findall(r"[一-鿿]", s or ""))


def load(run):
    pred = {r["id"]: r.get("prediction") for r in map(json.loads, open(f"{run}/predictions.jsonl"))}
    return {s["id"]: (s["record"].get("metadata", {}).get("segments") or [], pred.get(s["id"]), s["record"]["audio_slots"][0]["ref"]["duration"])
            for s in map(json.loads, open(f"{run}/samples.jsonl"))}


def turn_drops(segs, prediction):
    ref, owner = "", []
    for k, x in enumerate(segs): ref += chars(x["text"]); owner += [k] * len(chars(x["text"]))
    got = [0] * len(ref)
    for b in difflib.SequenceMatcher(None, ref, chars(prediction), autojunk=False).get_matching_blocks(): got[b.a:b.a + b.size] = [1] * b.size
    out = []
    for k in range(len(segs)):
        idx = [i for i, o in enumerate(owner) if o == k]
        out.append(sum(got[i] for i in idx) / max(1, len(idx)) < 0.3)
    return out


def roles(segs):
    s1 = segs[0]["speaker"]; s2 = next((x["speaker"] for x in segs if x["speaker"] != s1), None)
    return {x["speaker"]: "S1" if x["speaker"] == s1 else "S2" if x["speaker"] == s2 else "S3+" for x in segs}


for arg in sys.argv[1:]:
    name, runs = arg.split("=", 1); orig, first, same = (load(r) for r in runs.split(","))
    for vname, var, offset in (("cut-first", first, 1), ("cut-same", same, 0)):
        c = collections.defaultdict(lambda: [0, 0, 0])  # role -> turns, dropped in original, dropped in variant
        dialogues = 0
        for i, (vsegs, vpred, _) in var.items():
            if i not in orig or len({x["speaker"] for x in vsegs}) < 2: continue
            osegs, opred, odur = orig[i]
            if odur < 30 or len({x["speaker"] for x in osegs}) < 2: continue
            base = osegs[offset:offset + len(vsegs)]
            if [x["text"] for x in base] != [x["text"] for x in vsegs]: continue  # turn alignment must hold
            dialogues += 1; role = roles(osegs)
            od = turn_drops(osegs, opred)[offset:offset + len(vsegs)]; vd = turn_drops(vsegs, vpred)
            for x, a, b in zip(vsegs, od, vd):
                r = c[role[x["speaker"]]]; r[0] += 1; r[1] += a; r[2] += b
        print(f"{name:5s} {vname:9s} dialogues {dialogues:3d}  " + "  ".join(
            f"{k}: {v[1]}/{v[0]} ({100*v[1]/max(1,v[0]):4.1f}%) -> {v[2]} ({100*v[2]/max(1,v[0]):4.1f}%)" for k, v in sorted(c.items())))
    print()
