"""Spans a model leaves out of scene-asr (鼎桥场景普通识别) recordings, per delivery-evaluation model (README "结果：交付评估").

- coverage: output characters / reference characters (CJK chars and Latin words), by recording length.
- dialogues: the 30 s+ recordings with two or more speakers (metadata.segments). A turn is dropped when under 30% of
  its characters are matched in the output (difflib alignment of the whole reference); a dialogue has one speaker
  mostly dropped when that speaker's matched share is under 0.5 while the other's is over 0.85.
usage: python3 scene_asr_drops.py MODEL [MODEL ...]   (names of ae-runs/ctx-grpo-20261120-delivery-<model>)
"""
import collections
import difflib
import json
import re
import sys
from pathlib import Path

RUNS = Path("/workspace/data/ae-runs")
SCENE_ASR = "scene-asr-asronly-data.yaml"
units = lambda s: re.findall(r"[一-鿿]|[A-Za-z]+", s or "")
chars = lambda s: "".join(re.findall(r"[一-鿿]", s or ""))


def latest(model):
    runs = [r for r in sorted((RUNS / f"ctx-grpo-20261120-delivery-{model}/runs").glob("run-*"))
            if json.loads((r / "run.json").read_text())["config"].get("data_config", "").endswith(SCENE_ASR)
            and json.loads((r / "coverage.json").read_text()).get("complete")]
    return runs[-1]


def measure(run):
    samples = [json.loads(line) for line in open(run / "samples.jsonl")]
    pred = {r["id"]: r.get("prediction") for r in map(json.loads, open(run / "predictions.jsonl"))}
    cov = collections.defaultdict(lambda: [0, 0])
    dlg = collections.Counter()
    for s in samples:
        seconds = s["record"]["audio_slots"][0]["ref"]["duration"]
        cov["<30" if seconds < 30 else "30-45" if seconds < 45 else "45+"][0] += len(units(s["reference"]))
        cov["<30" if seconds < 30 else "30-45" if seconds < 45 else "45+"][1] += len(units(pred[s["id"]]))
        segs = s["record"].get("metadata", {}).get("segments") or []
        if seconds < 30 or len({x["speaker"] for x in segs}) < 2:
            continue
        ref, owner = "", []
        for k, x in enumerate(segs):
            ref += chars(x["text"]); owner += [k] * len(chars(x["text"]))
        got = [0] * len(ref)
        for b in difflib.SequenceMatcher(None, ref, chars(pred[s["id"]]), autojunk=False).get_matching_blocks():
            got[b.a:b.a + b.size] = [1] * b.size
        per = collections.defaultdict(lambda: [0, 0])
        for k, x in enumerate(segs):
            idx = [i for i, o in enumerate(owner) if o == k]
            share = sum(got[i] for i in idx) / max(1, len(idx))
            per[x["speaker"]][0] += sum(got[i] for i in idx); per[x["speaker"]][1] += len(idx)
            dlg["turns"] += 1
            if share < 0.3:
                dlg["dropped turns"] += 1
                dlg["dropped, first speaker" if x["speaker"] == segs[0]["speaker"] else "dropped, other speakers"] += 1
        shares = sorted(a / b for a, b in per.values())
        dlg["dialogues"] += 1
        dlg["one speaker mostly dropped"] += shares[0] < 0.5 and shares[-1] > 0.85
    return {k: round(b / a, 3) for k, (a, b) in sorted(cov.items())}, dict(dlg)


if __name__ == "__main__":
    for model in sys.argv[1:]:
        run = latest(model)
        print(model, run.name, *measure(run))
