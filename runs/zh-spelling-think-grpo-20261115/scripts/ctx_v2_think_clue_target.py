"""09-ctx-v2-eval: what avg3's think names (the quoted clue, the written character) against the gold edits of
the context-correction records, and how that goes with the correction (diagnostic over ae's per-item outputs).

A clue counts as found when a quote in the think shares 2+ consecutive Han characters with a gold evidence span;
the target counts as written when every character that differs between the wrong and right target word is in a
「写作」 quote. Correct-kind items only (synthetic) and mined items whose corpus transcript needed a correction.
With the think-off run as a second argument it also splits the Emilia accuracy by what the think-on run wrote
(the spelling template, or 「这段没有说明字怎么写」), think off against think on.
usage: ctx_v2_think_clue_target.py <ae run dir of the think-on run> [<ae run dir of the think-off run>]
"""
import collections
import difflib
import gzip
import json
import re
import sys

RUN = sys.argv[1]
OFF = sys.argv[2] if len(sys.argv) > 2 else None
REC = "/workspace/data/datasets/context_correction_zh/versions/{v}/records/{s}.jsonl.gz"
gold = {}
for v, splits in {"v1-20261018": ["synthetic", "mined_real"], "v2-20261116": ["synthetic_indep", "mined_real"]}.items():
    for s in splits:
        with gzip.open(REC.format(v=v, s=s), "rt") as f:
            lines = f.readlines()
        for line in lines:
            r = json.loads(line); cc = r["labels"]["context_correction"]
            tchars, evid = set(), []
            for seg in cc["segments"]:
                if seg["role"] == "target" and seg.get("alt") is not None:
                    sm = difflib.SequenceMatcher(None, seg["alt"], seg["text"])
                    for op, a1, a2, b1, b2 in sm.get_opcodes():
                        if op != "equal":
                            tchars |= set(seg["text"][b1:b2])
                if seg["role"] == "evidence":
                    evid.append(seg["text"])
            gold[r["id"]] = {"tchars": tchars, "evid": evid, "need": not r["metadata"].get("corpus_transcript_correct", False)}
def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


pred = {p["id"]: p for p in jsonl(f"{RUN}/predictions.jsonl")}
HAN = re.compile(r"[一-鿿]")


def parse(think):
    quotes = re.findall(r"「([^」]*)」", think or "")
    writes = re.findall(r"写作「([^」]*)」", think or "")
    return quotes, writes


def clue_hit(quotes, evid):
    # the quoted clue shares a run of >=2 Han chars with a gold evidence span (or the whole span when shorter)
    for q in quotes:
        for e in evid:
            for piece in re.split(r"[；;]", e):
                h = "".join(HAN.findall(piece))
                if h and (h in q or any(h[i:i + 2] in q for i in range(len(h) - 1))):
                    return True
    return False


rows = collections.defaultdict(collections.Counter)
for s in jsonl(f"{RUN}/context-correction-scores.jsonl"):
    if s["mode"] != "asr" or s["source"] not in ("synthetic", "synthetic_indep", "mined:emilia_zh"):
        continue
    g = gold[s["id"]]
    if s["kind"] == "confirm" or (s["kind"] == "mined" and not g["need"]) or not g["tchars"]:
        continue
    quotes, writes = parse(pred[s["id"]].get("think"))
    named = set("".join(writes))
    key = (s["source"], s["subtype"] if s["source"] != "mined:emilia_zh" else s["category"])
    c = rows[key]
    multi = len(g["tchars"]) > 1
    c["n"] += 1; c["ok"] += s["target_ok"]
    c["multi"] += multi; c["multi_ok"] += multi and s["target_ok"]
    hit = clue_hit(quotes, g["evid"]); c["clue"] += hit; c["clue_ok"] += hit and s["target_ok"]
    allc = g["tchars"] <= named; c["allchar"] += allc; c["allchar_ok"] += allc and s["target_ok"]
    c["anychar"] += bool(g["tchars"] & named)
    c["writes>1"] += len(writes) > 1
    c["clue_wrongchar"] += hit and not (g["tchars"] & named)


def pct(a, b):
    return f"{100 * a / b:.0f}" if b else "—"


print("| 来源 | 子类 | 需纠正条数 | 纠正成功 | 多字目标占比 | 多字目标成功 | think 线索对上 | 线索对上时成功 | think 写出全部目标字 | 写出全部目标字时成功 | 线索对、字全错 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (src, sub), c in sorted(rows.items()):
    n = c["n"]
    print(f"| {src} | {sub} | {n} | {pct(c['ok'], n)} | {pct(c['multi'], n)} | {pct(c['multi_ok'], c['multi'])} | {pct(c['clue'], n)} | {pct(c['clue_ok'], c['clue'])} | {pct(c['allchar'], n)} | {pct(c['allchar_ok'], c['allchar'])} | {pct(c['clue_wrongchar'], n)} |")
tot = collections.Counter()
for c in rows.values():
    tot.update(c)
print("think 里出现多于一个「写作」的条数：", tot["writes>1"], "/", tot["n"])

if OFF:
    def scores(run):
        return {s["id"]: s for s in jsonl(f"{run}/context-correction-scores.jsonl") if s["source"] == "mined:emilia_zh"}
    off, on = scores(OFF), scores(RUN)
    split = collections.defaultdict(lambda: [0, 0, 0])
    for item, s in on.items():
        kind = "无说明" if (pred[item].get("think") or "").startswith("这段没有说明") else "拼写模板"
        cell = split[(s["category"], "需纠正" if gold[item]["need"] else "原转写已对", kind)]
        cell[0] += 1; cell[1] += off[item]["target_ok"]; cell[2] += s["target_ok"]
    print()
    print("| 类别 | 原转写 | think 开写了什么 | 条数 | think 关准确率 | think 开准确率 |")
    print("|---|---|---|---|---|---|")
    for key, (n, a, b) in sorted(split.items()):
        print(f"| {key[0]} | {key[1]} | {key[2]} | {n} | {100 * a / n:.1f} | {100 * b / n:.1f} |")
