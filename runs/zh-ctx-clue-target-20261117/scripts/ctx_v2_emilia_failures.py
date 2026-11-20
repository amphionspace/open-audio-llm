"""Why Emilia items fail (2026-11-20 analysis after question 3), from ae's per-item scores and predictions.

Explicit items that need a correction:
- think off/on, by subtype: right / kept the common spelling / other, and whether the explanation was heard;
- for kept-common items whose explanation contains the right character: was the explanation transcribed verbatim
  (heard but not applied), bent to the wrong character, or changed otherwise.
Layout of synthetic training vs Emilia: where the explanation sits, how many target mentions, and the character gap.
Implicit items that need a correction: accuracy by the target word's jieba frequency (a rough rarity proxy; single
characters score high regardless of context). Needs jieba; run with an env that has it.
usage: ctx_v2_emilia_failures.py   (reads the latest runs of template, template-think, template-implicit-noexplain)
"""
import glob
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict

RUNS = "/workspace/data/ae-runs/ctx-clue-target-20261117-{}/runs/run-*"
MINED = "/workspace/data/datasets/context_correction_zh/versions/v2-20261116/records/mined_real.jsonl.gz"
SYNTH = "/workspace/data/datasets/context_correction_synth_zh/versions/train-v1-20261114/records/{}.jsonl.gz"
HAN = re.compile(r"[^一-鿿]")
TARGET = re.compile(r"\[\[T:([^|\]]+)\|([^\]]+)\]\]")
EVIDENCE = re.compile(r"\[\[D:")


def latest(name):
    return sorted(glob.glob(RUNS.format(name)))[-1]


def scores(name):
    out = {}
    for line in open(latest(name) + "/context-correction-scores.jsonl"):
        s = json.loads(line)
        if s["kind"] == "mined" and s["source"] == "mined:emilia_zh":
            out[s["id"]] = s
    return out


def predictions(name):
    return {x["id"]: HAN.sub("", x["prediction"].split("</think>")[-1])
            for x in map(json.loads, open(latest(name) + "/predictions.jsonl"))}


def layout(cc):
    mk = cc.get("marked_text") or ""
    ts = [m.start() for m in TARGET.finditer(mk)]
    ds = [m.start() for m in EVIDENCE.finditer(mk)]
    return cc.get("position"), min(len(ts), 3), (min(abs(a - b) for a in ts for b in ds) if ts and ds else None)


def pct(a, n):
    return f"{100 * a / n:.0f}" if n else "—"


def main():
    meta = {}
    for line in gzip.open(MINED, "rt"):
        r = json.loads(line)
        cc = r["labels"]["context_correction"]
        if cc["source"] == "mined:emilia_zh" and not r["metadata"]["corpus_transcript_correct"]:
            meta[r["id"]] = cc
    off, on, imp = scores("template"), scores("template-think"), scores("template-implicit-noexplain")
    explicit = [i for i in off if i in meta and meta[i]["category"] == "explicit"]

    print("## 显式说明题（需纠正，template）\n")
    print("| 子类 | 条数 | think 关：对 / 写成常用字 / 其他 | 听出说明 | think 开：对 |\n|---|---|---|---|---|")
    groups = defaultdict(list)
    for i in explicit:
        groups["全部"].append(i)
        groups[meta[i]["subtype"]].append(i)
    for name, ids in sorted(groups.items(), key=lambda x: -len(x[1])):
        n = len(ids)
        ok = sum(off[i]["target_ok"] for i in ids)
        alt = sum(bool(off[i]["target_alternative"]) for i in ids)
        heard = [off[i]["evidence_ok"] for i in ids if off[i]["evidence_ok"] is not None]
        print(f"| {name} | {n} | {pct(ok, n)} / {pct(alt, n)} / {pct(n - ok - alt, n)} | {pct(sum(heard), len(heard))} "
              f"| {pct(sum(on[i]['target_ok'] for i in ids), n)} |")

    print("\n## 写成常用字时，说明本身转写得怎样（说明里含正确字的题）\n")
    print("| | think 关 | think 开 |\n|---|---|---|")
    counts = {}
    for label, sc, pred in (("off", off, predictions("template")), ("on", on, predictions("template-think"))):
        c = Counter()
        for i in explicit:
            segs = meta[i]["segments"]
            targets = [(g["text"], g.get("alt")) for g in segs if g["role"] == "target"]
            hits = []
            for e in (HAN.sub("", g["text"]) for g in segs if g["role"] == "evidence"):
                for t, a in targets:
                    hits += [(e, e.replace(ch, a[k])) for k, ch in enumerate(t)
                             if a and k < len(a) and ch != a[k] and ch in e]
            if not hits:
                c["说明里没有正确字"] += 1
            elif sc[i]["target_alternative"]:
                p = pred[i]
                c["说明逐字转对，目标没改" if any(e in p for e, _ in hits) else
                  "说明被改成错字" if any(b in p for _, b in hits) else "说明转写有别的出入"] += 1
        counts[label] = c
    for key in ("说明逐字转对，目标没改", "说明被改成错字", "说明转写有别的出入", "说明里没有正确字"):
        print(f"| {key} | {counts['off'][key]} | {counts['on'][key]} |")

    print("\n## 说明的位置：合成训练数据与 Emilia\n")
    print("| | 条数 | 说明在目标之后 | 目标出现 ≥2 次 | 目标与说明的字距中位数 |\n|---|---|---|---|---|")
    rows = [("合成 train", [json.loads(x)["labels"]["context_correction"] for x in gzip.open(SYNTH.format("train"), "rt")]),
            ("合成 train_ts", [json.loads(x)["labels"]["context_correction"] for x in gzip.open(SYNTH.format("train_ts"), "rt")]),
            ("Emilia 显式需纠正", [meta[i] for i in explicit])]
    for name, ccs in rows:
        lay = [layout(cc) for cc in ccs]
        gaps = sorted(g for _, _, g in lay if g is not None)
        print(f"| {name} | {len(lay)} | {pct(sum(p == 'after' for p, _, _ in lay), len(lay))}% "
              f"| {pct(sum(t >= 2 for _, t, _ in lay), len(lay))}% | {gaps[len(gaps) // 2]} |")

    implicit = [i for i in off if i in meta and meta[i]["category"] == "implicit"]
    print("\n## 隐式题（需纠正）：按目标词的 jieba 词频（think 关，template 对 template-implicit-noexplain）\n")
    try:
        import jieba
    except ImportError:
        print("（没有 jieba，跳过）")
        return
    jieba.setLogLevel(60)
    jieba.initialize()
    freq = jieba.dt.FREQ
    words = {i: TARGET.search(meta[i]["marked_text"]).groups() for i in implicit}
    both_right = [freq.get(words[i][0], 0) for i in implicit if off[i]["target_ok"] and imp[i]["target_ok"]]
    both_wrong = [freq.get(words[i][0], 0) for i in implicit if not off[i]["target_ok"] and not imp[i]["target_ok"]]
    print(f"- 两组都对 {len(both_right)} 条：目标词不在词典 {pct(sum(f == 0 for f in both_right), len(both_right))}%，"
          f"词频中位数 {statistics.median(both_right):.0f}")
    print(f"- 两组都错 {len(both_wrong)} 条：目标词不在词典 {pct(sum(f == 0 for f in both_wrong), len(both_wrong))}%，"
          f"词频中位数 {statistics.median(both_wrong):.0f}\n")
    print("| 目标词词频 | 条数 | template | template-implicit-noexplain |\n|---|---|---|---|")
    for lo, hi, name in ((0, 0, "不在词典"), (1, 99, "1–99"), (100, 999, "100–999"), (1000, float("inf"), "≥1000")):
        ids = [i for i in implicit if lo <= freq.get(words[i][0], 0) <= hi]
        print(f"| {name} | {len(ids)} | {pct(sum(off[i]['target_ok'] for i in ids), len(ids))} "
              f"| {pct(sum(imp[i]['target_ok'] for i in ids), len(ids))} |")
    common = [i for i in implicit if freq.get(words[i][0], 0) >= 100 and freq.get(words[i][1], 0) >= 100]
    print(f"| 正确和错误写法都是常用词（≥100） | {len(common)} | {pct(sum(off[i]['target_ok'] for i in common), len(common))} "
          f"| {pct(sum(imp[i]['target_ok'] for i in common), len(common))} |")


if __name__ == "__main__":
    main()
