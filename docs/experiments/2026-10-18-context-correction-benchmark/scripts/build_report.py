#!/usr/bin/env python3
"""Build the HTML page that introduces the context-correction benchmark and its baselines.

Reads the registered records, the source directory and the two AmphionEval runs, copies the
example clips next to the page, and writes ``<out>/index.html``. Run it in an environment
with AmphionEval (the branch with ``context_correction``) installed:

    python scripts/build_report.py --out /workspace/workspace/mingdong/sites/context-correction-v1
    python -m http.server <port> --bind 0.0.0.0 --directory <out>
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from amphion_eval.integrations.open_audio_llm import context_correction as cc

DATA = Path("/workspace/data")
DATASET = DATA / "datasets/context_correction_zh"
VERSION = "v1-20261018"
SOURCE = DATASET / "source" / VERSION
RECORDS = DATASET / "versions" / VERSION / "records"
RUNS = {
    "34479": DATA / "ae-runs/checkpoint-34479/runs/run-20261019T011627Z-c082c6aa",
    "qwen3": DATA / "ae-runs/qwen3-asr-1.7b/runs/run-20261019T011627Z-31dd04ba",
}
WANDB = "https://wandb.ai/1016097967-amphion/open-audio-llm/runs/"
COLUMNS = [("34479", "asr", "34479 · 普通 ASR"), ("34479", "ts_asr", "34479 · TS 模式"),
           ("qwen3", "asr", "Qwen3-ASR-1.7B 官方")]
SUBTYPES = ["组词", "拆字", "偏旁结构", "否定对比", "同名一致性", "隐式语义"]
# Counts not cheap to recompute (scanning the 1.2 M-window file); from the experiment README.
WINDOWS = 1_194_252


def jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def esc(text):
    return html.escape(str(text), quote=True)


def pct(value, digits=1):
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def outcome(segments, prediction):
    """What the output wrote at each target: (gold, alternative, written, class)."""
    reference, owner = [], []
    for index, segment in enumerate(segments):
        chars = cc.characters(segment["text"])
        reference += chars
        owner += [index] * len(chars)
    paired = cc._paired(reference, cc.characters(prediction))
    results = []
    for index, segment in enumerate(segments):
        if segment.get("role") != "target":
            continue
        positions = [k for k, o in enumerate(owner) if o == index]
        alternative = cc.characters(segment.get("alt") or "")
        written = "".join(paired.get(p, "□") for p in positions)
        tested = list(range(len(positions)))
        if len(alternative) == len(positions):
            tested = [i for i in tested if alternative[i] != reference[positions[i]]] or tested
        got = [paired.get(positions[i]) for i in tested]
        if got == [reference[positions[i]] for i in tested]:
            kind = "gold"
        elif len(alternative) == len(positions) and got == [alternative[i] for i in tested]:
            kind = "alt"
        else:
            kind = "other"
        results.append((segment["text"], segment.get("alt") or "", written, kind))
    return results


def item_class(outcomes):
    kinds = {k for *_, k in outcomes}
    return "gold" if kinds == {"gold"} else ("alt" if "alt" in kinds else "other")


def load():
    records = {}
    for split in ("synthetic", "synthetic_ts", "mined_real"):
        for record in jsonl(RECORDS / f"{split}.jsonl.gz"):
            records[record["id"]] = record
    index = {e["cut_id"]: DATA / e["relative_path"]
             for e in jsonl(DATASET / "versions" / VERSION / "audio-index.jsonl.gz")}
    runs = {}
    for name, root in RUNS.items():
        runs[name] = {
            "summary": json.loads((root / "summary.json").read_text())["context_correction"],
            "scored": jsonl(root / "context-correction-scores.jsonl"),
            "predictions": {p["id"]: p for p in jsonl(root / "predictions.jsonl")},
        }
    return records, index, runs


def blocks(runs):
    """Per model and mode: overall, per subtype, per distance and position."""
    table = {}
    for name, mode, _ in COLUMNS:
        rows = [r for r in runs[name]["scored"] if r["mode"] == mode]
        synthetic = [r for r in rows if r["source"] == "synthetic"]
        table[(name, mode)] = {
            "all": cc._block(synthetic),
            "mined": cc._block([r for r in rows if r["kind"] == "mined"]),
            "subtype": {s: cc._block([r for r in synthetic if r["subtype"] == s]) for s in SUBTYPES},
            "distance": {d: cc._block([r for r in synthetic if r["distance"] == d])
                         for d in ("adjacent", "near", "far")},
            "position": {p: cc._block([r for r in synthetic if r["position"] == p])
                         for p in ("before", "after")},
        }
    return table


def outcomes_by_kind(records, runs):
    """Share of gold / alternative / other spellings per model, mode and item kind."""
    counts = defaultdict(Counter)
    for name, run in runs.items():
        for record_id, prediction in run["predictions"].items():
            record = records[record_id]
            label = record["labels"]["context_correction"]
            if label["category"] == "implicit" or label["kind"] == "mined":
                continue
            kind = item_class(outcome(label["segments"], prediction["prediction"]))
            counts[(name, record["task"], label["kind"])][kind] += 1
    return counts


def mined_split(records, runs):
    result = {}
    for name, run in runs.items():
        rows = [r for r in run["scored"] if r["kind"] == "mined"]
        for flag in (True, False):
            chosen = [r for r in rows if records[r["id"]]["metadata"]["corpus_transcript_correct"] is flag]
            result[(name, flag)] = (len(chosen), sum(r["target_ok"] for r in chosen) / len(chosen),
                                    sum(r["target_alternative"] for r in chosen) / len(chosen))
    return result


def funnel(records):
    pairs = sum("pair_id" in p for p in jsonl(SOURCE / "pairs.jsonl"))
    accepted = len(jsonl(SOURCE / "items.jsonl")) // 2
    qc = jsonl(SOURCE / "qc.jsonl")
    mined_root = SOURCE / "mined/wenetspeech_podcast"
    flagged = sum(len(r["flagged"]) for r in jsonl(mined_root / "scan.jsonl"))
    labels = jsonl(mined_root / "labels.jsonl")
    extracted = len(jsonl(mined_root / "items.jsonl"))
    splits = Counter(r["id"].split(":")[1] for r in records.values())
    return {
        "pairs": pairs, "accepted": accepted, "clips": len(qc), "clips_pass": sum(r["pass"] for r in qc),
        "synthetic_pairs": splits["synthetic"] // 2, "ts_pairs": splits["synthetic_ts"] // 2,
        "windows": WINDOWS, "flagged": flagged, "labeled": len(labels),
        "agreed": sum(not r["issues"] for r in labels), "extracted": extracted,
        "mined": splits["mined_real"],
    }


def pick_examples(records, runs):
    """Deterministic examples: shortest synthetic pairs of each kind of behaviour of 34479."""
    run = runs["34479"]["predictions"]
    pairs = defaultdict(dict)
    for record in records.values():
        label = record["labels"]["context_correction"]
        if record["task"] == "asr" and label["kind"] in ("correct", "confirm"):
            pairs[label["pair_id"]][label["kind"]] = record
    verdict = {}
    for pair_id, members in pairs.items():
        if len(members) == 2:
            verdict[pair_id] = {k: item_class(outcome(r["labels"]["context_correction"]["segments"],
                                                      run[r["id"]]["prediction"])) for k, r in members.items()}

    def shortest(predicate, subtype=None, limit=1):
        chosen = sorted(
            (p for p, v in verdict.items() if predicate(v)
             and (subtype is None or pairs[p]["correct"]["labels"]["context_correction"]["subtype"] == subtype)),
            key=lambda p: (len(pairs[p]["correct"]["target"]), p))
        return chosen[:limit]

    picked = []
    for subtype in ["组词", "拆字", "偏旁结构", "否定对比", "同名一致性"]:
        picked += [(p, "典型：纠正题沿用常用字，确认题写对")
                   for p in shortest(lambda v: v["correct"] == "alt" and v["confirm"] == "gold", subtype)]
    picked += [(p, "少数成功：纠正题按说明改对了") for p in shortest(lambda v: v["correct"] == "gold")]
    picked += [(p, "确认题写成第三种写法（同样没用上说明）") for p in shortest(lambda v: v["confirm"] == "other")]
    picked += [(p, "隐式语义：两条前半句相同，模型两条写成同一个词")
               for p in shortest(lambda v: {v["correct"], v["confirm"]} == {"gold", "alt"}, "隐式语义")]
    examples = [(note, [pairs[p]["correct"], pairs[p]["confirm"]]) for p, note in picked]

    mined = [r for r in records.values() if r["labels"]["context_correction"]["kind"] == "mined"]

    def mined_pick(flag, kind):
        chosen = [r for r in mined if r["metadata"]["corpus_transcript_correct"] is flag
                  and item_class(outcome(r["labels"]["context_correction"]["segments"], run[r["id"]]["prediction"])) == kind]
        return sorted(chosen, key=lambda r: (len(r["target"]), r["id"]))[:1]

    for flag, kind, note in ((False, "alt", "真实录音：语料原转写写错，34479 也写错"),
                             (False, "gold", "真实录音：语料原转写写错，34479 按说明写对"),
                             (True, "gold", "真实录音：原转写本来就对")):
        for record in mined_pick(flag, kind):
            examples.append((note, [record]))
    return examples


def marked(segments):
    parts = []
    for segment in segments:
        text = esc(segment["text"])
        role = segment.get("role")
        if role == "target":
            parts.append(f'<mark class="t" title="被测词（另一种写法：{esc(segment.get("alt") or "")}）">{text}</mark>')
        elif role == "evidence":
            parts.append(f'<span class="d" title="决定写法的上下文">{text}</span>')
        elif role == "protected":
            parts.append(f'<span class="p" title="保护词：必须保持">{text}</span>')
        else:
            parts.append(text)
    return "".join(parts)


VERDICT = {"gold": ("ok", "写对"), "alt": ("bad", "沿用被排除的写法"), "other": ("warn", "第三种写法")}
VERDICT_CONFIRM = {"gold": ("ok", "保持正确"), "alt": ("bad", "误改"), "other": ("warn", "第三种写法")}


def example_card(note, members, runs, index, audio_dir):
    rows = []
    for record in members:
        label = record["labels"]["context_correction"]
        cut = record["audio_slots"][-1]["ref"]["cut_id"]
        source = index[cut]
        target = audio_dir / f"{cut}.wav"
        if not target.exists():
            shutil.copyfile(source, target)
        kind = {"correct": "纠正题", "confirm": "确认题", "mined": "真实录音"}[label["kind"]]
        outputs = []
        for name, title in (("34479", "34479"), ("qwen3", "Qwen3-ASR")):
            prediction = runs[name]["predictions"][record["id"]]["prediction"]
            results = outcome(label["segments"], prediction)
            table = VERDICT_CONFIRM if label["kind"] == "confirm" else VERDICT
            css, text = table[item_class(results)]
            written = "、".join(w for _, _, w, _ in results)
            outputs.append(
                f'<div class="out"><span class="who">{title}</span>'
                f'<span class="badge {css}">{text}</span>'
                f'<span class="wrote">目标处写成 <b>{esc(written)}</b></span>'
                f'<details><summary>完整输出</summary><p>{esc(prediction)}</p></details></div>')
        alternatives = "、".join(sorted({s.get("alt") or "" for s in label["segments"] if s.get("role") == "target"}))
        rows.append(
            f'<div class="item"><div class="item-head"><span class="kind">{kind}</span>'
            f'<span class="meta">{esc(label.get("subtype") or "")} · 上下文排除的写法：{esc(alternatives)}</span></div>'
            f'<audio controls preload="none" src="audio/{esc(cut)}.wav"></audio>'
            f'<p class="ref">{marked(label["segments"])}</p>{"".join(outputs)}</div>')
    return f'<article class="example"><h4>{esc(note)}</h4>{"".join(rows)}</article>'


def bar(value, css=""):
    width = 0 if value is None else max(0.0, min(1.0, value)) * 100
    return (f'<div class="bar"><div class="track"><div class="fill {css}" style="width:{width:.1f}%"></div></div>'
            f'<span>{pct(value)}</span></div>')


def results_table(table):
    metrics = [
        ("correction_rate", "纠正成功率", "纠正题：被测字按上下文写对", "good"),
        ("unchanged_rate", "沿用率", "纠正题：仍写上下文排除的常用写法", "bad"),
        ("evidence_rate", "说明听清率", "说明部分无调拼音全对", "neutral"),
        ("conditional_correction_rate", "听清后纠正率", "说明听清的纠正题里写对的比例", "good"),
        ("keep_rate", "保持率", "确认题：被测字写对", "good"),
        ("over_correction_rate", "误改率", "确认题：改成了少见写法", "bad"),
        ("protected_keep_rate", "保护词保持率", "同音常用词保持不变", "good"),
        ("pair_accuracy", "成对正确率", "显式与同名：一对都对", "good"),
        ("implicit_pair_accuracy", "隐式语义成对正确率", "前半句相同的两条都对", "good"),
        ("cer", "CER", "字错率（守门）", "neutral"),
    ]
    head = "".join(f"<th>{esc(title)}</th>" for *_, title in COLUMNS)
    body = []
    for key, name, help_text, css in metrics:
        cells = "".join(f"<td>{bar(table[(m, mode)]['all'].get(key), css)}</td>" for m, mode, _ in COLUMNS)
        body.append(f'<tr><th scope="row">{esc(name)}<small>{esc(help_text)}</small></th>{cells}</tr>')
    return f'<div class="scroll"><table class="metrics"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def breakdown_table(table, key, values, labels):
    head = "".join(f"<th>{esc(title)}</th>" for *_, title in COLUMNS)
    body = []
    for value in values:
        cells = []
        for m, mode, _ in COLUMNS:
            block = table[(m, mode)][key][value]
            if value == "隐式语义":
                cells.append(f'<td><span class="num">{pct(block["implicit_pair_accuracy"])}</span>'
                             f'<small>成对正确 · 单条 {pct(block["implicit_accuracy"])}</small></td>')
            else:
                cells.append(f'<td><span class="num">{pct(block["correction_rate"])}</span>'
                             f'<small>保持 {pct(block["keep_rate"])} · 误改 {pct(block["over_correction_rate"])}</small></td>')
        items = table[COLUMNS[0][:2]][key][value]["items"]
        body.append(f'<tr><th scope="row">{esc(labels.get(value, value))}<small>{items} 条</small></th>{"".join(cells)}</tr>')
    return f'<div class="scroll"><table class="breakdown"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def outcome_bars(counts):
    rows = []
    for m, mode, title in COLUMNS:
        for kind, label, names in (("correct", "纠正题", ("写对", "沿用常用字", "第三种写法")),
                                   ("confirm", "确认题", ("写对", "误改成少见字", "第三种写法"))):
            c = counts[(m, mode, kind)]
            total = sum(c.values()) or 1
            parts = "".join(
                f'<div class="seg {css}" style="width:{c[k] / total * 100:.1f}%" title="{n} {c[k] / total * 100:.1f}%"></div>'
                for k, css, n in (("gold", "ok", names[0]), ("alt", "bad", names[1]), ("other", "warn", names[2])))
            legend = " · ".join(f"{n} {c[k] / total * 100:.0f}%" for k, n in zip(("gold", "alt", "other"), names))
            rows.append(f'<div class="stack-row"><div class="stack-label">{esc(title)} · {label}</div>'
                        f'<div class="stack">{parts}</div><div class="stack-legend">{esc(legend)}</div></div>')
    return "".join(rows)


CSS = """
:root{--bg:#f7f7f4;--surface:#ffffff;--text:#1d1d1b;--muted:#5f5f5a;--line:#e3e2dc;--accent:#2f5d8a;
--ok:#2f7d4f;--bad:#b4442f;--warn:#b7832a;--neutral:#6b7280;--mark:#fde68a;--evidence:#2f5d8a;--chip:#eef2f6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--surface:#1d1d1b;--text:#ecebe6;
--muted:#a3a29b;--line:#34332f;--accent:#8ab4e0;--ok:#5fbf86;--bad:#e2775f;--warn:#e0b25a;--neutral:#9ca3af;
--mark:#6b5a1e;--evidence:#8ab4e0;--chip:#262a30}}
:root[data-theme="dark"]{--bg:#141413;--surface:#1d1d1b;--text:#ecebe6;--muted:#a3a29b;--line:#34332f;--accent:#8ab4e0;
--ok:#5fbf86;--bad:#e2775f;--warn:#e0b25a;--neutral:#9ca3af;--mark:#6b5a1e;--evidence:#8ab4e0;--chip:#262a30}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 80px}
h1{font-size:30px;line-height:1.3;margin:0 0 8px}h2{font-size:22px;margin:48px 0 12px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:17px;margin:24px 0 8px}h4{font-size:15px;margin:0 0 12px;color:var(--muted);font-weight:600}
p{margin:8px 0}small{display:block;color:var(--muted);font-size:12.5px;font-weight:400;line-height:1.4}
.sub{color:var(--muted);margin:0}.lead{font-size:18px;margin:20px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px}
.card .v{font-size:30px;font-weight:700;line-height:1.2}.card .k{color:var(--muted);font-size:14px}
.quote{background:var(--surface);border-left:4px solid var(--accent);padding:12px 16px;border-radius:0 8px 8px 0}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;background:var(--surface);border:1px solid var(--line);border-radius:10px;font-size:14.5px}
th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
thead th{font-size:13px;color:var(--muted);font-weight:600}tbody th{font-weight:600;min-width:150px}
.metrics td{min-width:190px}.bar{display:flex;align-items:center;gap:8px}
.bar .track{flex:1;height:14px;background:var(--chip);border-radius:7px;overflow:hidden}
.bar .fill{height:100%;background:var(--neutral)}.bar .fill.good{background:var(--ok)}.bar .fill.bad{background:var(--bad)}
.bar span{width:48px;text-align:right;font-size:13.5px;font-weight:700;font-variant-numeric:tabular-nums}
.num{font-weight:700;font-size:16px}
.stack-row{display:grid;grid-template-columns:220px 1fr;gap:4px 16px;align-items:center;margin:10px 0}
.stack{display:flex;height:22px;border-radius:6px;overflow:hidden;background:var(--chip)}.seg.ok{background:var(--ok)}.seg.bad{background:var(--bad)}.seg.warn{background:var(--warn)}
.stack-label{font-size:14px}.stack-legend{grid-column:2;font-size:12.5px;color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--muted);margin:8px 0}
.dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
.steps{counter-reset:s;list-style:none;padding:0;margin:12px 0;display:grid;gap:10px}
.steps li{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 16px 12px 52px;position:relative}
.steps li::before{counter-increment:s;content:counter(s);position:absolute;left:16px;top:12px;width:24px;height:24px;border-radius:50%;background:var(--accent);color:var(--bg);font-size:13px;font-weight:700;text-align:center;line-height:24px}
.steps b{font-variant-numeric:tabular-nums}
.examples{display:grid;gap:16px}.example{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px}
.item{padding:12px 0;border-top:1px dashed var(--line)}.item:first-of-type{border-top:0;padding-top:0}
.item-head{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline}.kind{font-weight:700}.meta{color:var(--muted);font-size:13px}
audio{width:100%;max-width:420px;height:36px;margin:8px 0}
.ref{font-size:16px}mark.t{background:var(--mark);color:inherit;padding:0 2px;border-radius:3px;font-weight:700}
.d{text-decoration:underline;text-decoration-color:var(--evidence);text-decoration-thickness:2px;text-underline-offset:4px}
.p{border-bottom:2px dotted var(--warn)}
.out{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;font-size:14px;margin:6px 0}
.who{font-weight:600;min-width:84px}.wrote b{font-size:15px}
.badge{font-size:12px;font-weight:700;padding:2px 8px;border-radius:999px;color:#fff}.badge.ok{background:var(--ok)}.badge.bad{background:var(--bad)}.badge.warn{background:var(--warn)}
details{width:100%}summary{cursor:pointer;color:var(--muted);font-size:13px}details p{font-size:14px;color:var(--muted)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;background:var(--chip);padding:1px 5px;border-radius:4px;word-break:break-all}
ul{padding-left:20px}a{color:var(--accent)}
@media (max-width:640px){h1{font-size:24px}.stack-row{grid-template-columns:1fr}.stack-legend{grid-column:1}.card .v{font-size:26px}}
"""


def page(records, index, runs, out):
    table = blocks(runs)
    counts = outcomes_by_kind(records, runs)
    mined = mined_split(records, runs)
    numbers = funnel(records)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    examples = "".join(example_card(note, members, runs, index, audio_dir)
                       for note, members in pick_examples(records, runs))
    asr = table[("34479", "asr")]["all"]
    ts = table[("34479", "ts_asr")]["all"]
    qwen = table[("qwen3", "asr")]["all"]
    subtype_rows = breakdown_table(table, "subtype", SUBTYPES, {})
    distance_rows = breakdown_table(table, "distance", ["adjacent", "near", "far"],
                                    {"adjacent": "说明紧挨着", "near": "隔一两句", "far": "隔 10 秒以上"})
    position_rows = breakdown_table(table, "position", ["before", "after"],
                                    {"before": "说明在被测词之前", "after": "说明在被测词之后"})
    mined_rows = "".join(
        f'<tr><th scope="row">{label}<small>{mined[("34479", flag)][0]} 条</small></th>'
        f'<td><span class="num">{pct(mined[("34479", flag)][1])}</span><small>仍写原错误写法 {pct(mined[("34479", flag)][2])}</small></td>'
        f'<td><span class="num">{pct(mined[("qwen3", flag)][1])}</span><small>仍写原错误写法 {pct(mined[("qwen3", flag)][2])}</small></td></tr>'
        for flag, label in ((False, "语料原转写写错，需要按说明纠正"), (True, "语料原转写本来就对")))
    cost34479 = runs["34479"]["summary"]["cost"]
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>上下文纠偏评测集</title><style>{CSS}</style></head><body><main>
<header>
<h1>上下文纠偏评测集 v1</h1>
<p class="sub"><code>context_correction_zh@{VERSION}</code> · AmphionEval 评测集 <code>open-audio-llm/context-correction-zh@{VERSION}</code> · 2026-10-18</p>
</header>

<p class="lead"><b>结论：checkpoint-34479 基本不会按同一段话里的上下文改写法。</b>说话人已经说明了"灰是灰色的灰"，模型仍写常用字；说明本身九成以上都听清了，所以问题是<b>听清了没用上</b>，不是没听清。34479 与同底座的官方 Qwen3-ASR-1.7B 持平，问题来自底座，不是 SFT 造成的退化。</p>

<div class="cards">
<div class="card"><div class="v">{pct(asr["correction_rate"])}</div><div class="k">34479 纠正成功率（普通 ASR）</div></div>
<div class="card"><div class="v">{pct(asr["evidence_rate"], 0)}</div><div class="k">说明听清率（无调拼音）</div></div>
<div class="card"><div class="v">{pct(asr["conditional_correction_rate"])}</div><div class="k">听清后纠正率</div></div>
<div class="card"><div class="v">{pct(ts["correction_rate"])}</div><div class="k">TS 模式纠正成功率</div></div>
<div class="card"><div class="v">{pct(asr["implicit_pair_accuracy"])}</div><div class="k">隐式语义成对正确率</div></div>
</div>

<h2>要测的问题</h2>
<p>客户反馈：说话人说"我叫王光 huī，灰是灰色的灰"，34479 输出"我叫王光<b>辉</b>"，期望"我叫王光<b>灰</b>"。名字和说明在同一次请求（≤ 30 秒）里，正确写法完全能从这段话本身推出来。</p>
<p class="quote">评测集要回答：同一段话里，上下文已经确定了某个同音词怎么写，模型会不会按上下文写——而且在上下文指向常用写法时，不会乱改。</p>

<h2>评测集里有什么</h2>
<div class="scroll"><table>
<thead><tr><th>来源</th><th>任务</th><th>规模</th><th>内容</th></tr></thead><tbody>
<tr><th scope="row">合成</th><td>普通 ASR</td><td>{numbers["synthetic_pairs"]} 对（{numbers["synthetic_pairs"] * 2} 条）</td><td>LLM 出题，Qwen3-TTS 克隆测试划分说话人朗读，平均 16 秒</td></tr>
<tr><th scope="row">合成 · TS</th><td>TS-ASR</td><td>{numbers["ts_pairs"]} 对</td><td>同一批题中不超过 26.5 秒的，配同一说话人另一句真实录音作注册音频（客户实际使用的模式）</td></tr>
<tr><th scope="row">真实录音</th><td>普通 ASR</td><td>{numbers["mined"]} 条</td><td>WenetSpeech 播客里说话人自己讲写法的片段（"春晓的晓""曹操的曹"）</td></tr>
</tbody></table></div>

<h3>六种题型，每题成对</h3>
<div class="scroll"><table><thead><tr><th>题型</th><th>例子</th></tr></thead><tbody>
<tr><th scope="row">组词</th><td>我叫王光灰，灰是灰色的灰</td></tr>
<tr><th scope="row">拆字</th><td>耳东陈、立早章</td></tr>
<tr><th scope="row">偏旁结构</th><td>单人旁两个土的佳</td></tr>
<tr><th scope="row">否定对比</th><td>不是光辉的辉，是灰色的灰</td></tr>
<tr><th scope="row">同名一致性</th><td>名字出现多次，只有一处带说明，每一处都要写对</td></tr>
<tr><th scope="row">隐式语义</th><td>没有说明，后文决定前面同音词的写法：这次期终考试，考完就放假 / 这次期中考试，考完还有半学期</td></tr>
</tbody></table></div>
<p>每道<b>纠正题</b>（上下文指向少见写法）都配一道<b>确认题</b>：场景和读音相同，上下文指向常用写法。只会保留常用字的模型确认题满分、纠正题零分；一律改成少见字的模型正好相反——所以要看成对。约三成题另含一个必须保持不变的同音常用词。说明在后为主（80%），距离分紧挨着、隔一两句、隔 10 秒以上。</p>

<h2>怎么造的</h2>
<p>内容全部交给 LLM，代码只做机械校验；没通过的题隔离并保留原因，不做人工听审。</p>
<ol class="steps">
<li><b>出题</b>：qwen3.6-plus 按规格写成对的题，用标记标出被测词、说明、保护词和"不看上下文会写成什么"。共 <b>{numbers["pairs"]:,}</b> 对。</li>
<li><b>复核</b>：deepseek-v4-pro 独立判断写法是否被上下文唯一确定、不看说明时是否确实写常用字、是否同音、是否自然；代码校验带调拼音相同、两题互换写法等。通过 <b>{numbers["accepted"]:,}</b> 对。</li>
<li><b>朗读</b>：Qwen3-TTS 零样本克隆 AISHELL-3、KeSpeech 测试划分的 60 个说话人，一对题同一个声音。</li>
<li><b>音频质检</b>：SenseVoiceSmall 转写后按拼音比对：整段错误率 ≤ 10%、说明部分拼音全对、被测词带调拼音全对。{numbers["clips"]:,} 条里通过 <b>{numbers["clips_pass"]:,}</b> 条。每种题型取前 100 对（隐式 300 对）两条都通过的；拆字只有 95 对。</li>
<li><b>真实录音</b>：本地 Qwen3.6-27B 扫 {numbers["windows"]:,} 个播客窗口，命中 {numbers["flagged"]:,} 个；两个 LLM 各自写出修正后的转写，一致且只改字不改音才收录（{numbers["agreed"]} 条，去重后 {numbers["extracted"]} 条），再过同样的音频质检：<b>{numbers["mined"]}</b> 条。</li>
</ol>

<h2>怎么打分</h2>
<p>确定性打分，不用 LLM 判分。参考与输出逐字对齐，归一化与 CER 相同。</p>
<ul>
<li><b>纠正成功率</b>：纠正题里，被测字（正确写法和被排除写法不同的那几个字）写对。名字里没被说明的字写错不算（"刘木阳"写成"刘牧阳"）。</li>
<li><b>沿用率 / 误改率</b>：写成了上下文排除的那个写法。确认题写成第三种写法算保持失败，不算误改。</li>
<li><b>说明听清率</b>：说明部分无调拼音全对。说明里的同音字写错（"凯是木字旁"）仍算听清。<b>听清后纠正率</b>只看听清了的题，用来区分"没听清"和"听清了没用上"。</li>
<li><b>成对正确率</b>：一对题都对。隐式语义只按成对计分。</li>
</ul>

<h2>结果</h2>
<p>贪心解码，<code>max_new_tokens=512</code>，没有截断；34479 平均每条输出 {cost34479["mean_completion_tokens"]:.0f} 个 token。合成部分：</p>
{results_table(table)}

<h3>模型在被测词处写了什么</h3>
<div class="legend"><span><span class="dot" style="background:var(--ok)"></span>写对</span><span><span class="dot" style="background:var(--bad)"></span>写成上下文排除的写法</span><span><span class="dot" style="background:var(--warn)"></span>第三种写法</span></div>
{outcome_bars(counts)}
<p>纠正题里约四分之三仍是常用字；确认题真正误改的只有 4–7%，其余失败是第三种写法（说了"鸟叫的鸣"仍写"钟明"），同样是没用上说明。</p>

<h3>按题型</h3>
<p>格子里是纠正成功率（隐式语义为成对正确率）。</p>
{subtype_rows}
<h3>按说明的位置和距离</h3>
{position_rows}
{distance_rows}

<h3>真实录音</h3>
<div class="scroll"><table><thead><tr><th></th><th>34479</th><th>Qwen3-ASR</th></tr></thead><tbody>{mined_rows}</tbody></table></div>
<p>真实录音的需纠正部分只有 46 条，多为常见词，不和合成部分直接比较。</p>

<h2>样例</h2>
<p><mark class="t">黄底</mark>是被测词，<span class="d">下划线</span>是决定写法的上下文，<span class="p">点线</span>是保护词。样例按规则自动挑选（每种情况取文本最短的一对），音频可直接播放。</p>
<div class="examples">{examples}</div>

<h2>运行记录与复现</h2>
<ul>
<li>34479：普通 ASR 用交付包 <code>serve.sh asr</code>，TS 用 open-audio-llm main 的 SEP 服务（关闭 prefix cache），vLLM 0.18.0。运行 <a href="{WANDB}run-20261019T011627Z-c082c6aa">run-20261019T011627Z-c082c6aa</a></li>
<li>Qwen3-ASR-1.7B 官方 vLLM 0.18.0（没有 TS 模式）。运行 <a href="{WANDB}run-20261019T011627Z-31dd04ba">run-20261019T011627Z-31dd04ba</a></li>
<li>数据：<code>whai:audio-data/datasets/context_correction_zh/</code>（本机 <code>/workspace/data/datasets/context_correction_zh/</code>）；评测集定义与结果：COS <code>anfeiweng-audiollm-resource-1373259071:amphion-eval</code></li>
<li>脚本：audio-data-contract <code>scripts/context_correction/</code>；打分：AmphionEval <code>integrations/open_audio_llm/context_correction.py</code>；报告：open-audio-llm <code>docs/experiments/2026-10-18-context-correction-benchmark/</code></li>
</ul>

<h2>已知局限</h2>
<ul>
<li>真实录音只有 {numbers["mined"]} 条（计划 200 条以上），需纠正的 46 条。</li>
<li>合成题的"常用写法"由出题 LLM 判断，不一定是被测模型的先验，所以确认题失败里有一部分是第三种写法。</li>
<li>音频质检用 SenseVoiceSmall 和拼音库，个别多音字会误判；没有人工听审。合成说话人没有远场和噪声。</li>
</ul>
</main></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, index, runs = load()
    (args.out / "index.html").write_text(page(records, index, runs, args.out), encoding="utf-8")
    print(args.out / "index.html", len(list((args.out / "audio").glob("*.wav"))), "clips")


if __name__ == "__main__":
    main()
