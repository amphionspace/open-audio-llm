"""Pair a confidence-gate AmphionEval run with its baseline run (same set, same serving stack).

Per source: error rate / false-output rate before and after, both rescored from predictions.jsonl
with the installed AmphionEval scorer (a run's own summary.json may predate scoring v2). Per run: which samples the gate
emptied (raw output became ``language None<asr_text>`` where the baseline had text), how wrong
those samples were in the baseline, how many other predictions moved (vLLM batching noise), and
how many kept outputs still contain a repetition loop (a wider loop rule would empty them too).

    python compare_gate.py BASELINE_RUN_DIR GATE_RUN_DIR [...more pairs]
"""
import json, re, sys
from collections import defaultdict

from amphion_eval.integrations.open_audio_llm.scoring import summarize, tokens, variety
from amphion_eval.legacy.open_audio_llm.qwen3_asr import normalize as presence_text
from rapidfuzz.distance import Levenshtein

NO_SPEECH = "language None<asr_text>"
LOOP = re.compile(r"(.{1,4}?)\1{7,}")


def load(run):
    return {r["id"]: r for r in map(json.loads, open(f"{run}/predictions.jsonl"))}


def errors(row, prediction):
    kind = variety(row["language"])
    ref = tokens(row["reference"], kind)
    return Levenshtein.distance(ref, tokens(prediction, kind)), len(ref)


def rescored(rows):
    return summarize([{k: r.get(k) for k in ("source", "language", "task", "reference", "prediction", "terms")} for r in rows.values()])


def main():
    pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
    for base_dir, gate_dir in pairs:
        cfg = json.load(open(f"{gate_dir}/run.json"))["config"]
        name = cfg.get("eval_set") or cfg.get("data_config", "").rsplit("/", 1)[-1]
        base, gate = load(base_dir), load(gate_dir)
        common = base.keys() & gate.keys()
        rejected, moved, loops = [], [], []
        for i in common:
            b, g = base[i], gate[i]
            if (g.get("raw_text") or "").startswith(NO_SPEECH) and presence_text(b["prediction"]):
                rejected.append(i)
            elif g["prediction"] != b["prediction"]:
                moved.append(i)
            if presence_text(g["prediction"]) and LOOP.search(re.sub(r"\W", "", g["prediction"])):
                loops.append(i)
        speech = [i for i in rejected if presence_text(base[i]["reference"])]
        e = u = 0
        for i in speech:
            x, y = errors(base[i], base[i]["prediction"])
            e, u = e + x, u + y
        le = lu = 0
        for i in loops:
            if presence_text(gate[i]["reference"]):
                x, y = errors(gate[i], gate[i]["prediction"])
                le, lu = le + x, lu + y
        print(f"\n## {name}  baseline {base_dir.rsplit('/', 1)[-1]}  gate {gate_dir.rsplit('/', 1)[-1]}  n={len(common)}")
        print(f"   拒识 {len(rejected)} 条（{100 * len(rejected) / len(common):.2f}%）：其中有参考文本的 {len(speech)} 条，"
              f"它们在 baseline 的 CER/WER {100 * e / u if u else 0:.1f}%（参考 {u} 单位）；无参考（应为空）的 {len(rejected) - len(speech)} 条")
        print(f"   未拒识但输出变了 {len(moved)} 条（{100 * len(moved) / len(common):.2f}%，批处理抖动）")
        print(f"   保留的输出里仍含复读 {len(loops)} 条" + (f"，有参考的 CER {100 * le / lu:.1f}%" if lu else ""))
        mb, mg = rescored({i: base[i] for i in common}), rescored({i: gate[i] for i in common})
        # the gate only rewrites rejected answers: baseline predictions with just those emptied
        # isolate its effect from vLLM batching noise in the rerun
        only = rescored({i: {**base[i], "prediction": ""} if i in set(rejected) else base[i] for i in common})
        by_source = defaultdict(int)
        for i in rejected:
            by_source[gate[i]["source"]] += 1
        rows = []
        for src in sorted(mb.keys() & mg.keys()):
            a, c = mb[src], mg[src]
            if a.get("error_rate") is not None:
                rows.append((src, 100 * a["error_rate"], 100 * c["error_rate"], a["reference_units"], a["errors"], c["errors"], by_source[src], a["utterances"]))
            elif a.get("false_alarm_rate") is not None:
                print(f"     {src[:64]:64} 误输出率 {100 * a['false_alarm_rate']:6.2f}% → {100 * c['false_alarm_rate']:6.2f}%  (n={a['negative_utterances']})")
        if rows:
            wb = 100 * sum(r[4] for r in rows) / sum(r[3] for r in rows)
            wg = 100 * sum(r[5] for r in rows) / sum(r[3] for r in rows)
            mb_ = sum(r[1] for r in rows) / len(rows)
            mg_ = sum(r[2] for r in rows) / len(rows)
            worse = sum(r[2] > r[1] + 1e-9 for r in rows)
            better = sum(r[2] < r[1] - 1e-9 for r in rows)
            print(f"   {len(rows)} 个切片：加权 {wb:.3f} → {wg:.3f} ({wg - wb:+.3f})，宏平均 {mb_:.3f} → {mg_:.3f} ({mg_ - mb_:+.3f})，变差 {worse} / 变好 {better}（含重跑抖动）")
            src_rows = [s_ for s_ in only if only[s_].get("error_rate") is not None and s_ in mb]
            wo = 100 * sum(only[s_]["errors"] for s_ in src_rows) / sum(only[s_]["reference_units"] for s_ in src_rows)
            mo = 100 * sum(only[s_]["error_rate"] for s_ in src_rows) / len(src_rows)
            print(f"   只算插件（baseline 预测里把被拒样本置空）：加权 {wb:.3f} → {wo:.3f} ({wo - wb:+.3f})，宏平均 {mb_:.3f} → {mo:.3f} ({mo - mb_:+.3f})")
            fa_only = [(s_, mb[s_]["false_alarm_rate"], only[s_]["false_alarm_rate"]) for s_ in only if only[s_].get("false_alarm_rate") is not None and mb[s_].get("error_rate") is None]
            if fa_only:
                print("   只算插件的误输出率：" + "；".join(f"{s_.split(':')[1].split('/')[0]} {100 * a:.2f}%→{100 * c:.2f}%" for s_, a, c in fa_only))
            for src, a, c, units, *_, nrej, n in sorted(rows, key=lambda r: r[1] - r[2]):
                if abs(c - a) >= 0.005 or nrej:
                    print(f"     {src[:64]:64} {a:7.2f} → {c:7.2f} ({c - a:+.2f})  拒识 {nrej}/{n}")


if __name__ == "__main__":
    main()
