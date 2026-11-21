"""The README "判定" (written before training) for grpo-avg3 and grpo-interp050, each against the start J (start-j),
think off, from compare.sh's outputs and the runs' per-item scores.

1. main: Emilia "需要纠正 · explicit" items of the context-correction eval v2 (the ctx_v2_mined_groups.py group,
   runs/zh-ctx-clue-target-20261117): new right > new wrong, exact two-sided sign test p < 0.05;
2. context correction kept: overall keep rate and Emilia accuracy, 95% CI low >= -1 point;
3. TS not worse: scene TS CER and false alarms; v12 and first-other CER, false alarms and misses (CI low <= 0);
4. gates not worse: asrreg zh CER and term hits, Common Voice zh CER, asrreg en WER, GigaSpeech non-speech false
   alarms, Common Voice en WER, police CER and term hits (CI low <= 0; for term hits CI high >= 0).
Reported only: TS-mode items of the context-correction eval left empty, scene-TS target-present items left empty,
think on. Both candidates pass -> avg3.
usage: python3 gate.py
"""
import gzip
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_runs import latest  # noqa: E402

C = Path(__file__).resolve().parents[1] / "tasks/03-eval/compare"
MINED = "/workspace/data/datasets/context_correction_zh/versions/v2-20261116/records/mined_real.jsonl.gz"
POINT = 0.01
START = "start-j"


def fmt(x, scale=100):
    return (f"{x['a'] * scale:.2f} → {x['b'] * scale:.2f}，{x['diff'] * scale:+.2f} "
            f"[{x['ci'][0] * scale:+.2f}, {x['ci'][1] * scale:+.2f}]{'*' if x['significant'] else ''}")


def load(cand, suite):
    return json.loads((C / f"cmp-{cand}-{suite}.json").read_text())


def sign_test(wins, losses):
    n = wins + losses
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2 if n else 1.0
    return min(p, 1.0)


def emilia_explicit(cand):
    meta = {}
    for line in gzip.open(MINED, "rt", encoding="utf-8"):
        r = json.loads(line)
        meta[r["id"]] = (not r["metadata"]["corpus_transcript_correct"], r["labels"]["context_correction"]["category"])
    def scores(model):
        out = {}
        for line in open(latest(model, "ctx-off") / "context-correction-scores.jsonl", encoding="utf-8"):
            s = json.loads(line)
            if s["kind"] == "mined" and "emilia_zh" in s["source"] and meta.get(s["id"]) == (True, "explicit"):
                out[s["id"]] = s["target_ok"]
        return out
    a, b = scores(START), scores(cand)
    assert set(a) == set(b), "start and candidate scored different items"
    wins = sum(b[i] and not a[i] for i in a)
    losses = sum(a[i] and not b[i] for i in a)
    return len(a), sum(a.values()), sum(b.values()), wins, losses, sign_test(wins, losses)


def empty_ts(model):
    rows = [json.loads(l) for l in open(latest(model, "ctx-off") / "predictions.jsonl", encoding="utf-8")]
    return sum(r["task"] == "ts_asr" and not r["prediction"].strip() for r in rows)


def empty_scene(model):
    rows = [json.loads(l) for l in open(latest(model, "scene-ts") / "predictions.jsonl", encoding="utf-8")]
    return sum(bool(r["reference"].strip()) and not r["prediction"].strip() for r in rows)


def check(rows, label, x, ok):
    rows.append(ok)
    print(f"  {'通过' if ok else '不通过'}  {label}：{fmt(x)}")


def candidate(cand):
    print(f"== {cand} 对起点 J（think 关）")
    rows = []
    n, a, b, wins, losses, p = emilia_explicit(cand)
    main = wins > losses and p < 0.05
    print(f"  {'通过' if main else '不通过'}  1. Emilia 需要纠正·显式说明（{n} 条）：{100 * a / n:.1f} → {100 * b / n:.1f}，"
          f"新对 {wins}、新错 {losses}，符号检验 p={p:.3g}")
    cc = load(cand, "ctx-off")["context_correction"]
    check(rows, "2. 全部保持率", cc["overall"]["keep_rate"], cc["overall"]["keep_rate"]["ci"][0] >= -POINT)
    em = cc["by"]["source"]["mined:emilia_zh"]["mined_accuracy"]
    check(rows, "2. Emilia 准确率", em, em["ci"][0] >= -POINT)
    scene = load(cand, "scene-ts")["overall"]["all/CER"]
    for m, name in (("error_rate", "CER"), ("false_alarm_rate", "误出字率")):
        check(rows, f"3. 场景 TS {name}", scene[m], scene[m]["ci"][0] <= 0)
    for suite, label in (("v12", "中文 TS-ASR 测试集 v12"), ("first-other", "跟错人 first-other v3")):
        o = load(cand, suite)["overall"]["all/CER"]
        for m, name in (("error_rate", "CER"), ("false_alarm_rate", "误触发率"), ("miss_rate", "漏识别率")):
            check(rows, f"3. {label} {name}", o[m], o[m]["ci"][0] <= 0)
    asr = load(cand, "asr-benchmark")
    src = asr["sources"]
    gates = [("asrreg 中文 CER", asr["overall"]["all/CER"]["error_rate"], False),
             ("asrreg 术语命中率", asr["overall"]["all/CER"]["term_hit_rate"], True),
             ("Common Voice 中文 CER", src["common_voice_zh@icefall-20260908:test/asr/zh-CN"]["error_rate"], False),
             ("asrreg 英文 WER", asr["overall"]["all/WER"]["error_rate"], False),
             ("GigaSpeech 无语音误触发率", asr["overall"]["all/WER"]["false_alarm_rate"], False),
             ("Common Voice 英文 WER", src["common_voice_en@icefall-20260908:test/asr/en"]["error_rate"], False)]
    police = load(cand, "police")["overall"]["all/CER"]
    gates += [("警务全量 CER", police["error_rate"], False), ("警务术语命中率", police["term_hit_rate"], True)]
    for label, x, higher_better in gates:
        check(rows, f"4. {label}", x, x["ci"][1] >= 0 if higher_better else x["ci"][0] <= 0)
    on = load(cand, "ctx-on")["context_correction"]
    print(f"  参考  think 开 · 全部保持率：{fmt(on['overall']['keep_rate'])}；Emilia 准确率："
          f"{fmt(on['by']['source']['mined:emilia_zh']['mined_accuracy'])}")
    print(f"  参考  上下文纠偏 v2 TS 题输出空：J {empty_ts(START)}，{cand} {empty_ts(cand)}；"
          f"场景 TS 目标在场输出空：J {empty_scene(START)}，{cand} {empty_scene(cand)}")
    rest = all(rows)
    return main, rest


def main():
    results = {c: candidate(c) for c in ("grpo-avg3", "grpo-interp050")}
    passed = [c for c, (m, r) in results.items() if m and r]
    if passed:
        pick = "grpo-avg3" if "grpo-avg3" in passed else passed[0]
        print(f"结论：{pick} 四项都满足，进入交付评估（对交付版的差由用户定夺）")
    elif any(r and not m for m, r in results.values()):
        print("结论：主判定不满足，其余满足。GRPO 没把显式说明题改对，按数字记录、按失败类型分析，不做参数扫描")
    else:
        print("结论：第 2–4 项有不满足，按数字记录、按来源定位，不进交付评估")


if __name__ == "__main__":
    main()
