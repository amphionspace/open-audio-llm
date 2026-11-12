"""Collect Spanish-parity results from AmphionEval runs into results.json and print markdown tables."""
import glob, json, os, sys
AE = "/workspace/data/ae-runs"
MODELS = {  # label: (ae model dir, description)
    "qwen": ("qwen3-asr-1.7b", "Qwen3-ASR-1.7B 原版"),
    "sumfull": ("whitebox-merge-sum-full", "两轮相加（11-06 交付候选，未见西语）"),
    "sft076": ("ts-robust-076-avg3", "SFT 076 avg3（无西语）"),
    "v69": ("ts-robust-081-avg3", "v69：+ 西语 ASR / 场景池"),
    "v70": ("ts-robust-083-avg3", "v70：+ 西语在线混音、干扰人池、同语言干扰词"),
    "v71": ("ts-robust-084-avg3", "v71：v70 + 目标/干扰人按口音标注分开"),
    "v72": ("ts-robust-085-avg3", "v72：v71 + 西语单他人缺席 10%→40%"),
    "v73": ("ts-robust-086-avg3", "v73：v72 + 中英同样改"),
    "v74": ("ts-robust-087-avg3", "v74：v73 数据，6000 步"),
    "v76": ("ts-robust-089-avg3", "v76：v74 + 西语近音热词数据 + hotword_miss_prob 0.15"),
    "v77": ("ts-robust-090-avg3", "v77：v76 + hotword_confusable_miss_prob 0.5"),
    "v78": ("ts-robust-091-avg3", "v78：v77 + 西语近音热词数据权重 1.0→3.0"),
    "es081": ("whitebox-merge-sumfull-es081", "两轮相加 + (v69 − SFT076)"),
    "es086": ("whitebox-merge-sumfull-es086", "两轮相加 + (v73 − SFT076)"),
    "es091": ("whitebox-merge-sumfull-es091", "两轮相加 + (v78 − SFT076)"),
}
ES_ASR_CFG = "configs/es/es-asr-test-data.yaml"

def runs(model):
    out = {}
    for d in sorted(glob.glob(f"{AE}/{model}/runs/*")):
        try:
            c = json.load(open(d + "/run.json"))["config"]; s = json.load(open(d + "/summary.json"))
        except Exception:
            continue
        key = c.get("eval_set") or ("es-asr" if str(c.get("data_config", "")).endswith(ES_ASR_CFG) else None)
        if key: out[key.split("/")[-1]] = (d, s)
    return out

def agg(m, filt=lambda k: True):
    e = r = fa = neg = miss = pos = 0
    for k, v in m.items():
        if not filt(k): continue
        if v.get("reference_units"): e += v["errors"]; r += v["reference_units"]
        if v.get("negative_utterances"): neg += v["negative_utterances"]; fa += round(v["negative_utterances"] * (v["false_alarm_rate"] or 0))
        if v.get("miss_rate") is not None:
            n = v["utterances"] - (v.get("negative_utterances") or 0); pos += n; miss += round(n * v["miss_rate"])
    return {"error_rate": round(100 * e / r, 2) if r else None, "false_alarms": fa, "negatives": neg, "misses": miss, "positives": pos}

res = {"date": "2026-11-12", "backend": "vLLM 部署推理栈（LoRA 合并后加载）",
       "remote": "whai:open-audio-llm/experiments/2026-11-12-spanish-parity/ae-runs/", "models": {}}
for lab, (mdl, desc) in MODELS.items():
    rr = runs(mdl); ent = {"ae_model": mdl, "description": desc, "runs": {k: os.path.basename(d) for k, (d, s) in rr.items()}, "metrics": {}}
    for es, (d, s) in rr.items():
        if es == "es-asr":
            ent["metrics"][es] = {k.split("@")[0]: round(100 * v["errors"] / v["reference_units"], 2) for k, v in s["metrics"].items()}
        elif es.startswith("hotword-bench"):
            h = s["hotword_bench"]["metrics"]; conds = {}
            for k, v in h.items():
                conds[k] = {f: round(v[f] * (1 if f == "false_alarms_per_100" else 100), 2) for f in ("error_rate", "b_error_rate", "f1", "false_alarms_per_100") if v.get(f) is not None}
            ent["metrics"][es] = conds
        elif "metrics" in s:
            ent["metrics"][es] = agg(s["metrics"])
            if es.startswith("ts-asr-crosslingual-es"):
                sel = lambda suf: (lambda k: k.split(":")[1].split("/")[0].endswith(suf))
                ent["metrics"][es + "#same"] = agg(s["metrics"], sel("same")); ent["metrics"][es + "#cross"] = agg(s["metrics"], sel("cross"))
    res["models"][lab] = ent
json.dump(res, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
