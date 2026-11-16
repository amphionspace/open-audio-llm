"""Gate evaluation configs (think off) of one checkpoint, same data and parameters as the 11.06 release's own runs.

Each config copies the release run's recorded config (run.json) and changes only the model name, the server addresses
and the output directory, so the release numbers are the baseline on the same data, normalization and scoring.
Run after make_eval_configs.py: the servers (ports, served names) come from configs/eval-<name>/serve-*.yaml.

usage: make_gate_configs.py <name> [--model <model name>] [key ...]
  writes configs/eval-<name>/after-training-gate-<key>.yaml for every gate; each key listed also gets
  after-training-gate-<key>-think.yaml (think on, model <name>-think), e.g. to see what think on does to target-absent
  and non-speech records. --model (default <name>) lets several config dirs, i.e. several server pairs, write one model.
"""
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
RELEASE_RUNS = Path("/workspace/data/ae-runs/amphionasr2-grpo-v10interp050/runs")
# key -> release run (11-05/11-06, old TS prompt; Chinese scoring is the same in native scoring v3 and v4)
GATES = {
    "v12": "run-20261105T182838Z-fd77aed9",  # 中文 TS-ASR 测试集 ts-asr-benchmark@v12-20261022
    "asr-benchmark": "run-20261106T021048Z-4a3e3c39",  # asrreg 普通 ASR 回归
    "hotwords": "run-20261106T062642Z-77949f35",  # qwen3-asr-hotwords@20261001
    "police": "run-20261106T021103Z-24d7e316",  # 警务全量
    "chime4": "run-20261105T182837Z-f1fca950",  # CHiME-4 et05 real（英文）
    "customer-asr": "run-20261106T040023Z-6d445026",  # 鼎桥客户录音 customer-asr@v5 普通识别
    "customer-hotwords": "run-20261106T041330Z-10e76b87",  # 鼎桥客户录音 customer-asr@v5 热词
    "customer-ts": "run-20261106T040238Z-022113fb",  # 鼎桥客户录音 customer-ts@v4 机主识别
    # the rest of the 11.06 package's validation (validation/results/results.md), for the delivery evaluation
    "babble4": "run-20261105T185355Z-b5256f15",  # 背景人声与空白 asr-babble-bench@v4-20261104
    "ts-en": "run-20261105T183124Z-a3273891",  # 英文 TS ts-asr-benchmark-en@v1-20261031
    "ts-hotwords": "run-20261105T184337Z-1127dc9d",  # TS + 热词 ts-asr-hotwords@v1-20261102
    "badcase": "run-20261105T185507Z-64038efc",  # 鼎桥 e警保 回退集 tdtech/badcase@v3-20261104
    "first-other": "run-20261105T185559Z-9b91bc4f",  # 跟错人 ts-asr-first-other@v3-20261105
    "hotword-bench": "run-20261106T021048Z-47cca8d9",  # hotword-bench@v3-20261020
    "police-single": "run-20261106T021520Z-b93e11b0",  # 警务单说术语
    "scene-asr": "run-20261106T021635Z-0e77fcaf",  # 鼎桥场景普通识别
    "scene-ts": "run-20261106T021900Z-53c44b85",  # 鼎桥场景 TS tdtech/scene-ts@v1-20261028
}


def main():
    name, think_keys = sys.argv[1], sys.argv[2:]
    model_name = name
    if think_keys[:1] == ["--model"]:
        model_name, think_keys = think_keys[1], think_keys[2:]
    unknown = set(think_keys) - set(GATES)
    if unknown:
        sys.exit(f"unknown gate keys: {sorted(unknown)}")
    out = HERE / "configs" / f"eval-{name}"
    serve = {route: yaml.safe_load((out / f"serve-{route}.yaml").read_text())["task"]["arguments"]
             for route in ("asr", "tsasr")}
    for key, run in GATES.items():
        config = json.loads((RELEASE_RUNS / run / "run.json").read_text())["config"]
        if config.get("eval_set"):
            config.pop("data_config", None)  # resolved copy of the eval set; the name resolves it again
        for route, url_key, model_key in (("asr", "server_url", "served_model"),
                                          ("tsasr", "tsasr_server_url", "tsasr_served_model")):
            if config.get(url_key):
                config[url_key] = f"http://127.0.0.1:{serve[route]['--port']}"
                config[model_key] = serve[route]["--served-model-name"]
        for think in (False, True) if key in think_keys else (False,):
            model = f"{model_name}-think" if think else model_name
            variant = dict(config, model_name=model, output_root=f"/workspace/data/ae-runs/spelling-think-grpo-20261115-{model}",
                           wandb={"entity": "1016097967-amphion", "project": "open-audio-llm"}, publish={"enabled": False})
            if think:
                variant["parameters"] = {**config["parameters"], "think": True}
            mode = "think 开" if think else "think 关"
            header = f"# 守门（{mode}）：交付版 {run} 的配置，只换模型、服务与输出目录（make_gate_configs.py）。\n"
            (out / f"after-training-gate-{key}{'-think' if think else ''}.yaml").write_text(
                header + yaml.safe_dump(variant, allow_unicode=True, sort_keys=False))
    print(out)


if __name__ == "__main__":
    main()
