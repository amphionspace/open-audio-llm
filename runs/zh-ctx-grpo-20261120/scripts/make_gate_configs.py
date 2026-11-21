"""Gate evaluation configs (think off) of one checkpoint, same data and parameters as the 11.06 release's own runs.

Copied from runs/zh-spelling-think-grpo-20261115/scripts/make_gate_configs.py (oal-think-exp, "08"). Each config copies
the release run's recorded config (run.json) and changes only the model name, the server addresses and the output
directory. Run with ae 0.14 (new TS prompt), as the start arm's own TS runs (zh-ctx-clue-target-20261117 questions 7
and 8), so candidate and start are compared on the same client; against the release, the TS items' difference also
holds the prompt change (as in 08).
Gates: 08's eight (README "判定": the rows of 08's gate table), plus follow-wrong (first-other, question 8) and scene TS.
The servers (ports, served names) come from configs/eval-<name>/serve-*.yaml (make_eval_configs.py).
usage: make_gate_configs.py <name>   (writes configs/eval-<name>/after-training-gate-<key>.yaml for every gate)
"""
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
RELEASE_RUNS = Path("/workspace/data/ae-runs/amphionasr2-grpo-v10interp050/runs")
GATES = {
    "v12": "run-20261105T182838Z-fd77aed9",  # 中文 TS-ASR 测试集 ts-asr-benchmark@v12-20261022
    "asr-benchmark": "run-20261106T021048Z-4a3e3c39",  # asrreg 普通 ASR 回归
    "hotwords": "run-20261106T062642Z-77949f35",  # qwen3-asr-hotwords@20261001
    "police": "run-20261106T021103Z-24d7e316",  # 警务全量
    "chime4": "run-20261105T182837Z-f1fca950",  # CHiME-4 et05 real（英文）
    "customer-asr": "run-20261106T040023Z-6d445026",  # 鼎桥客户录音 customer-asr@v5 普通识别
    "customer-hotwords": "run-20261106T041330Z-10e76b87",  # 鼎桥客户录音 customer-asr@v5 热词
    "customer-ts": "run-20261106T040238Z-022113fb",  # 鼎桥客户录音 customer-ts@v4 机主识别
    "first-other": "run-20261105T185559Z-9b91bc4f",  # 跟错人 ts-asr-first-other@v3-20261105
    "scene-ts": "run-20261106T021900Z-53c44b85",  # 鼎桥场景 TS tdtech/scene-ts@v1-20261028
}


def main():
    name = sys.argv[1]
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
        assert not config["parameters"].get("think"), key
        config.update(model_name=f"ctx-grpo-{name}", output_root=f"/workspace/data/ae-runs/ctx-grpo-20261120-{name}",
                      wandb={"entity": "1016097967-amphion", "project": "open-audio-llm"}, publish={"enabled": False})
        header = f"# 守门（think 关）：交付版 {run} 的配置，只换模型、服务与输出目录（scripts/make_gate_configs.py）。\n"
        (out / f"after-training-gate-{key}.yaml").write_text(header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    print(out)


if __name__ == "__main__":
    main()
