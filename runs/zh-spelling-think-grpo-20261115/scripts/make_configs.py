"""Write this experiment's GRPO configs from the configs of the release's GRPO round (oal-grpo round 10, v10 recipe).

Only the start point, the code snapshot and this experiment's paths change:
- configs/train-data-grpo.yaml   <- train-data-grpo-v10.yaml, mined plans read from tasks/01-mining/
- configs/train-grpo.yaml        <- train-grpo-v10.yaml, start = round-2 think checkpoint-3000, code-v2, own attempts dir
- configs/mining/serve-{asr,tsasr}-{0..6}.yaml <- mining-v15interp/ (the v10 lanes, last used by round 14),
  start = round-2 think checkpoint-3000, code-v2, own served names and attempts dirs

usage: make_configs.py
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
GRPO = Path("/workspace/workspace/mingdong/repo/oal-grpo/runs/amphionasr2-grpo-20261101")
# round-2 think checkpoint-3000 in the ae model cache (ae-models.sh fetch whai:.../checkpoint-3000/ prints this path)
START = "/workspace/data/ae-state/model-cache/entries/checkpoint-3000-927844461eb1"
START_REMOTE = ("whai:open-audio-llm/runs/zh-spelling-think-r2-20261115/tasks/01-train-think/attempts/001/"
                "artifacts/training/checkpoint-3000/")
NAME = "AmphionASR-spelling-think-grpo-v10recipe-from-r2think3000-8gpu-400steps"


def replace_all(text, pairs):
    for old, new in pairs:
        if old not in text:
            raise SystemExit(f"not found: {old}")
        text = text.replace(old, new)
    return text


def data_config():
    text = (GRPO / "configs/train-data-grpo-v10.yaml").read_text()
    text = replace_all(text, [(f"{GRPO}/tasks/23-grpo-v10-from-v9interp/mining/", f"{HERE}/tasks/01-mining/")])
    header = ("# 【说字 think GRPO】由 oal-grpo runs/amphionasr2-grpo-20261101/configs/train-data-grpo-v10.yaml（交付版那一轮的数据配置）"
              "经 scripts/make_configs.py 复制，只把五份挖掘计划换成在第二轮 think checkpoint-3000 上重挖的结果（tasks/01-mining）。\n")
    (HERE / "configs/train-data-grpo.yaml").write_text(header + text)


def train_config():
    lines = (GRPO / "configs/train-grpo-v10.yaml").read_text().splitlines(keepends=True)
    body = "".join(line for line in lines if not line.startswith("#"))  # round-10 header comments
    body = replace_all(body, [
        ("code-v12", "code-v2"),
        ("train-data-grpo-v10.yaml", "train-data-grpo.yaml"),
        ("/workspace/data/ae-runs/amphionasr2-grpo-v9interp050/checkpoint", START),
        ("../tasks/23-grpo-v10-from-v9interp/attempts", "../tasks/02-grpo/attempts"),
        ("AmphionASR-grpo-v10-from-v9interp050-crowded-ts-mining-tspresent0.10-8gpu-400steps-modelonly", NAME),
    ])
    header = ("# 【说字 think GRPO】由 oal-grpo 的 train-grpo-v10.yaml（交付版那一轮）经 scripts/make_configs.py 复制。\n"
              f"# 与第十轮只差：起点为第二轮 think checkpoint-3000（{START_REMOTE}，经 ae 模型缓存），代码 code-v2（main f68e180），\n"
              "# 数据 train-data-grpo.yaml（在起点上重挖）。奖励、学习率、KL（参考模型 = 起点）、400 步、只存权重、种子都同第十轮。\n")
    (HERE / "configs/train-grpo.yaml").write_text(header + body)


def mining_configs():
    out = HERE / "configs/mining"
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("asr", "tsasr"):
        for lane in range(7):
            config = yaml.safe_load((GRPO / f"configs/mining-v15interp/serve-{kind}-{lane}.yaml").read_text())
            config["runtime"]["pythonpath"] = [{"path": "../../shared/code-v2/src"}]
            config["task"]["model"]["path"] = START
            config["task"]["arguments"]["--served-model-name"] = f"mine-r2think-{kind}"
            config["recording"]["attempts_dir"]["path"] = f"../../tasks/01-mining/serve-{kind}-{lane}/attempts"
            gpu, port = config["runtime"]["gpus"][0], config["task"]["arguments"]["--port"]
            header = (f"# 挖掘：第二轮 think checkpoint-3000 {'TS' if kind == 'tsasr' else '普通识别'}服务（第 {lane} 路，GPU {gpu}，"
                      f"端口 {port}）。由 oal-grpo configs/mining-v15interp/serve-{kind}-{lane}.yaml 经 scripts/make_configs.py 复制，"
                      "只改模型、代码、服务名与记录目录。\n")
            (out / f"serve-{kind}-{lane}.yaml").write_text(
                header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    data_config()
    train_config()
    mining_configs()
