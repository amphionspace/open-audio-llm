"""tasks/13: GRPO configs = tasks/12's (make_grpo_configs.py output) with the ts-transcribe plan replaced by ts-hard.

Only the sixth plan changes: tasks/12's five behaviour plans are reused as mined there (same start, same lanes), and
ts-hard (grpo_hard_plan.py, from tasks/12's TS lanes plus this task's extra TS lanes) takes ts-transcribe's share.
Everything else (training text, code, parameters, rewards, MTP) stays as in tasks/12, so tasks/12 is the control.
Outputs: configs/train-data-grpo-hard.yaml, configs/train-grpo-hard.yaml, configs/grpo-mining-hard/serve-tsasr-{0..6}.yaml
(tasks/12's TS servers with this task's attempts dirs).
usage: make_grpo_hard_configs.py
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
T12, T13 = "tasks/12-grpo-transcribe", "tasks/13-grpo-hard"


def main():
    d = yaml.safe_load((HERE / "configs/train-data-grpo-transcribe.yaml").read_text())
    mined = d["grpo"]["mined"]
    assert Path(mined[-1]["plan"]).name == "plan-ts-transcribe.jsonl", mined[-1]
    mined[-1] = {"plan": str(HERE / T13 / "mining/plan-ts-hard.jsonl"), "share": mined[-1]["share"]}
    assert all(T12 in p["plan"] for p in mined[:-1])
    header = ("# 由 scripts/make_grpo_hard_configs.py 生成，不要手改。= tasks/12 的 train-data-grpo-transcribe.yaml，"
              "只把第六份计划 ts-transcribe 换成 ts-hard（scripts/grpo_hard_plan.py，同份额）；另外五份计划沿用 tasks/12 的挖题结果。\n")
    (HERE / "configs/train-data-grpo-hard.yaml").write_text(
        header + yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=1000))

    t = yaml.safe_load((HERE / "configs/train-grpo-transcribe.yaml").read_text())
    t["data"]["config"] = {"path": "train-data-grpo-hard.yaml"}
    t["recording"]["attempts_dir"]["path"] = f"../{T13}/attempts"
    t["runtime"]["distributed"]["port"] = 29742
    t["tracking"]["name"] = "AmphionASR-ts-overlap-grpo-v26recipe-plus-ts-hard-plan-spread0.3-from1117-textfinal-mtp-8gpu-400steps"
    header = ("# 由 scripts/make_grpo_hard_configs.py 从 configs/train-grpo-transcribe.yaml（tasks/12）生成，不要手改。"
              "只改数据配置（ts-hard 计划）、记录目录、端口和名字。\n")
    (HERE / "configs/train-grpo-hard.yaml").write_text(header + yaml.safe_dump(t, allow_unicode=True, sort_keys=False, width=1000))

    out = HERE / "configs/grpo-mining-hard"
    out.mkdir(parents=True, exist_ok=True)
    for lane in range(7):
        c = yaml.safe_load((HERE / f"configs/grpo-mining/serve-tsasr-{lane}.yaml").read_text())
        c["recording"]["attempts_dir"]["path"] = f"../../{T13}/mining/serve-tsasr-{lane}/attempts"
        header = f"# 挖题（tasks/13 加挖的 TS 线）：同 configs/grpo-mining/serve-tsasr-{lane}.yaml，只改记录目录。由 scripts/make_grpo_hard_configs.py 生成。\n"
        (out / f"serve-tsasr-{lane}.yaml").write_text(header + yaml.safe_dump(c, allow_unicode=True, sort_keys=False))
    (out / "sources-ts.txt").write_text((HERE / "configs/grpo-mining/sources-ts.txt").read_text())
    print("ok")


if __name__ == "__main__":
    main()
