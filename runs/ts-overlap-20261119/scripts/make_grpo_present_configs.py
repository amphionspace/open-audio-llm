"""tasks/14: GRPO configs = tasks/13's (make_grpo_hard_configs.py output) with the ts-present plan's share 0.06 -> 0.12.

tasks/12 and 13 moved the presence threshold toward rejection (misses 357 -> 384-457 on v12, new misses on samples with
no interferer, near/far mismatch and short enrollment); ts-present (target present, some samples write nothing) is the
plan that pushes toward writing. Only its share changes; the plans themselves are reused (tasks/12's five, tasks/13's
ts-hard), the replay takes the rest (0.43 -> 0.37).
usage: make_grpo_present_configs.py
"""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
T14 = "tasks/14-grpo-hard-present"
PRESENT_SHARE = 0.12


def main():
    d = yaml.safe_load((HERE / "configs/train-data-grpo-hard.yaml").read_text())
    present = [p for p in d["grpo"]["mined"] if Path(p["plan"]).name == "plan-ts-present.jsonl"]
    assert len(present) == 1 and present[0]["share"] == 0.06, present
    present[0]["share"] = PRESENT_SHARE
    header = ("# 由 scripts/make_grpo_present_configs.py 生成，不要手改。= tasks/13 的 train-data-grpo-hard.yaml，"
              f"只把 ts-present 计划的份额 0.06 改成 {PRESENT_SHARE}（回放 0.43 -> 0.37）。\n")
    (HERE / "configs/train-data-grpo-hard-present.yaml").write_text(
        header + yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=1000))
    t = yaml.safe_load((HERE / "configs/train-grpo-hard.yaml").read_text())
    t["data"]["config"] = {"path": "train-data-grpo-hard-present.yaml"}
    t["recording"]["attempts_dir"]["path"] = f"../{T14}/attempts"
    t["runtime"]["distributed"]["port"] = 29743
    t["tracking"]["name"] = "AmphionASR-ts-overlap-grpo-v26recipe-ts-hard-plan-ts-present0.12-from1117-textfinal-mtp-8gpu-400steps"
    header = ("# 由 scripts/make_grpo_present_configs.py 从 configs/train-grpo-hard.yaml（tasks/13）生成，不要手改。"
              "只改数据配置（ts-present 份额 0.12）、记录目录、端口和名字。\n")
    (HERE / "configs/train-grpo-hard-present.yaml").write_text(header + yaml.safe_dump(t, allow_unicode=True, sort_keys=False, width=1000))
    print("ok")


if __name__ == "__main__":
    main()
