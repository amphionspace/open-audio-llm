"""Write the two training arms from round 2's think run (runs/zh-spelling-think-r2-20261115).

Both arms copy round 2's data and training configs and add joint MTP (AGENTS.md "MTP"; the 11.06 start has no mtp/,
so the module starts from the 11.17 delivery package's). They differ only in the think files:
- template:     round 2's 12 think files as they are (``<pinyin> 这个音，说话人说是「…」，写作「…」。``);
- clue-target:  the same rows rewritten by make_think_v3.py (``线索：说话人说「…」。目标：写「…」，不写「…」。``).

usage: make_configs.py   (writes configs/train-data-{arm}.yaml, configs/train-{arm}.yaml)
"""
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
# Round 2 lives on the oal-think-exp branch (docs/spelling-think-grpo), not on main.
R2 = Path("/workspace/workspace/mingdong/repo/oal-think-exp/runs/zh-spelling-think-r2-20261115")
OLD_THINK = ("/workspace/data/datasets/context_correction_clean_zh/versions/think-v2-20261113/",
             "/workspace/data/datasets/context_correction_synth_zh/versions/train-v1-20261114/think.jsonl.gz")
NEW_THINK = "/workspace/data/datasets/context_correction_clean_zh/versions/think-v3-20261117/"
MTP_INIT = "/workspace/data/packages/amphionasr-1.7b-v3-v18aes-mtp-candidate/model/mtp/model.safetensors"
MTP = {"--joint_mtp_weight": 0.1, "--joint_mtp_steps": 3, "--joint_mtp_step_decay": 0.8, "--joint_mtp_lr": 0.0001,
       "--joint_mtp_init": MTP_INIT}
ARMS = {"template": {"port": 29826, "name": "template-think"},
        "clue-target": {"port": 29836, "name": "clue-target-think"}}
CODE = EXP / "shared/code-v1"


def think_files(files, arm):
    if arm == "template":
        return files
    out = []
    for f in files:
        if f.startswith(OLD_THINK[0]):
            out.append(NEW_THINK + f[len(OLD_THINK[0]):])
        elif f == OLD_THINK[1]:
            out.append(NEW_THINK + "context_correction_synth_zh/think.jsonl.gz")
        else:
            raise ValueError(f)
    assert all(Path(f).exists() for f in out), out
    return out


def main():
    data = yaml.safe_load((R2 / "configs/train-data-think.yaml").read_text())
    train = yaml.safe_load((R2 / "configs/train-think.yaml").read_text())
    old_code = str(R2 / "shared/code-v2")
    for arm, spec in ARMS.items():
        d = {**data, "think": {**data["think"], "files": think_files(data["think"]["files"], arm)}}
        data_path = EXP / f"configs/train-data-{arm}.yaml"
        data_path.write_text(
            f"# 由 scripts/make_configs.py 从 runs/zh-spelling-think-r2-20261115/configs/train-data-think.yaml 生成，不要手改。\n"
            f"# {arm}：" + ("think 文件与第二轮相同。\n" if arm == "template" else
                           "think 文件换成 context_correction_clean_zh/versions/think-v3-20261117（同一批行，改写成线索 + 目标），其余与第二轮相同。\n")
            + yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=1000))
        t = yaml.safe_load(yaml.safe_dump(train))
        rt = t["runtime"]
        rt["cwd"] = str(EXP.parents[1])
        rt["pythonpath"] = [p.replace(old_code, str(CODE)) for p in rt["pythonpath"]]
        rt["distributed"]["port"] = spec["port"]
        t["data"]["config"] = str(data_path)
        args = t["task"]["arguments"]
        args["--external_plugins"] = [p.replace(old_code, str(CODE)) for p in args["--external_plugins"]]
        args["--deepspeed"] = str(EXP / "configs/zero1-bf16-no-overflow-check.json")  # copied from round 2
        args.update(MTP)
        t["preflight"][0]["arguments"]["--data_config"] = str(data_path)
        t["recording"]["attempts_dir"]["path"] = f"../tasks/01-train-{arm}/attempts"
        t["tracking"]["pythonpath"] = [str(CODE / "src")]
        t["tracking"]["name"] = f"AmphionASR-ctx-{spec['name']}-r2recipe-mtp-from-v10interp050-3k-1node-codev1"
        t["storage"]["local_root"] = str(EXP.parents[1])
        text = yaml.safe_dump(t, allow_unicode=True, sort_keys=False, width=1000)
        assert old_code not in text and "zh-spelling-think-r2-20261115/configs/train-data" not in text
        (EXP / f"configs/train-{arm}.yaml").write_text(
            f"# 由 scripts/make_configs.py 从 runs/zh-spelling-think-r2-20261115/configs/train-think.yaml 生成，不要手改。\n"
            f"# {arm} 组：第二轮的训练参数，加联合 MTP（权重 0.1、3 步、衰减 0.8、lr 1e-4，从 11.17 交付包的 mtp/ 初始化），"
            f"代码 shared/code-v1（main 0a1b117），数据 configs/train-data-{arm}.yaml。\n" + text)
        print(arm, data_path)


if __name__ == "__main__":
    main()
