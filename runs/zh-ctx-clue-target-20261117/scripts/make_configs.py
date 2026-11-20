"""Write the two training arms from round 2's think run (runs/zh-spelling-think-r2-20261115).

Both arms copy round 2's data and training configs and add joint MTP (AGENTS.md "MTP"; the 11.06 start has no mtp/,
so the module starts from the 11.17 delivery package's). They differ only in the think files:
- template:     round 2's 12 think files as they are (``<pinyin> 这个音，说话人说是「…」，写作「…」。``);
- clue-target:  the same rows rewritten by make_think_v3.py (``线索：说话人说「…」。目标：写「…」，不写「…」。``).

With ``--implicit``, once context_correction_synth_zh@implicit-v1-20261117 is built (implicit-semantics items, which
round 2's data has none of), it also writes two arms on top of clue-target, each adding that data the way round 2
added its synthetic data (each record seen about EXPLAINED_VIEWS times in the run):
- clue-target-implicit:            its think file (clue + target, the clue being the semantic context);
- clue-target-implicit-noexplain:  the same rows with every think-on line replaced by round 2's fixed sentence
  for records without an explanation (scripts/make_implicit_noexplain_think.py), so the two differ only in
  whether the think reasons about the implicit item;
and adds their tasks to experiment.yaml (the experiment refuses a task whose config does not exist yet).
A third implicit arm (question 3, 2026-11-20) puts the same data on top of template instead:
- template-implicit-noexplain:     round 2's think files plus the implicit data with the fixed sentence, i.e. the
  combination proposed for the context-correction SFT recipe (round-2 think, implicit items without reasoning).
With ``--ablation`` (question 4, 2026-11-20), two arms that split round 2's scene-TS regression between its two
additions over round 1 (round 1's data is round 2's minus the two synthetic sources; the code differs only in #158/#159,
which set think.empty_prob / uncovered_prob; #156/#157 touch the replay KL, which neither round uses):
- template-no-empty-think:  template with think.empty_prob and uncovered_prob 0, i.e. round 1's think allocation;
- template-no-synth:        template without the synthetic spelling data (train, train_ts) and its think file, i.e.
  round 1's data with #158/#159's think allocation.

usage: make_configs.py [--implicit] [--ablation] [arm ...]   (writes configs/train-data-{arm}.yaml, configs/train-{arm}.yaml for
       the named arms, all of them when none is named; configs written on the node are root's and stay as they are)
"""
import gzip
import sys
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
IMPLICIT_ID, IMPLICIT_VERSION = "context_correction_synth_zh", "implicit-v1-20261117"
IMPLICIT = Path(f"/workspace/data/datasets/{IMPLICIT_ID}/versions/{IMPLICIT_VERSION}")
IMPLICIT_ARMS = {
    "clue-target-implicit": {"port": 29846, "name": "clue-target-implicit-think", "think": "think.jsonl.gz",
                             "task": "线索 + 目标 think，加隐式题数据（think 写语义线索 + 目标）+ 联合 MTP"},
    "clue-target-implicit-noexplain": {"port": 29856, "name": "clue-target-implicit-noexplain-think",
                                       "think": "think-no-explanation.jsonl.gz",
                                       "task": "线索 + 目标 think，加隐式题数据（think 只写固定句）+ 联合 MTP"},
    "template-implicit-noexplain": {"port": 29866, "name": "template-implicit-noexplain-think",
                                    "think": "think-no-explanation.jsonl.gz",
                                    "task": "第二轮 think 模板，加隐式题数据（think 只写固定句）+ 联合 MTP"},
}
SYNTH = "context_correction_synth_zh"
ABLATION_ARMS = {
    "template-no-empty-think": {"port": 29876, "name": "template-no-empty-uncovered-think",
                                "task": "第二轮 think 模板，think.empty_prob / uncovered_prob 改为 0（第一轮的 think 分配）+ 联合 MTP"},
    "template-no-synth": {"port": 29886, "name": "template-no-synth-think",
                          "task": "第二轮 think 模板，去掉合成说字数据（即第一轮数据 + PR 158、159 的 think 分配）+ 联合 MTP"},
}
STEPS, SAMPLES_PER_STEP, EXPLAINED_VIEWS = 3000, 188, 3  # as round 2 (scripts/make_data_config.py there)
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


def records(split):
    with gzip.open(IMPLICIT / "records" / f"{split}.jsonl.gz", "rt", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def with_implicit(data, think_file):
    """Round 2's rule for adding its synthetic data: weight so that each record is seen EXPLAINED_VIEWS times."""
    total = sum(s["weight"] for s in data["train"])
    counts = {split: records(split) for split in ("train", "train_ts")}
    share = EXPLAINED_VIEWS * sum(counts.values()) / (STEPS * SAMPLES_PER_STEP)
    weight = total * share / (1 - share)
    d = yaml.safe_load(yaml.safe_dump(data))
    for split, count in counts.items():
        d["train"].append({"dataset_id": IMPLICIT_ID, "version": IMPLICIT_VERSION, "split": split,
                           "weight": round(weight * count / sum(counts.values()), 4)})
    d["think"]["files"].append(str(IMPLICIT / think_file))
    assert Path(d["think"]["files"][-1]).exists(), d["think"]["files"][-1]
    return d, counts, share


def main():
    data = yaml.safe_load((R2 / "configs/train-data-think.yaml").read_text())
    train = yaml.safe_load((R2 / "configs/train-think.yaml").read_text())
    old_code = str(R2 / "shared/code-v2")
    arms = dict(ARMS)
    if "--implicit" in sys.argv[1:]:
        arms.update(IMPLICIT_ARMS)
    if "--ablation" in sys.argv[1:]:
        arms.update(ABLATION_ARMS)
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    assert set(only) <= set(arms), set(only) - set(arms)
    arms = {arm: spec for arm, spec in arms.items() if not only or arm in only}
    for arm, spec in arms.items():
        base = "template" if arm.startswith("template") else "clue-target"
        d = {**data, "think": {**data["think"], "files": think_files(data["think"]["files"], base)}}
        note = ("think 文件与第二轮相同。\n" if base == "template" else
                "think 文件换成 context_correction_clean_zh/versions/think-v3-20261117（同一批行，改写成线索 + 目标），其余与第二轮相同。\n")
        if arm == "template-no-empty-think":
            assert d["think"]["empty_prob"] == d["think"]["uncovered_prob"] == 0.03, d["think"]
            d["think"] = {**d["think"], "empty_prob": 0.0, "uncovered_prob": 0.0}
            note = "think 文件与第二轮相同；think.empty_prob、uncovered_prob 由 0.03 改为 0（第一轮的 think 分配）。\n"
        if arm == "template-no-synth":
            dropped = [s for s in d["train"] if s["dataset_id"] == SYNTH]
            assert [s["split"] for s in dropped] == ["train", "train_ts"], dropped
            d = {**d, "train": [s for s in d["train"] if s["dataset_id"] != SYNTH],
                 "think": {**d["think"], "files": [f for f in d["think"]["files"] if f != OLD_THINK[1]]}}
            assert len(d["think"]["files"]) == len(data["think"]["files"]) - 1
            note = "去掉合成说字数据（train、train_ts）和它的 think 文件，即第一轮的数据；think 分配同第二轮。\n"
        if arm in IMPLICIT_ARMS:
            d, counts, share = with_implicit(d, spec["think"])
            note = (note.rstrip("\n") + f"另加 {IMPLICIT_ID}@{IMPLICIT_VERSION} {counts}，约占样本 {share:.2%}，"
                    f"每条约见 {EXPLAINED_VIEWS} 次，think 文件 {spec['think']}。\n")
        data_path = EXP / f"configs/train-data-{arm}.yaml"
        data_path.write_text(
            f"# 由 scripts/make_configs.py 从 runs/zh-spelling-think-r2-20261115/configs/train-data-think.yaml 生成，不要手改。\n"
            f"# {arm}：" + note + yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=1000))
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
    extra = {**(IMPLICIT_ARMS if "--implicit" in sys.argv[1:] else {}),
             **(ABLATION_ARMS if "--ablation" in sys.argv[1:] else {})}
    if extra:
        experiment = EXP / "experiment.yaml"
        text = experiment.read_text()
        for arm, spec in extra.items():
            if arm in arms and f"id: train-{arm}," not in text:
                text += (f"- {{id: train-{arm}, name: {spec['task']}，1 节点 8 卡全参 3000 步, "
                         f"config: configs/train-{arm}.yaml, depends_on: []}}\n")
        experiment.write_text(text)


if __name__ == "__main__":
    main()
