"""Configs of the paired mainline baseline K0 (README "设计"): K's round (runs/zh-ctx-clue-target-20261117 question 9 SFT,
then runs/zh-ctx-grpo-k-20261122 GRPO) with the context-correction parts taken out and everything else as K's.

Taken out (the context-correction package as K trained it):
- the synthetic spelling data, context_correction_synth_zh splits train, train_ts and train_ts_twins;
- the SFT's think section (round-2 think: think rows on the recordings with a spoken explanation and on the synthetic
  data, plus the small empty / ordinary think shares), so every sample trains as plain ASR / TS-ASR (think off);
- in GRPO, the ctx plan, the context targets, the qwen3_asr_context reward and the two ctx mining lanes.
Kept: context_correction_synth_zh train_asr_dialog (plain-ASR long dialogues added in K against the twins' long-dialogue
drops, not context correction), every other source, version, weight and parameter of K's own configs.

- sft:        configs/train-data-k0.yaml from K's SFT data config, configs/train-k0.yaml from K's SFT config (paths,
              attempts dir, port, W&B forward and run name changed), experiment.yaml.
- grpo START: configs/train-data-grpo.yaml, configs/train-grpo.yaml and configs/mining/ from kgrpo's own generated
              configs (runs/zh-ctx-grpo-k-20261122/configs, same versions), the start replaced by START (K0's
              checkpoint-3000 in the ae model cache); every other mined plan keeps its share, the replay keeps the rest.
Arm k0s2 (README "风险": the gain landed near the gate, so a second K0 with other seeds estimates run-to-run noise):
the same configs with the trainer --seed, the data seed and the mining epochs changed, under its own file and task names.
usage: make_configs.py sft [k0s2] | make_configs.py grpo START [k0s2]
"""
import math
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
Q9 = Path("/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117")
KGRPO = REPO / "runs/zh-ctx-grpo-k-20261122"
K_ARM = "template-twins-dialog-final-1117"
SYNTH, CTX_SPLITS = "context_correction_synth_zh", ("train", "train_ts", "train_ts_twins")
PROXY = "http://10.119.18.89:17920"  # dev container IP since 2026-11-26 (K's configs name the old one)
OLD_PROXY = "http://10.119.17.252:17920"
# per arm: config files, task dirs, ports, run-name tag and seeds (trainer --seed, data seed; None = K's own)
ARMS = {
    "k0": dict(sft_data="train-data-k0.yaml", sft_cfg="train-k0.yaml", grpo_data="train-data-grpo.yaml",
               grpo_cfg="train-grpo.yaml", mining_cfg="mining", sft="01-sft-k0", mining="02-mining", grpo="03-grpo",
               ports=(29946, 29956), tag="", seeds=None),
    "k0s2": dict(sft_data="train-data-k0s2.yaml", sft_cfg="train-k0s2.yaml", grpo_data="train-data-grpo-k0s2.yaml",
                 grpo_cfg="train-grpo-k0s2.yaml", mining_cfg="mining-k0s2", sft="05-sft-k0s2", mining="06-mining-k0s2",
                 grpo="07-grpo-k0s2", ports=(29947, 29957), tag="-seed2", seeds=(43, 20261128)),
}


def dump(path, header, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=1000))


def drop_ctx(train):
    kept = [s for s in train if not (s["dataset_id"] == SYNTH and s["split"] in CTX_SPLITS)]
    assert len(train) - len(kept) == len(CTX_SPLITS), [s["split"] for s in train if s["dataset_id"] == SYNTH]
    assert any(s["dataset_id"] == SYNTH and s["split"] == "train_asr_dialog" for s in kept)
    return kept


def write_experiment(init_model):
    """experiment.yaml; a task only once its config exists (grpo written by ``grpo START`` after that arm's SFT)."""
    tasks = [{"id": "sft-k0", "name": "K0 SFT：K 的数据去掉纠偏部分，从 11.17 起训 3000 步（1 节点 8 卡，K 的参数）",
              "config": "configs/train-k0.yaml", "depends_on": []}]
    if (HERE / "configs/train-grpo.yaml").exists():
        tasks.append({"id": "grpo", "name": "K0 GRPO 400 步（1 节点 8 卡，K 那一轮的配方去掉 ctx 计划和奖励，计划在 K0 上重挖）",
                      "config": "configs/train-grpo.yaml", "depends_on": ["sft-k0"]})
    if (HERE / "configs/train-k0s2.yaml").exists():
        tasks.append({"id": "sft-k0s2", "name": "K0 第二个种子的 SFT：同 K0，只换训练种子和数据种子（估运行间波动）",
                      "config": "configs/train-k0s2.yaml", "depends_on": []})
    if (HERE / "configs/train-grpo-k0s2.yaml").exists():
        tasks.append({"id": "grpo-k0s2", "name": "K0 第二个种子的 GRPO 400 步：同 K0 的配方，换种子，计划在它自己的 SFT 上重挖",
                      "config": "configs/train-grpo-k0s2.yaml", "depends_on": ["sft-k0s2"]})
    experiment = {
        "version": 1, "name": "zh-ctx-mainline-pair-20261127",
        "purpose": "主线一轮（SFT → GRPO）的成对对照：K 那一轮去掉上下文纠偏部分（合成说字数据、think、GRPO 的纠偏计划和奖励），"
                   "其余同 K；和已有的 K + GRPO（kgrpo）比，得到纠偏数据和方法在一轮 SFT → GRPO 里的净收益和回退",
        "policy": {"evaluation_backend": "vllm", "human_review_required": False, "training_coverage": "replay_quota",
                   "training_steps": "SFT 3000, GRPO 400 (as K)"},
        "inputs": {"init_model": init_model},
        "tasks": tasks,
    }
    dump(HERE / "experiment.yaml", "# 由 scripts/make_configs.py 生成，不要手改。\n", experiment)


def sft(arm):
    A = ARMS[arm]
    d = yaml.safe_load((Q9 / f"configs/train-data-{K_ARM}.yaml").read_text())
    d["train"] = drop_ctx(d["train"])
    assert "think" in d
    d.pop("think")
    # K's window of 2000 passes only by the largest-remainder rounding: without the three sources, the smallest one
    # (chime6 train_2spk) is rounded to a quota of 0 and the sampler refuses to start (01-sft-k0 attempt 001's
    # preflight). Same rule as kgrpo's make_configs.py: the smallest divisor of epoch_samples at which every source's
    # exact quota is at least 1; source shares are unchanged.
    weights = [s["weight"] for s in d["train"]]
    epoch, window = d["replay"]["epoch_samples"], d["replay"]["window_samples"]
    need = max(window, math.ceil(sum(weights) / min(weights)))
    d["replay"]["window_samples"] = next(x for x in range(need, epoch + 1) if epoch % x == 0)
    if A["seeds"]:
        d["seed"] = A["seeds"][1]
    data = HERE / "configs" / A["sft_data"]
    dump(data, f"# 由 scripts/make_configs.py sft 生成，不要手改。= K 的 SFT 数据配置（zh-ctx-clue-target-20261117 configs/train-data-{K_ARM}.yaml），"
               "去掉合成说字数据 train、train_ts、train_ts_twins 和 think 段；replay.window_samples 按 kgrpo 的规则加大到每个来源配额至少 1；其余来源、版本、权重不变。\n", d)
    t = yaml.safe_load((Q9 / f"configs/train-{K_ARM}.yaml").read_text())
    t["data"]["config"] = str(data)
    assert len(t["preflight"]) == 1
    t["preflight"][0]["arguments"]["--data_config"] = str(data)
    t["recording"]["attempts_dir"]["path"] = f"../tasks/{A['sft']}/attempts"
    t["runtime"]["cwd"] = str(REPO)
    t["storage"]["local_root"] = str(REPO)  # checkpoints go to whai:open-audio-llm/runs/zh-ctx-mainline-pair-20261127/...
    t["runtime"]["distributed"]["port"] = A["ports"][0]
    if A["seeds"]:
        assert t["task"]["arguments"]["--seed"] == 42
        t["task"]["arguments"]["--seed"] = A["seeds"][0]
    env = t["runtime"]["environment"]
    assert env["WANDB_HTTP_PROXY"] == env["WANDB_HTTPS_PROXY"] == OLD_PROXY
    env["WANDB_HTTP_PROXY"] = env["WANDB_HTTPS_PROXY"] = PROXY
    t["tracking"]["name"] = "AmphionASR-ctx-mainline-pair-k0-noctx-nothink-asrdialog-esreplay-final-mtp-from1117-3k-1node-codev1" + A["tag"]
    dump(HERE / "configs" / A["sft_cfg"],
         f"# 由 scripts/make_configs.py sft 从 K 的 SFT 配置（zh-ctx-clue-target-20261117 configs/train-{K_ARM}.yaml）生成，不要手改。\n"
         f"# 只改：数据配置换成 {A['sft_data']}，记录目录、工作目录与存储根、端口、W&B 转发（开发机 IP 11-26 起为 10.119.18.89）、run 名"
         + ("、训练种子和数据种子（第二个种子）" if A["seeds"] else "") + "。\n", t)
    write_experiment(t['task']['arguments']['--model'])
    print("sft configs:", len(d["train"]), "sources")


def grpo(start, arm):
    A = ARMS[arm]
    assert Path(start, "mtp/model.safetensors").exists(), start
    d = yaml.safe_load((KGRPO / "configs/train-data-grpo.yaml").read_text())
    d["train"] = drop_ctx(d["train"])
    g = d["grpo"]
    assert [Path(p["plan"]).name for p in g["mined"]][-1] == "plan-ctx.jsonl"
    g["mined"] = [{**p, "plan": str(HERE / "tasks" / A["mining"] / Path(p["plan"]).name)} for p in g["mined"][:-1]]
    g.pop("context_targets")
    # Same window rule as kgrpo's make_configs.py: the smallest divisor of epoch_samples at which every source's exact
    # quota is at least 1 (fewer sources here, so it can only stay or shrink toward the SFT's 2000).
    weights = [s["weight"] for s in d["train"]]
    epoch = d["replay"]["epoch_samples"]
    need = max(2000, math.ceil(sum(weights) / min(weights)))
    d["replay"]["window_samples"] = next(x for x in range(need, epoch + 1) if epoch % x == 0)
    if A["seeds"]:
        d["seed"] = A["seeds"][1]
    dump(HERE / "configs" / A["grpo_data"],
         "# 由 scripts/make_configs.py grpo 生成，不要手改。= kgrpo 的 configs/train-data-grpo.yaml，去掉合成说字数据 train、train_ts、"
         "train_ts_twins，去掉 ctx 计划和 context_targets；其余计划份额不变，回放拿剩下的。\n", d)
    t = yaml.safe_load((KGRPO / "configs/train-grpo.yaml").read_text())
    a = t["task"]["arguments"]
    old_start = a["--model"]
    assert a["--reward_funcs"][-1] == "qwen3_asr_context"
    a["--reward_funcs"], a["--reward_weights"] = a["--reward_funcs"][:-1], a["--reward_weights"][:-1]
    a["--model"] = start
    assert a["--joint_mtp_init"] == f"{old_start}/mtp/model.safetensors"
    a["--joint_mtp_init"] = f"{start}/mtp/model.safetensors"
    t["recording"]["attempts_dir"]["path"] = f"../tasks/{A['grpo']}/attempts"
    t["runtime"]["distributed"]["port"] = A["ports"][1]
    if A["seeds"]:
        assert a["--seed"] == 42
        a["--seed"] = A["seeds"][0]
    env = t["runtime"].get("environment", {})
    for k in ("WANDB_HTTP_PROXY", "WANDB_HTTPS_PROXY"):
        if k in env:
            assert env[k] == OLD_PROXY
            env[k] = PROXY
    t["tracking"]["name"] = "AmphionASR-ctx-mainline-pair-k0-grpo-v10recipe-noctxplan-from-k0-3000-textfinal-mtp-8gpu-400steps" + A["tag"]
    text = yaml.safe_dump(t, allow_unicode=True, sort_keys=False, width=1000)
    assert old_start not in text and "qwen3_asr_context" not in text
    (HERE / "configs" / A["grpo_cfg"]).write_text(
        "# 由 scripts/make_configs.py grpo 从 kgrpo 的 configs/train-grpo.yaml 生成，不要手改。\n"
        f"# 只改：起点换成 K0 checkpoint-3000（{start}）和它的 mtp/，去掉奖励 qwen3_asr_context，记录目录、端口、W&B 转发、run 名"
        + ("、训练种子和数据种子（第二个种子）" if A["seeds"] else "") + "。\n" + text)
    (HERE / "configs/zero1-bf16-no-overflow-check.json").write_text((KGRPO / "configs/zero1-bf16-no-overflow-check.json").read_text())
    out = HERE / "configs" / A["mining_cfg"]
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((KGRPO / "configs/mining").glob("serve-*.yaml")):
        s = f.read_text()
        assert s.count(f"path: {old_start}\n") == 1, f
        s = s.replace(f"path: {old_start}\n", f"path: {start}\n").replace("K 的", "K0 的")
        if arm != "k0":  # K0's serve records sit in kgrpo's tasks/01-mining name (02-mining README); this arm's in its own dir
            rec = f"path: ../../tasks/01-mining/{f.stem}/attempts\n"
            assert s.count(rec) == 1, f
            s = s.replace(rec, f"path: ../../tasks/{A['mining']}/{f.stem}/attempts\n")
        (out / f.name).write_text(s)
    for name in ("plain", "miss", "ts"):
        (out / f"sources-{name}.txt").write_text((KGRPO / f"configs/mining/sources-{name}.txt").read_text())
    keys = {f"{s['dataset_id']}@{s['version']}:{s['split']}" for s in d["train"]}
    lanes = {n: (out / f"sources-{n}.txt").read_text().split() for n in ("plain", "miss", "ts")}
    missing = sorted({k for ks in lanes.values() for k in ks} - keys)
    assert not missing, missing
    write_experiment(yaml.safe_load((HERE / "configs" / A["sft_cfg"]).read_text())["task"]["arguments"]["--model"])
    print("grpo configs:", len(d["train"]), "sources, window", d["replay"]["window_samples"],
          {n: len(ks) for n, ks in lanes.items()}, "plans", [Path(p["plan"]).name for p in g["mined"]])


if __name__ == "__main__":
    {"sft": lambda: sft(sys.argv[2] if len(sys.argv) > 2 else "k0"),
     "grpo": lambda: grpo(sys.argv[2].rstrip("/"), sys.argv[3] if len(sys.argv) > 3 else "k0")}[sys.argv[1]]()
