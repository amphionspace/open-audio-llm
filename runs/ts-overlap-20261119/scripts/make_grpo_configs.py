"""tasks/12: write the GRPO configs (from 11.17, the release round's GRPO recipe, plus an overlap-transcription plan).

The control is GRPO round 26 of runs/amphionasr2-grpo-20261101 (worktree oal-grpo, tasks/39): GRPO from the same
weights (whitebox-merge-v18a-es116inc = 11.17) with the round-9 recipe, mined anew on 11.17, five plans, joint MTP from
the 11.17 package's mtp/. This task changes only:
- training text: every train source with a final version (final_versions.py) moves to it (user rule, 2026-11-21);
- plans: a sixth plan, ts-transcribe (grpo_mining_plans.py), with TRANSCRIBE_SHARE of the prompts; the five plans keep
  round 26's shares, the replay keeps the rest;
- code: shared/code-v4 (main 15eacbf) instead of round 26's code-v16, as the context-correction GRPO
  (runs/zh-ctx-grpo-20261120, oal-ctx-grpo) ran it with final text: its mining servers (vLLM 0.32 with the TS SEP and
  per-request enrollment fixes) are copied, with 11.17 as the model.
Outputs: configs/train-data-grpo-transcribe.yaml, configs/train-grpo-transcribe.yaml, configs/zero1-bf16-no-overflow-check.json,
configs/grpo-mining/serve-{asr,tsasr}-{0..6}.yaml and sources-{plain,miss,ts}.txt (round 26's lanes, final versions).
usage: make_grpo_configs.py
"""
from pathlib import Path

import yaml

from final_versions import final_key, final_version

HERE = Path(__file__).resolve().parents[1]
V26 = Path("/workspace/workspace/mingdong/repo/oal-grpo/runs/amphionasr2-grpo-20261101")
CTX = Path("/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120")
START = "/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365"  # 11.17 weights
MTP_INIT = "/workspace/data/packages/amphionasr-1.7b-v3-v18aes-mtp-candidate/model/mtp/model.safetensors"  # 11.17's MTP
TASK = "tasks/12-grpo-transcribe"
CODE = "../shared/code-v4"
# The other behaviour plans hold 0.06-0.15 each (round 26); the new one gets the size of the largest TS plan
# rounds have used (ts-present 0.10 in round 10), a little more because its prompts are the most numerous.
TRANSCRIBE_SHARE = 0.12
# Round 26's mining lanes (scratch/acp-dlv/running/19-grpo-v26-mining.sh), by lane kind.
PLAIN = ["hw_grpo_candidates@v1-20261101:zh", "ts_nonspeech_pool@env-20261101:all", "hard_hw_sft@hard-20261014:train",
         "ts_nonspeech_pool@v1-20261016:all", "ts_nonspeech_pool@babble-20261024:all"]
MISS = ["alimeeting_sdm@asr-20261014-punc-20261102:train", "aishell4@asr-20261014-punc-20261102:train",
        "threed_speaker_zh_far_asr@train-punc-v1-20261102:train", "asr_distant_talk_pool@v1-20261102-punc-20261102:text_zh",
        "ami_sdm@asr-20261014-punc-20261102:train", "asr_distant_talk_pool@v1-20261102:text_en",
        "wenetspeech_clean@train-punc-v1-20261102:train", "aishell2@train-punc-v1-20261102:train",
        "kespeech@train-punc-v1-20261102:train", "gigaspeech@train-punc-v1-20261102:train",
        "librispeech@train-punc-v1-20261102:train", "magicdata_ramc@train-punc-v1-20261102:train"]
TT = "turn-taking-20261014-utts-punc-20261102"
TS = ([f"{d}@{TT}:{s}" for d in ("alimeeting_sdm", "aishell4", "ami_sdm") for s in ("train_2spk", "train_3spk", "train_neg")]
      + [f"magicdata_ramc@{TT}:{s}" for s in ("train_2spk", "train_neg")]
      + [f"chime6@turn-taking-20261014:{s}" for s in ("train_3spk", "train_neg")]
      + [f"emilia2_dialog@dialog-turn-taking-20261013-enroll:{s}" for s in ("train_2spk", "train_neg")]
      + [f"ts_real_train@v2-20261018-utts-punc-20261102:{s}" for s in ("alimeeting", "aishell4")]
      + ["ts_real_train@v1-20261017-utts-punc-20261102:ramc", "ts_real_train@v3-20261020-utts-punc-20261102:emilia2_1spk"]
      + [f"ts_real_train@en-v1-20261101-utts-punc-20261102:{s}" for s in ("ami", "dipco")]
      + [f"ts_real_train@en-v1-20261101-utts:{s}" for s in ("icsi", "notsofar")]
      + [f"ts_scene_pool@{v}" for v in (
          "multi-20261017-punchw-20261102:zh", "multi-20261017-punchw2-20261102:en",
          "alt-twins-20261024-punchw-20261102:zh", "alt-twins-20261024-punchw2-20261102:en",
          "solo-20261020-punchw-20261102:zh", "solo-20261020-punchw2-20261102:en",
          "handover-20261021-punchw-20261102:zh", "handover-20261021-punchw2-20261102:en",
          "xsession-near-20261023-punchw-20261102:zh", "magicdata-near-20261023-punchw-20261102:zh",
          "cvaccent-near-20261101-punchw2-20261102:en", "hard-20261016-punchw-20261102:zh", "hard-20261016-punchw2-20261102:en")]
      + ["aishellmix@pack3-current-audio-record-20260910-punc-20261102:train_neg",
         "librimix@960-current-audio-record-20260910-punc-20261102:train_neg"])


def key(source):
    return f"{source['dataset_id']}@{source['version']}:{source['split']}"


def data_config(final):
    d = yaml.safe_load((V26 / "configs/train-data-grpo-v26.yaml").read_text())
    switched = 0
    for source in d["train"]:
        old = f"{source['dataset_id']}@{source['version']}:"
        for name in ("include_records", "exclude_records"):
            if source.get(name) and any(i.startswith(old) for i in Path(source[name]).read_text().split()):
                raise SystemExit(f"{source[name]} carries {old} ids: rewrite it for the final version first (spec PR #282)")
        version = final(source["dataset_id"], source["version"])
        switched += version != source["version"]
        source["version"] = version
    mined = [{**p, "plan": str(HERE / TASK / "mining" / Path(p["plan"]).name)} for p in d["grpo"]["mined"]]
    assert [Path(p["plan"]).stem for p in mined] == ["plan-hotword", "plan-nonspeech", "plan-miss", "plan-ts-absent",
                                                      "plan-ts-present"], mined
    mined.append({"plan": str(HERE / TASK / "mining/plan-ts-transcribe.jsonl"), "share": TRANSCRIBE_SHARE})
    d["grpo"] = {**d["grpo"], "mined": mined, "mined_only": [final_key(final, k) for k in d["grpo"]["mined_only"]]}
    keys = {key(s) for s in d["train"]}
    lanes = {name: [final_key(final, k) for k in ks] for name, ks in (("plain", PLAIN), ("miss", MISS), ("ts", TS))}
    missing = sorted({k for ks in lanes.values() for k in ks} - keys)
    if missing:
        raise SystemExit(f"mining sources not in the data config: {missing}")
    header = ("# 由 scripts/make_grpo_configs.py 生成，不要手改。= GRPO 第 26 轮的 train-data-grpo-v26.yaml（oal-grpo，从 11.17 权重起、"
              f"第 9 轮配方），train 里有终版的 {switched} 个来源换成训练文本终版（text-final-v1-20261120）；计划指向本任务重挖的结果，"
              f"另加 ts-transcribe 计划（份额 {TRANSCRIBE_SHARE}），其余五份计划的份额不变。\n")
    (HERE / "configs/train-data-grpo-transcribe.yaml").write_text(
        header + yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=1000))
    return lanes, switched


def train_config():
    t = yaml.safe_load((V26 / "configs/train-grpo-v26.yaml").read_text())
    old = "../shared/code-v16"
    rt = t["runtime"]
    rt["pythonpath"] = [{"path": p["path"].replace(old, CODE)} for p in rt["pythonpath"]]
    rt["distributed"]["port"] = 29741
    args = t["task"]["arguments"]
    assert args["--model"] == START, args["--model"]
    args["--external_plugins"] = [{"path": p["path"].replace(old, CODE)} for p in args["--external_plugins"]]
    assert args["--reward_funcs"] == ["qwen3_asr_accuracy", "qwen3_asr_presence", "qwen3_asr_hotword", "qwen3_asr_format"]
    args["--joint_mtp_init"] = MTP_INIT
    t["data"]["config"] = {"path": "train-data-grpo-transcribe.yaml"}
    t["recording"]["attempts_dir"]["path"] = f"../{TASK}/attempts"
    t["tracking"]["pythonpath"] = [{"path": f"{CODE}/src"}]
    t["tracking"]["name"] = "AmphionASR-ts-overlap-grpo-v26recipe-plus-ts-transcribe-plan-from1117-textfinal-mtp-8gpu-400steps"
    for k, v in rt.get("environment", {}).items():
        if "17920" in str(v):
            rt["environment"][k] = str(v).replace("17920", "17922")  # this experiment's W&B forward (scratch tcp_forward.py)
    text = yaml.safe_dump(t, allow_unicode=True, sort_keys=False, width=1000)
    assert "code-v16" not in text and "17920" not in text, text
    header = ("# 由 scripts/make_grpo_configs.py 从 GRPO 第 26 轮的 configs/train-grpo-v26.yaml 生成，不要手改。参数与第 26 轮相同"
              "（第十轮配方：lr 1e-6、beta 0.04、400 步、32 条音频 × 8 个采样、联合 MTP 权重 0.1）。\n"
              "# 改动：代码 shared/code-v4（main 15eacbf，代替第 26 轮的 code-v16），MTP 从 11.17 交付包的 mtp/ 接上（与第 26 轮同一份权重），"
              "数据配置 train-data-grpo-transcribe.yaml，W&B 转发端口、记录目录和名字。\n")
    (HERE / "configs/train-grpo-transcribe.yaml").write_text(header + text)
    (HERE / "configs/zero1-bf16-no-overflow-check.json").write_text((V26 / "configs/zero1-bf16-no-overflow-check.json").read_text())


def mining_configs(lanes):
    out = HERE / "configs/grpo-mining"
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("asr", "tsasr"):
        for lane in range(7):
            c = yaml.safe_load((CTX / f"configs/mining/serve-{kind}-{lane}.yaml").read_text())
            c["runtime"]["pythonpath"] = [{"path": "../../shared/code-v4/src"}]
            c["task"]["model"]["path"] = START
            c["task"]["arguments"]["--served-model-name"] = f"mine-1117-{kind}"
            c["recording"]["attempts_dir"]["path"] = f"../../{TASK}/mining/serve-{kind}-{lane}/attempts"
            gpu, port = c["runtime"]["gpus"][0], c["task"]["arguments"]["--port"]
            header = (f"# 挖题：11.17 的{'TS' if kind == 'tsasr' else '普通识别'}服务（第 {lane} 路，GPU {gpu}，端口 {port}）。"
                      "由 runs/zh-ctx-grpo-20261120 的 configs/mining 经 scripts/make_grpo_configs.py 生成，只改模型、代码、服务名与记录目录。\n")
            (out / f"serve-{kind}-{lane}.yaml").write_text(header + yaml.safe_dump(c, allow_unicode=True, sort_keys=False))
    for name, keys in lanes.items():
        (out / f"sources-{name}.txt").write_text("".join(k + "\n" for k in keys))


def main():
    assert Path(MTP_INIT).exists() and Path(START, "model.safetensors").exists()
    final = final_version()
    lanes, switched = data_config(final)
    train_config()
    mining_configs(lanes)
    print({"switched_train_sources": switched, **{name: len(keys) for name, keys in lanes.items()}})


if __name__ == "__main__":
    main()
