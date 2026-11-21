"""Write this experiment's GRPO configs: the release round's GRPO (oal-grpo round 10, as copied by
runs/zh-spelling-think-grpo-20261115 on oal-think-exp, "08") on top of a context-correction SFT arm.

- configs/train-data-grpo.yaml: the start arm's SFT data config (runs/zh-ctx-clue-target-20261117, the replay), its
  versions moved to the training-text final versions (final_versions.py), with the SFT-only parts dropped (think,
  batching, validation) and 08's GRPO parts added: the two mined-only sources, ``objective``, ``rollout``, and a
  ``grpo`` section with 08's five plans, the ``ctx`` plan and ``context_targets``.
- configs/train-grpo.yaml: 08's train-grpo.yaml (round 10's parameters: lr 1e-6, beta 0.04, 400 steps, 8 x 8
  rollouts) with the start arm's checkpoint-3000, code-v3 (main 15eacbf), the context reward (weight 1.0, as the
  hotword reward) and joint MTP (AGENTS.md "MTP": weight 0.1, 3 steps, decay 0.8, lr 1e-5, from the start's mtp/).
- configs/mining/serve-{asr,tsasr}-{0..6}.yaml: 08's mining servers with the start checkpoint, code-v3 and vLLM 0.32
  (main's TS server needs it), served names and attempts dirs of this experiment.
- configs/mining/sources-{plain,miss,ts,ctx-asr,ctx-ts}.txt: the mining lanes' source keys, 08's lists (round 10's)
  moved to the final versions, plus the ctx lanes (context-correction candidates); every key must be a source of
  the data config.
usage: make_configs.py ARM START TARGETS   (ARM: the SFT arm, e.g. template-no-synth-ts; START: its checkpoint-3000 in
       the ae model cache, as ae-models.sh fetch prints it; TARGETS: where the finalized context targets JSONL goes,
       in the dataset version that also gets the rewritten record lists; it may not exist yet)
"""
import sys
from pathlib import Path

import yaml

from final_versions import final_key, final_version

HERE = Path(__file__).resolve().parents[1]
SFT = Path("/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117")
R08 = Path("/workspace/workspace/mingdong/repo/oal-think-exp/runs/zh-spelling-think-grpo-20261115")
CODE = "../shared/code-v3"
SERVING = "/workspace/workspace/mingdong/miniconda3/envs/vllm-serving-032/bin/python"
SFT_ONLY = ("think", "batching", "validation_samples", "validation_max_duration", "validation")
MINED_ONLY = (("hw_grpo_candidates", "v1-20261101", "zh"), ("ts_nonspeech_pool", "env-20261101", "all"))
# ctx gets the share of the other behaviour plans (0.06-0.10, README "设计"); the replay keeps the rest.
CTX_SHARE = 0.08
MTP = {"--joint_mtp_weight": 0.1, "--joint_mtp_steps": 3, "--joint_mtp_step_decay": 0.8, "--joint_mtp_lr": 1.0e-5}
# 08's mining lanes (scripts/mining.sh there, the release round's), by lane kind.
PLAIN = ["hw_grpo_candidates@v1-20261101:zh", "ts_nonspeech_pool@env-20261101:all", "hard_hw_sft@hard-20261014:train",
         "ts_nonspeech_pool@v1-20261016:all", "ts_nonspeech_pool@babble-20261024:all"]
MISS = ["alimeeting_sdm@asr-20261014-punc-20261102:train", "aishell4@asr-20261014-punc-20261102:train",
        "threed_speaker_zh_far_asr@train-punc-v1-20261102:train", "asr_distant_talk_pool@v1-20261102-punc-20261102:text_zh",
        "ami_sdm@asr-20261014-punc-20261102:train", "asr_distant_talk_pool@v1-20261102:text_en",
        "wenetspeech_clean@train-punc-v1-20261102:train", "aishell2@train-punc-v1-20261102:train",
        "kespeech@train-punc-v1-20261102:train", "gigaspeech@train-punc-v1-20261102:train",
        "librispeech@train-punc-v1-20261102:train", "magicdata_ramc@train-punc-v1-20261102:train"]
TS = ([f"alimeeting_sdm@turn-taking-20261014-utts-punc-20261102:{s}" for s in ("train_2spk", "train_3spk", "train_neg")]
      + [f"aishell4@turn-taking-20261014-utts-punc-20261102:{s}" for s in ("train_2spk", "train_3spk", "train_neg")]
      + [f"ami_sdm@turn-taking-20261014-utts-punc-20261102:{s}" for s in ("train_2spk", "train_3spk", "train_neg")]
      + [f"magicdata_ramc@turn-taking-20261014-utts-punc-20261102:{s}" for s in ("train_2spk", "train_neg")]
      + [f"chime6@turn-taking-20261014:{s}" for s in ("train_3spk", "train_neg")]
      + [f"emilia2_dialog@dialog-turn-taking-20261013-enroll:{s}" for s in ("train_2spk", "train_neg")]
      + [f"ts_real_train@v2-20261018-utts-punc-20261102:{s}" for s in ("alimeeting", "aishell4")]
      + [f"ts_real_train@v1-20261017-utts-punc-20261102:{s}" for s in ("ramc", "alimeeting", "aishell4")]
      + [f"ts_real_train@v3-20261020-utts-punc-20261102:{s}" for s in ("alimeeting_en_mn", "alimeeting_en_mf", "emilia2_1spk")]
      + [f"ts_real_train@en-v1-20261101-utts-punc-20261102:{s}" for s in ("ami", "dipco")]
      + [f"ts_real_train@en-v1-20261101-utts:{s}" for s in ("icsi", "notsofar")]
      + ["ts_scene_pool@multi-20261017-punchw-20261102:zh", "ts_scene_pool@multi-20261017-punchw2-20261102:en",
         "ts_scene_pool@alt-twins-20261024-punchw-20261102:zh", "ts_scene_pool@alt-twins-20261024-punchw2-20261102:en",
         "ts_scene_pool@solo-20261020-punchw-20261102:zh", "ts_scene_pool@solo-20261020-punchw2-20261102:en",
         "ts_scene_pool@handover-20261021-punchw-20261102:zh", "ts_scene_pool@handover-20261021-punchw2-20261102:en",
         "ts_scene_pool@xsession-near-20261023-punchw-20261102:zh", "ts_scene_pool@magicdata-near-20261023-punchw-20261102:zh",
         "ts_scene_pool@cvaccent-near-20261101-punchw2-20261102:en", "ts_scene_pool@hard-20261016-punchw-20261102:zh",
         "ts_scene_pool@hard-20261016-punchw2-20261102:en",
         "aishellmix@pack3-current-audio-record-20260910-punc-20261102:train_neg",
         "librimix@960-current-audio-record-20260910-punc-20261102:train_neg"])
# Context-correction candidates: the synthetic spelling data the arm trains on, and the recipe's subsets of
# recordings with a spoken explanation (include_records entries, which share their key with the full source and
# come after it, so the key resolves to the subset in grpo_mining and GRPOPromptPlan alike).
SYNTH = "context_correction_synth_zh"
# Splits of the explained subsets that are TS-ASR mixtures (the rest, e.g. emilia2_short train_zh, are plain ASR).
TS_SPLITS = {"train_2spk", "train_3spk"}


def key(source):
    return f"{source['dataset_id']}@{source['version']}:{source['split']}"


def record_lists(source, version, out_dir):
    """Rewrite include / exclude lists whose ids carry the old version (records built from Lhotse cuts, e.g.
    wenetspeech_clean) for the final version; split:id lists are version-free and stay."""
    old = f"{source['dataset_id']}@{source['version']}:"
    for name in ("include_records", "exclude_records"):
        if not source.get(name):
            continue
        ids = Path(source[name]).read_text().split()
        if not any(i.startswith(old) for i in ids):
            continue
        assert all(i.startswith(old) for i in ids), source[name]
        new = f"{source['dataset_id']}@{version}:"
        path = out_dir / f"{Path(source[name]).stem}-{version}.txt"
        out_dir.mkdir(parents=True, exist_ok=True)
        text = "".join(new + i[len(old):] + "\n" for i in ids)
        # The list is a registered artifact, and its size and mtime key the loader's index cache
        # (catalog_cache._legacy_source_identity): rewriting the same content forces a full index rebuild.
        if not (path.exists() and path.read_text() == text):
            path.write_text(text)
        source[name] = str(path)


def data_config(arm, targets, final):
    d = yaml.safe_load((SFT / f"configs/train-data-{arm}.yaml").read_text())
    for name in SFT_ONLY:
        d.pop(name, None)
    for source in d["train"]:
        version = final(source["dataset_id"], source["version"])
        record_lists(source, version, targets.parent)
        source["version"] = version
    r08 = yaml.safe_load((R08 / "configs/train-data-grpo.yaml").read_text())
    for dataset_id, version, split in MINED_ONLY:
        source = [s for s in r08["train"] if (s["dataset_id"], s["version"], s["split"]) == (dataset_id, version, split)]
        assert len(source) == 1, (dataset_id, version, split)
        d["train"].append({**source[0], "version": final(dataset_id, version)})
    d["objective"], d["rollout"] = r08["objective"], r08["rollout"]
    plans = [{**p, "plan": str(HERE / "tasks/01-mining" / Path(p["plan"]).name)} for p in r08["grpo"]["mined"]]
    plans.append({"plan": str(HERE / "tasks/01-mining/plan-ctx.jsonl"), "share": CTX_SHARE})
    d["grpo"] = {**r08["grpo"], "mined": plans,
                 "mined_only": [final_key(final, k) for k in r08["grpo"]["mined_only"]],
                 "context_targets": [str(targets)]}
    keys = [key(s) for s in d["train"]]
    explained = sorted({key(s) for s in d["train"] if s.get("include_records")})
    lanes = {"plain": PLAIN, "miss": MISS, "ts": TS}
    lanes = {name: [final_key(final, k) for k in ks] for name, ks in lanes.items()}
    asr_synth = [key(s) for s in d["train"] if s["dataset_id"] == SYNTH and s["split"] == "train"]
    ts_synth = [key(s) for s in d["train"] if s["dataset_id"] == SYNTH and s["split"] == "train_ts"]
    ts_explained = [k for k in explained if k.split(":")[-1] in TS_SPLITS]
    lanes["ctx-asr"] = asr_synth + [k for k in explained if k not in ts_explained]
    lanes["ctx-ts"] = ts_synth + ts_explained
    missing = sorted({k for ks in lanes.values() for k in ks} - set(keys))
    if missing:
        raise SystemExit(f"mining sources not in the data config: {missing}")
    (HERE / "configs").mkdir(exist_ok=True)
    header = (f"# 由 scripts/make_configs.py 生成，不要手改。回放 = runs/zh-ctx-clue-target-20261117/configs/train-data-{arm}.yaml"
              "（起点那轮 SFT 的配方），版本换成训练文本终版（text-final-v1-20261120），去掉 think、batching、validation；"
              "GRPO 部分（两个只挖不放回的来源、objective、rollout、五份计划的份额）同 08，另加 ctx 计划和 context_targets。\n")
    (HERE / "configs/train-data-grpo.yaml").write_text(header + yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=1000))
    return lanes


def train_config(arm, start):
    t = yaml.safe_load((R08 / "configs/train-grpo.yaml").read_text())
    old = "../shared/code-v2"  # 08's snapshot (main f68e180); this experiment's CODE happens to have the same name
    rt = t["runtime"]
    rt["pythonpath"] = [{"path": p["path"].replace(old, CODE)} for p in rt["pythonpath"]]
    rt["distributed"]["port"] = 29736
    args = t["task"]["arguments"]
    args["--model"] = start
    args["--external_plugins"] = [{"path": p["path"].replace(old, CODE)} for p in args["--external_plugins"]]
    assert args["--reward_funcs"] == ["qwen3_asr_accuracy", "qwen3_asr_presence", "qwen3_asr_hotword", "qwen3_asr_format"]
    args["--reward_funcs"] = args["--reward_funcs"] + ["qwen3_asr_context"]
    args["--reward_weights"] = args["--reward_weights"] + [1.0]
    args.update({**MTP, "--joint_mtp_init": f"{start}/mtp/model.safetensors"})
    args["--deepspeed"] = {"path": "zero1-bf16-no-overflow-check.json"}
    t["recording"]["attempts_dir"]["path"] = "../tasks/02-grpo/attempts"
    t["tracking"]["pythonpath"] = [{"path": f"{CODE}/src"}]
    t["tracking"]["name"] = f"AmphionASR-ctx-grpo-v10recipe-ctxplan-from-{arm}-3000-textfinal-mtp-8gpu-400steps"
    text = yaml.safe_dump(t, allow_unicode=True, sort_keys=False, width=1000)
    assert "927844461eb1" not in text
    paths = [p["path"] for p in rt["pythonpath"] + args["--external_plugins"] + t["tracking"]["pythonpath"]]
    assert all(p.startswith(CODE + "/") for p in paths), paths
    header = ("# 由 scripts/make_configs.py 从 08 的 configs/train-grpo.yaml（交付版第十轮的参数）生成，不要手改。\n"
              f"# 改动：起点 {arm} checkpoint-3000（{start}），代码 shared/code-v3（main 15eacbf），奖励加 qwen3_asr_context（权重 1.0），"
              "开联合 MTP（权重 0.1、3 步、衰减 0.8、lr 1e-5，从起点的 mtp/ 接上），记录目录和名字。\n")
    (HERE / "configs/train-grpo.yaml").write_text(header + text)
    zero = (R08 / "configs/zero1-bf16-no-overflow-check.json").read_text()
    (HERE / "configs/zero1-bf16-no-overflow-check.json").write_text(zero)


def mining_configs(arm, start, lanes):
    out = HERE / "configs/mining"
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("asr", "tsasr"):
        for lane in range(7):
            c = yaml.safe_load((R08 / f"configs/mining/serve-{kind}-{lane}.yaml").read_text())
            c["runtime"]["python"] = {"path": SERVING}
            c["runtime"]["pythonpath"] = [{"path": "../../shared/code-v3/src"}]
            c["task"]["model"]["path"] = start
            if kind == "asr":
                # The start carries the TS SEP parameter (thinker.audio_tower.sep_token); vLLM 0.32's own Qwen3-ASR class
                # refuses it. The plugin registers the in-repo class (SEP off: plain ASR) only with the SEP or the MTP-draft
                # switch on, as the eval ASR server has it; the TS servers have the SEP on. No draft is configured here.
                c["runtime"]["environment"]["OPEN_AUDIO_LLM_ENABLE_MTP_DRAFT"] = "1"
            else:
                # grpo_mining names each TS prompt's enrollment length (mm_processor_kwargs.enroll_seconds), as GRPO
                # renders it; vLLM 0.32 refuses per-request mm kwargs (400) unless the server trusts them. The server
                # listens on 127.0.0.1 only. The eval servers need no flag: AmphionEval sends a fixed 5 s enrollment.
                c["task"]["flags"] = c["task"]["flags"] + ["--trust-request-mm-kwargs"]
            c["task"]["arguments"]["--served-model-name"] = f"mine-ctx-{kind}"
            c["recording"]["attempts_dir"]["path"] = f"../../tasks/01-mining/serve-{kind}-{lane}/attempts"
            gpu, port = c["runtime"]["gpus"][0], c["task"]["arguments"]["--port"]
            header = (f"# 挖题：{arm} checkpoint-3000 的{'TS' if kind == 'tsasr' else '普通识别'}服务（第 {lane} 路，GPU {gpu}，端口 {port}）。"
                      "由 08 的 configs/mining 经 scripts/make_configs.py 生成，只改模型、代码、vLLM 0.32、服务名与记录目录。\n")
            (out / f"serve-{kind}-{lane}.yaml").write_text(header + yaml.safe_dump(c, allow_unicode=True, sort_keys=False))
    for name, keys in lanes.items():
        (out / f"sources-{name}.txt").write_text("".join(k + "\n" for k in keys))


def main():
    arm, start, targets = sys.argv[1], sys.argv[2].rstrip("/"), Path(sys.argv[3])
    assert Path(start, "mtp/model.safetensors").exists(), start
    final = final_version()
    lanes = data_config(arm, targets, final)
    train_config(arm, start)
    mining_configs(arm, start, lanes)
    experiment = {
        "version": 1, "name": "zh-ctx-grpo-20261120",
        "purpose": f"在上下文纠偏 SFT（{arm} checkpoint-3000）上按交付版第十轮的 GRPO 配方，加一份上下文纠偏计划和奖励 qwen3_asr_context，"
                   "让 think 关时真实录音的显式说明题多改对，守门和场景 TS 不变差",
        "policy": {"evaluation_backend": "vllm", "human_review_required": False, "training_coverage": "replay_quota",
                   "training_steps": 400},
        "inputs": {"init_model": start},
        "tasks": [{"id": "grpo", "name": "GRPO 400 步（1 节点 8 卡，第十轮配方 + ctx 计划，计划在起点上重挖）",
                   "config": "configs/train-grpo.yaml", "depends_on": []}],
    }
    (HERE / "experiment.yaml").write_text("# 由 scripts/make_configs.py 生成，不要手改。\n"
                                          + yaml.safe_dump(experiment, allow_unicode=True, sort_keys=False, width=1000))
    print({name: len(keys) for name, keys in lanes.items()})


if __name__ == "__main__":
    main()
