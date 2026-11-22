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
With ``--synth-ts`` (question 6, 2026-11-21: template-no-synth restored scene TS but lost Emilia accuracy), two
arms that change only the synthetic train_ts source:
- template-no-synth-ts:     template without synthetic train_ts (synthetic train and the think file stay; the think
  rows of train_ts records are then unused, the file being keyed by dataset@version);
- template-synth-ts-mix:    template with synthetic train_ts mixed online like the clean one-speaker TS sources (the
  online_mix block of aishellmix train_1spk's main entry: interferers, 5% target absent, 7.5% non-speech, RIR, noise
  floor, enrollment degradations). It is the only TS source of the recipe without online mixing.
With ``--synth-ts-twins`` (question 7, 2026-11-21: neither question-6 arm passed; a TTS voice was always the target in
training, so the model transcribed any TTS voice), one arm:
- template-synth-ts-twins:  template plus context_correction_synth_zh@ts-twins-v1-20261121 (scenes of the synthetic
  data's own cloned voices, 30% twins with an absent voice enrolled, scripts/build_synth_ts_twins.py), weighted by round
  2's rule (each record seen about EXPLAINED_VIEWS times) and mixed with the online_mix of ts_scene_pool alt-twins zh,
  whose recipe it copies; its think file is added. Synthetic train_ts stays as it is.

With ``--k`` (question 9, 2026-11-22: J left out whole speakers of long dialogues in plain ASR, and its increment
merged onto 11.17 made Spanish TS worse), one arm:
- template-twins-dialog-final-1117 (K):  J's data on the training-text final versions (every ``text_final_v1*.yaml``
  list, else ``superseded_by``; think-v2 and its include lists from think-v2-20261113-final-v1-20261120), plus
  context_correction_synth_zh@asr-dialog-v1-20261122 (plain-ASR dialogues of the twin scenes' voices) at the twins'
  per-record weight, plus es116's Spanish sources, interference pool and hotword block (the training of 11.17's
  Spanish increment), Spanish at es116's share of samples; trained from the 11.17 package with J's parameters.

usage: make_configs.py [--implicit] [--ablation] [--synth-ts] [--synth-ts-twins] [--k] [arm ...]   (writes configs/train-data-{arm}.yaml, configs/train-{arm}.yaml for
       the named arms, all of them when none is named; configs written on the node are root's and stay as they are)
"""
import gzip
import json
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
SYNTH_TS_ARMS = {
    "template-no-synth-ts": {"port": 29896, "name": "template-no-synth-ts-think",
                             "task": "第二轮 think 模板，去掉合成 train_ts（保留合成 train）+ 联合 MTP"},
    "template-synth-ts-mix": {"port": 29906, "name": "template-synth-ts-onlinemix-think",
                              "task": "第二轮 think 模板，合成 train_ts 改走在线混音（同 aishellmix train_1spk 主条目）+ 联合 MTP"},
}
TWINS_VERSION = "ts-twins-v1-20261121"
TWINS = Path(f"/workspace/data/datasets/{SYNTH}/versions/{TWINS_VERSION}")
TWINS_ARMS = {
    "template-synth-ts-twins": {"port": 29916, "name": "template-synth-ts-twins-think",
                                "task": "第二轮 think 模板，加合成音色自己的 TS 场景（30% 孪生缺席）+ 联合 MTP"},
}
K_ARM = "template-twins-dialog-final-1117"
K_ARMS = {
    K_ARM: {"port": 29926, "name": "twins-asrdialog-esreplay-final",
            "task": "J 的配方换终版文本，加合成音色的普通识别长对话和 es116 的西语回放，从 11.17 起训 + 联合 MTP"},
}
DIALOG_VERSION = "asr-dialog-v1-20261122"
DIALOG = Path(f"/workspace/data/datasets/{SYNTH}/versions/{DIALOG_VERSION}")
THINK_FINAL = "/workspace/data/datasets/context_correction_clean_zh/versions/think-v2-20261113-final-v1-20261120/"
START_1117 = "/workspace/data/packages/amphionasr-1.7b-v3-v18aes-mtp-candidate/model"
# es116: the SFT whose increment is 11.17's Spanish (model_info.json of the 11.17 package).
ES116 = Path("/workspace/workspace/mingdong/repo/open-audio-llm-ts-robust/runs/ts-robust-acp-20261015/tasks/01-train/"
             "attempts/116/data-effective.yaml")
CATALOG = Path("/workspace/data/registry/catalog")
FINAL_SUFFIX = "-final-v1-20261120"
STEPS, SAMPLES_PER_STEP, EXPLAINED_VIEWS = 3000, 188, 3  # as round 2 (scripts/make_data_config.py there)
CODE = EXP / "shared/code-v1"


def final_version():
    """Training-text final version of a version: ``<version>-final-v1-20261120`` when registered (every
    ``text_final_v1*.yaml`` list: Spanish has its own), else its ``superseded_by`` (the -ctxclean intermediates), else
    the version itself (no final; Portuguese and the synthetic data are final as they are)."""
    finals = {(d["dataset_id"], d["version"]) for path in CATALOG.glob("text_final_v1*.yaml")
              for d in yaml.safe_load(path.read_text())}
    superseded = {(d["dataset_id"], d["version"]): (d.get("provenance") or {}).get("superseded_by")
                  for path in CATALOG.glob("*.yaml") for d in yaml.safe_load(path.read_text()) or []}

    def final(dataset_id, version):
        if (dataset_id, version + FINAL_SUFFIX) in finals:
            return version + FINAL_SUFFIX
        target = superseded.get((dataset_id, version))
        if target:
            name, new = target.split("@", 1)
            if name != dataset_id or (name, new) not in finals:
                raise ValueError(f"{dataset_id}@{version} superseded by {target}, which is not a text-final version")
            return new
        return version

    return final


def spanish(source):
    return source["dataset_id"].endswith("_es") or source["split"] == "es" or source["version"].startswith("es-")


def to_k(d, twins_weight, twins_count):
    """J's data config (``d``) -> K's; returns it and the numbers for the header."""
    final = final_version()
    d = yaml.safe_load(yaml.safe_dump(d))
    includes = {key: item["path"] for path in Path(THINK_FINAL).glob("*/includes.json")
                for key, item in json.loads(path.read_text()).items()}
    for s in d["train"]:
        version = final(s["dataset_id"], s["version"])
        if s.get("include_records"):
            assert s["include_records"].startswith(OLD_THINK[0]), s
            s["include_records"] = includes[f"{s['dataset_id']}@{version}|{s['split']}"]
        if s.get("exclude_records"):
            # split:id lists carry over; only lists of version-carrying IDs (Lhotse sources) would need rewriting.
            assert not any(i.startswith(f"{s['dataset_id']}@") for i in Path(s["exclude_records"]).read_text().split()), s
        s["version"] = version
    for v in d["validation"]:
        v["version"] = final(v["dataset_id"], v["version"])
    d["think"]["files"] = [THINK_FINAL + f[len(OLD_THINK[0]):] if f.startswith(OLD_THINK[0]) else f
                           for f in d["think"]["files"]]
    assert all(Path(f).exists() for f in d["think"]["files"]), d["think"]["files"]
    with gzip.open(DIALOG / "records/train_asr_dialog.jsonl.gz", "rt", encoding="utf-8") as stream:
        dialogs = sum(1 for line in stream if line.strip())
    twins = [i for i, s in enumerate(d["train"]) if (s["dataset_id"], s["version"]) == (SYNTH, TWINS_VERSION)]
    assert len(twins) == 1, twins
    # The twins' weight per record: each dialogue is seen about as often as a twin scene.
    dialog = {"dataset_id": SYNTH, "version": DIALOG_VERSION, "split": "train_asr_dialog",
              "weight": round(twins_weight * dialogs / twins_count, 4)}
    d["train"].insert(twins[0] + 1, dialog)
    es = yaml.safe_load(ES116.read_text())
    es_sources = [s for s in es["train"] if spanish(s)]
    assert len(es_sources) == 15 and not any(spanish(s) for s in d["train"]), [k for k in es_sources]
    assert not any("lookalike_prob" in (s.get("online_mix") or {}) for s in es_sources)
    share = sum(s["weight"] for s in es_sources) / sum(s["weight"] for s in es["train"])
    rest = sum(s["weight"] for s in d["train"])
    scale = share * rest / (1 - share) / sum(s["weight"] for s in es_sources)
    d["train"] += [{**s, "version": final(s["dataset_id"], s["version"]), "weight": round(s["weight"] * scale, 4)}
                   for s in es_sources]
    es_pool = [s for s in es["interference_sources"] if s["language"] == "es"]
    assert len(es_pool) == 1, es_pool
    d["interference_sources"] = d["interference_sources"] + es_pool
    # 11.17's Spanish increment was trained with es116's hotword block (miss / confusable-miss / same-language
    # distractors, eshw-confmiss08); J's is v7set34's, without them.
    d["hotwords"] = es["hotwords"]
    moved = sum(1 for s in d["train"] if s["version"].endswith(FINAL_SUFFIX))
    return d, {"dialogs": dialogs, "dialog_weight": dialog["weight"], "es_share": share, "es_sources": len(es_sources),
               "final_sources": moved, "sources": len(d["train"])}


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
    if "--synth-ts" in sys.argv[1:]:
        arms.update(SYNTH_TS_ARMS)
    if "--synth-ts-twins" in sys.argv[1:]:
        arms.update(TWINS_ARMS)
    if "--k" in sys.argv[1:]:
        arms.update(K_ARMS)
    only =[a for a in sys.argv[1:] if not a.startswith("--")]
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
        if arm == "template-no-synth-ts":
            dropped = [s for s in d["train"] if s["dataset_id"] == SYNTH and s["split"] == "train_ts"]
            assert len(dropped) == 1 and "online_mix" not in dropped[0], dropped
            d = {**d, "train": [s for s in d["train"] if s is not dropped[0]]}
            note = "去掉合成 train_ts；合成 train 和 think 文件保留（train_ts 记录的 think 行不再用到）；其余同 template。\n"
        if arm == "template-synth-ts-mix":
            solo = [s for s in d["train"] if (s["dataset_id"], s["split"]) == ("aishellmix", "train_1spk")][0]
            assert solo["online_mix"]["absent_prob"] == 0.05 and solo["online_mix"]["nonspeech_prob"] == 0.075, solo
            ts = [s for s in d["train"] if s["dataset_id"] == SYNTH and s["split"] == "train_ts"]
            assert len(ts) == 1 and "online_mix" not in ts[0], ts
            d = {**d, "train": [{**s, "online_mix": yaml.safe_load(yaml.safe_dump(solo["online_mix"]))} if s is ts[0] else s
                                for s in d["train"]]}
            note = ("合成 train_ts 加上 aishellmix train_1spk 主条目的 online_mix（干扰人 0–4 个、5% 目标缺席、7.5% 非语音、"
                    "混响、底噪、注册音频劣化），权重不变；其余同 template。\n")
        if arm in ("template-synth-ts-twins", K_ARM):
            alt = [s for s in d["train"] if (s["dataset_id"], s["version"], s["split"])
                   == ("ts_scene_pool", "alt-twins-20261024-punchw-20261102", "zh")]
            assert len(alt) == 1 and alt[0]["online_mix"]["absent_prob"] == 0.0, alt
            ts = [i for i, s in enumerate(d["train"]) if s["dataset_id"] == SYNTH and s["split"] == "train_ts"]
            assert len(ts) == 1, ts
            with gzip.open(TWINS / "records/train_ts_twins.jsonl.gz", "rt", encoding="utf-8") as stream:
                count = sum(1 for line in stream if line.strip())
            # Round 2's rule (with_implicit): the weight at which each record is seen EXPLAINED_VIEWS times.
            total = sum(s["weight"] for s in d["train"])
            share = EXPLAINED_VIEWS * count / (STEPS * SAMPLES_PER_STEP)
            twins = {"dataset_id": SYNTH, "version": TWINS_VERSION, "split": "train_ts_twins",
                     "weight": round(total * share / (1 - share), 4),
                     "online_mix": yaml.safe_load(yaml.safe_dump(alt[0]["online_mix"]))}
            d = {**d, "train": d["train"][: ts[0] + 1] + [twins] + d["train"][ts[0] + 1:],
                 "think": {**d["think"], "files": d["think"]["files"] + [str(TWINS / "think.jsonl.gz")]}}
            assert Path(d["think"]["files"][-1]).exists()
            note = (f"另加 {SYNTH}@{TWINS_VERSION} train_ts_twins {count} 条（合成音色自己的 TS 场景，30% 孪生缺席），"
                    f"约占样本 {share:.2%}，每条约见 {EXPLAINED_VIEWS} 次，online_mix 同 ts_scene_pool alt-twins zh，"
                    "加它的 think 文件；合成 train_ts 不变，其余同 template。\n")
        if arm == K_ARM:
            d, k = to_k(d, twins["weight"], count)
            note = (f"J（template-synth-ts-twins）的数据换训练文本终版（{k['final_sources']}/{k['sources']} 个来源是 "
                    f"-final-v1-20261120 版本，think-v2 和 include 列表换 think-v2-20261113-final-v1-20261120）；"
                    f"另加 {SYNTH}@{DIALOG_VERSION} train_asr_dialog {k['dialogs']} 条（合成音色的普通识别长对话），"
                    f"权重 {k['dialog_weight']}（与孪生场景每条同权）；另加 es116 的 {k['es_sources']} 个西语来源、"
                    f"西语干扰人池和热词设置，西语占样本 {k['es_share']:.2%}（同 es116）。\n")
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
        if arm == K_ARM:
            args["--model"] = START_1117
        t["preflight"][0]["arguments"]["--data_config"] = str(data_path)
        t["recording"]["attempts_dir"]["path"] = f"../tasks/01-train-{arm}/attempts"
        t["tracking"]["pythonpath"] = [str(CODE / "src")]
        t["tracking"]["name"] = (f"AmphionASR-ctx-q9-{spec['name']}-mtp-from1117-3k-1node-codev1" if arm == K_ARM else
                                 f"AmphionASR-ctx-{spec['name']}-r2recipe-mtp-from-v10interp050-3k-1node-codev1")
        t["storage"]["local_root"] = str(EXP.parents[1])
        text = yaml.safe_dump(t, allow_unicode=True, sort_keys=False, width=1000)
        assert old_code not in text and "zh-spelling-think-r2-20261115/configs/train-data" not in text
        (EXP / f"configs/train-{arm}.yaml").write_text(
            f"# 由 scripts/make_configs.py 从 runs/zh-spelling-think-r2-20261115/configs/train-think.yaml 生成，不要手改。\n"
            f"# {arm} 组：第二轮的训练参数{'（起点换成 11.17 交付包 ' + START_1117 + '）' if arm == K_ARM else ''}，加联合 MTP（权重 0.1、3 步、衰减 0.8、lr 1e-4，从 11.17 交付包的 mtp/ 初始化），"
            f"代码 shared/code-v1（main 0a1b117），数据 configs/train-data-{arm}.yaml。\n" + text)
        print(arm, data_path)
    extra = {**(IMPLICIT_ARMS if "--implicit" in sys.argv[1:] else {}),
             **(ABLATION_ARMS if "--ablation" in sys.argv[1:] else {}),
             **(SYNTH_TS_ARMS if "--synth-ts" in sys.argv[1:] else {}),
             **(TWINS_ARMS if "--synth-ts-twins" in sys.argv[1:] else {}),
             **(K_ARMS if "--k" in sys.argv[1:] else {})}
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
