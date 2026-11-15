"""Write the round-2 think data config from round 1's think config.

Round 1's sources, cleaned versions and explained-record upsampling stay as they are. Added:
- the synthetic spelling data (train: ASR, train_ts: TS-ASR), weighted so that each record is seen about
  EXPLAINED_VIEWS times in the run, the rule round 1 used for its upsampled explained records;
- its think file;
- ``empty_prob`` and ``uncovered_prob`` at round 1's ``ordinary_prob``: empty answers, other languages and unscanned
  sources write the fixed sentence at the same rate as ordinary records of the scanned sources, so think-on examples
  without an explanation are an even draw over the whole recipe.

usage: make_data_config.py   (writes configs/train-data-think.yaml)
"""
import gzip
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
ROUND1 = HERE.parent / "zh-spelling-think-20261113" / "configs" / "train-data-think.yaml"
SYNTH_ID, SYNTH_VERSION = "context_correction_synth_zh", "train-v1-20261114"
SYNTH = Path(f"/workspace/data/datasets/{SYNTH_ID}/versions/{SYNTH_VERSION}")
SYNTH_SPLITS = ("train", "train_ts")
STEPS, SAMPLES_PER_STEP = 3000, 188  # as round 1: 073 consumed 0.94 x 2,000,000 samples in 10,000 steps
EXPLAINED_VIEWS = 3  # as round 1


def records(split):
    with gzip.open(SYNTH / "records" / f"{split}.jsonl.gz", "rt", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def main():
    config = yaml.safe_load(ROUND1.read_text())
    think = config["think"]
    total = sum(s["weight"] for s in config["train"])
    counts = {split: records(split) for split in SYNTH_SPLITS}
    samples = STEPS * SAMPLES_PER_STEP
    share = EXPLAINED_VIEWS * sum(counts.values()) / samples
    synth_weight = total * share / (1 - share)
    for split, count in counts.items():
        config["train"].append({"dataset_id": SYNTH_ID, "version": SYNTH_VERSION, "split": split,
                                "weight": round(synth_weight * count / sum(counts.values()), 4)})
    probability = think["ordinary_prob"]
    config["think"] = {**think, "files": [*think["files"], str(SYNTH / "think.jsonl.gz")],
                       "empty_prob": probability, "uncovered_prob": probability}
    header = (
        "# 由 scripts/make_data_config.py 从第一轮 think 组的数据配置（runs/zh-spelling-think-20261113/configs/train-data-think.yaml）生成，不要手改。\n"
        f"# 加入合成说字数据 {SYNTH_ID}@{SYNTH_VERSION} {counts}，合计 weight {round(synth_weight, 3)}，约占样本 {share:.2%}，每条约见 {EXPLAINED_VIEWS} 次。\n"
        f"# think：有说明的记录以 {think['explained_prob']} 开 think；扫描过的来源的普通中文记录、空答案、扫描没覆盖的记录（其他来源、其他语种）都以 {probability} 写固定句。\n")
    (HERE / "configs" / "train-data-think.yaml").write_text(
        header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False, width=1000))
    print(json.dumps({"synth_records": counts, "synth_weight": synth_weight, "share": share, "total_before": total,
                      "think_files": len(config["think"]["files"])}, indent=1))


if __name__ == "__main__":
    main()
