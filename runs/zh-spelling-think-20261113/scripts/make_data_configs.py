"""Write the think and control data configs from the delivery recipe (v7set34).

Both arms: every source the spelling cleaning covers reads its cleaned version (<version>-ctxclean-20261110), and
each split with enough think records gets a second source that keeps only those records (include_records), so that
the explained records are seen about EXPLAINED_VIEWS times in the run. The think arm adds the ``think`` block.

usage: make_data_configs.py <think version dir>   (writes configs/train-data-{think,control}.yaml)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "configs" / "train-data-v7set34-source.yaml"
CLEAN = Path("/workspace/data/datasets/context_correction_clean_zh/versions/v1-20261110")
# Cleaned sources whose derived version replaces the recipe's version (build.json of each).
CLEANED = ["wenetspeech_clean", "emilia2_mix_utts", "magicdata_ramc_punc", "magicdata_ramc_tt_punc", "aishell4_punc",
           "aishell4_tt_punc", "alimeeting_sdm_punc", "alimeeting_sdm_tt_punc", "emilia2_dialog", "emilia2_short", "vitw"]
METADATA_CACHE = "/workspace/workspace/mingdong/repo/open-audio-llm-ts-robust/runs/ts-robust-acp-20261015/shared/metadata-cache"
STEPS, SAMPLES_PER_STEP = 3000, 188  # 073 consumed 0.94 x 2,000,000 samples in 10,000 steps
EXPLAINED_VIEWS = 3  # each explained record about three times in the run
MIN_SUBSET = 100  # fewer records would round to a zero quota in a 2,000-sample replay window
EXPLAINED_PROB = 0.5
ORDINARY_PROB = 0.03  # ordinary think-on samples about as many as explained ones (see README)


def derived_versions():
    """{(dataset_id, original version): derived version} of the cleaned versions the recipe trains on."""
    found = {}
    for source in CLEANED:
        build = json.loads((CLEAN / source / "build.json").read_text())
        if "derived" in build:  # Lhotse: one derived version, named <original>-ctxclean-20261110
            targets = [(build["derived"].rsplit("-ctxclean-", 1)[0], build["derived"])]
        else:  # audio records: one entry per original version
            targets = [(k, v["derived"]) for k, v in build.items() if k != "positive_units"]
        for original, derived in targets:
            if derived:  # None: nothing changed, the original version stays
                dataset, version = original.split("@")
                found[(dataset, version)] = derived.split("@")[1]
    return found


def main():
    think_dir = Path(sys.argv[1])
    base = yaml.safe_load(SOURCE.read_text())
    swaps = derived_versions()
    think_rows = Counter()
    includes = {}
    for path in sorted(think_dir.glob("*/includes.json")):
        for key, entry in json.loads(path.read_text()).items():
            includes[tuple(key.split("|"))] = entry  # (dataset@version, split) -> {path, records}
    train, swapped, used = [], Counter(), set()
    for source in base["train"]:
        key = (source["dataset_id"], source["version"])
        if key in swaps:
            source = {**source, "version": swaps[key]}
            swapped[f"{key[0]}@{key[1]}"] += 1
        train.append(source)
    total = sum(s["weight"] for s in base["train"])
    samples = STEPS * SAMPLES_PER_STEP
    subsets = []
    for source in train:
        entry = includes.get((f"{source['dataset_id']}@{source['version']}", source["split"]))
        if entry and entry["records"] >= MIN_SUBSET and (source["dataset_id"], source["version"], source["split"]) not in used:
            used.add((source["dataset_id"], source["version"], source["split"]))
            subsets.append((source, entry))
    # Subset weights: share of the run = views x records / samples, solved against the recipe's total weight.
    wanted = sum(EXPLAINED_VIEWS * e["records"] for _, e in subsets)
    share = wanted / samples
    subset_weight = total * share / (1 - share)
    keep = ("dataset_id", "version", "split", "task", "require_clean_pass", "max_duration")
    for source, entry in subsets:
        weight = round(subset_weight * EXPLAINED_VIEWS * entry["records"] / wanted, 4)
        train.append({**{k: source[k] for k in keep if k in source}, "include_records": entry["path"], "weight": weight})
        think_rows[f"{source['dataset_id']}@{source['version']}:{source['split']}"] = entry["records"]
    # Scans and record caches are keyed by content: reuse the delivery recipe's cache instead of rebuilding it.
    control = {**base, "metadata_cache": METADATA_CACHE, "train": train}
    files = sorted(str(p) for p in think_dir.glob("*/think.jsonl.gz"))
    think = {**control, "think": {"files": files, "explained_prob": EXPLAINED_PROB, "ordinary_prob": ORDINARY_PROB}}
    header = (
        "# 由 scripts/make_data_configs.py 从交付配方 v7set34（runs/ts-robust-acp-20261015/configs/train-data-v7set34.yaml）生成，不要手改。\n"
        f"# 说字清洗换版本 {dict(swapped)}；\n"
        f"# 说明记录上采样来源（include_records，合计 weight {round(subset_weight, 3)}，约占样本 {share:.2%}，每条约见 {EXPLAINED_VIEWS} 次）：{dict(think_rows)}。\n")
    (HERE / "configs" / "train-data-control.yaml").write_text(
        header + "# 对照组：没有 think 一节。\n" + yaml.safe_dump(control, allow_unicode=True, sort_keys=False, width=1000))
    (HERE / "configs" / "train-data-think.yaml").write_text(
        header + f"# think 组：有说明的记录以 {EXPLAINED_PROB} 开 think，同来源的普通中文记录以 {ORDINARY_PROB} 写固定句。\n"
        + yaml.safe_dump(think, allow_unicode=True, sort_keys=False, width=1000))
    print(json.dumps({"swapped": swapped, "subsets": think_rows, "subset_weight": subset_weight, "share": share,
                      "think_files": len(files)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
