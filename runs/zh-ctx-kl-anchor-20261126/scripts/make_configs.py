"""Data configs of runs/zh-ctx-kl-anchor-20261126 (README "设计").

- configs/replay-data.yaml: 11.17's task mix as the TS line replays it (oal-clean-teacher
  runs/ts-overlap-20261119/configs/train-data-encreplay.yaml, tasks 25/28; 125 sources on final text, preflighted at
  world size 16), every version taken to its newest training-text final (fleurs_zh v2 -> v3, 2026-11-24).
- configs/ce-data-real.yaml: the five description-record subsets the spelling-think control upsampled (the K arm's
  final-text entries, include lists of context_correction_clean_zh think-v2-20261113-final-v1-20261120), versions to
  the newest final. Weight = records in the list and replay.epoch_samples = window_samples = their total, so one epoch
  draws each record about once.
- configs/ce-data-real-synth.yaml: the same plus context_correction_synth_zh@train-v1-20261114 train (synthetic
  spelling explanations, plain ASR; final-compliant as registered, 2026-11-21).
The CE configs take every non-train section from the replay config (augmentation, noise, RIR, interference, hotwords),
so CE samples are augmented like the recipe; their batches hold one sample; no objective/validation sections.
Include lists whose record IDs carry the version (Lhotse-built sources such as wenetspeech_clean) are rewritten to the
new version under configs/records/; other lists are used as registered.
usage: make_configs.py   (dev machine, no GPU)
"""
import copy
import gzip
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
TS_REPLAY = Path("/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119/configs/train-data-encreplay.yaml")
K_DATA = Path("/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117/configs/"
              "train-data-template-twins-dialog-final-1117.yaml")
THINK_LISTS = "/workspace/data/datasets/context_correction_clean_zh/versions/think-v2-20261113-final-v1-20261120/"
SYNTH = ("context_correction_synth_zh", "train-v1-20261114", "train")
SYNTH_RECORDS = Path("/workspace/data/datasets/context_correction_synth_zh/versions/train-v1-20261114/records/train.jsonl.gz")
# One CE sample per batch; 240 s covers the longest description record (wenetspeech_clean is cut at 120 s).
CE_BATCHING = {"max_duration": 240, "max_samples": 1, "num_buckets": 30, "buffer_size": 2000, "drop_last": False,
               "packing": "sorted"}
CE_DROP = ("train", "objective", "validation", "validation_samples", "validation_max_duration", "replay", "batching")
sys.path.insert(0, str(TS_REPLAY.parents[1] / "scripts"))
from final_versions import final_version  # noqa: E402  (follows superseded_by to the end, guards cycles)

newest = final_version()


def dump(path, data, note):
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=1000)
    path.write_text(f"# {note}\n# 由 scripts/make_configs.py 生成，不要手改。\n{text}")


def lines(path):
    return [line for line in Path(path).read_text().splitlines() if line.strip()]


def description_entries():
    """K's five description subsets (the control's upsampled entries), versions and record lists moved to the newest final."""
    k = yaml.safe_load(K_DATA.read_text())["train"]
    entries = [s for s in k if str(s.get("include_records", "")).startswith(THINK_LISTS)]
    assert sorted((s["dataset_id"], s["split"]) for s in entries) == sorted(
        [("emilia2_mix", "train_2spk"), ("emilia2_mix", "train_3spk"), ("emilia2_dialog", "train_2spk"),
         ("wenetspeech_clean", "train"), ("emilia2_short", "train_zh")]), entries
    out = []
    for s in entries:
        s = copy.deepcopy(s)
        old, new = s["version"], newest(s["dataset_id"], s["version"])
        ids = lines(s["include_records"])
        prefix = f"{s['dataset_id']}@{old}:"
        if any(i.startswith(prefix) for i in ids):
            assert all(i.startswith(prefix) for i in ids), s["include_records"]
            ids = [f"{s['dataset_id']}@{new}:" + i[len(prefix):] for i in ids]
            path = EXP / "configs/records" / Path(s["include_records"]).name.replace(old, new)
            text = "\n".join(ids) + "\n"
            if not path.exists() or path.read_text() != text:  # the list's mtime keys the loader's index cache
                path.write_text(text)
            s["include_records"] = str(path)
        else:
            assert not any("@" in i.split(":", 1)[0] for i in ids), s["include_records"]
        s["version"] = new
        s["weight"] = len(ids)
        out.append(s)
    return out


def main():
    replay = yaml.safe_load(TS_REPLAY.read_text())
    moved = 0
    for s in replay["train"]:
        new = newest(s["dataset_id"], s["version"])
        if new != s["version"]:
            for key in ("include_records", "exclude_records"):
                if key in s:
                    head = lines(s[key])[0]
                    assert not head.startswith(f"{s['dataset_id']}@"), (s, key)  # a versioned list would need rewriting
            s["version"] = new
            moved += 1
    dump(EXP / "configs/replay-data.yaml", replay,
         f"回放流：TS 线任务 25/28 的 11.17 原有任务回放（{TS_REPLAY}），{moved} 个来源换成最新终版。只算对 11.17 的 KL。")

    header = {k: v for k, v in replay.items() if k not in CE_DROP}
    real = description_entries()
    with gzip.open(SYNTH_RECORDS, "rt", encoding="utf-8") as stream:
        synth_count = sum(1 for line in stream if line.strip())
    synth = {"dataset_id": SYNTH[0], "version": newest(SYNTH[0], SYNTH[1]), "split": SYNTH[2], "weight": synth_count}
    for name, train, note in (
        ("real", real, "CE 流（臂 A）：对照组上采样的 5 个说明录音子集，终版 v2 文本（含说字清洗）"),
        ("real-synth", real + [synth], "CE 流（臂 B）：臂 A 加合成说字数据 train（普通识别）"),
    ):
        total = int(sum(s["weight"] for s in train))
        data = {**header, "replay": {"epoch_samples": total, "window_samples": total}, "batching": CE_BATCHING,
                "train": train}
        dump(EXP / f"configs/ce-data-{name}.yaml", data, f"{note}；每个来源的权重 = 记录数，一个 epoch 共 {total} 条。")
        print(name, total, [(s["dataset_id"], s["split"], s["weight"]) for s in train])
    print("replay sources moved to a newer final:", moved)


if __name__ == "__main__":
    main()
