"""Training data configurations of this round: arm A (control) and arm B (+ target-speaker ASR tasks).

Both start from the frozen configuration of zh-nosimilar-control-acp-20261016 (prepare 002: real meetings 49, Emilia2
dialogue 20, synthetic multi-speaker 12, single-speaker ASR 12.6, rest 6.4; its preflight passed on 2026-10-16, the
training was interrupted before any checkpoint).

A  = that configuration with the Emilia2 dialogue exclusion list replaced by the 0.80 split cap (4+ speakers, MOSS
     speaker merges and, new, the 1,345 windows where MOSS split one voice into two; similar-voice-views-20261014
     split-cap-0.80, copied under /workspace/data).
B  = A with every source scaled to 80 and the target-speaker (TS-ASR) sources of the TS-robust recipe v7set23 added
     as 20 (their relative weights, online-mixing settings and exclusion lists unchanged), plus the global sections
     only TS online mixing reads (interference, RIR, floor, nonspeech sources). Batching, replay, augmentation and
     noise sources stay those of A, so the diarization part of B is sampled and augmented as in A.
     A's speed perturbation (augmentation.speed_prob 0.25) keeps every source on the JSONL path (the Lance reader does
     not implement it), and that path accepts online mixing only for present-target records, while each ts_real_train
     split is half absent-target windows (the TS line served them through Lance). So each ts_real_train source is
     split in two, the weight shared by record count: its present records keep online mixing, its absent records are
     used as recorded (no online mixing); the record lists sit next to the version under /workspace/data.

Catalog: a directory with a frozen copy of the registry catalog (/workspace/data/registry/catalog) and the entries of
the control's frozen catalog that the registry does not declare (synthetic and localized meeting views). Where both
declare a version the registry entry is used: the frozen Emilia2 dialogue entry still points at the m4a audio index,
whose files were deleted on 2026-10-18 (the registry entry points at the FLAC copy).
Roots: the registry roots with the control's roots on top (legacy_asr keeps the localized AISHELL-1 copy).

    python make_recipes.py
"""
import gzip
import json
import shutil
from pathlib import Path

import yaml

RUN = Path(__file__).resolve().parents[1]
CONTROL = Path("/workspace/workspace/mingdong/repo/open-audio-llm/runs/zh-nosimilar-control-acp-20261016/tasks/01-prepare/attempts/002/artifacts")
TS_RECIPE = Path("/workspace/workspace/mingdong/repo/open-audio-llm-ts-robust/runs/ts-robust-acp-20261015/configs/train-data-v7set23.yaml")
REGISTRY = Path("/workspace/data/registry")
EMILIA2 = ("emilia2_zh_dialogue_diarize", "moss-transcribe-diarize-0.9B")
EXCLUDE = Path("/workspace/data/datasets/emilia2_zh_dialogue_diarize/versions/moss-transcribe-diarize-0.9B/"
               "excluded-records-4plus-merge-split-cap-0.80-20261028.txt")
TS_SHARE = 20.0
TS_SECTIONS = ("interference_sources", "rir_sources", "floor_sources", "nonspeech_sources")
SPLIT_TAG = "diar-restart-20261028"


def split_real_windows(source, catalog, roots):
    """A ts_real_train source with online mixing as (present with mixing, absent as recorded), weights by count."""
    entry = catalog[(source["dataset_id"], source["version"])]
    present, absent = set(), set()
    for name in entry["splits"][source["split"]]["artifacts"]["records"]:
        artifact = next(a for a in entry["artifacts"] if a["name"] == name)
        with gzip.open(Path(roots[artifact["root_alias"]]) / artifact["relative_path"], "rt", encoding="utf-8") as stream:
            for row in map(json.loads, stream):
                kept = row.get("task") == "ts_asr" and ((row.get("target") or "").strip() or (row.get("metadata") or {}).get("scene"))
                (present if kept else absent).add(row["id"])
    excluded = set(Path(source["exclude_records"]).read_text().split()) if source.get("exclude_records") else set()
    n_present, n_absent = len(present - excluded), len(absent - excluded)
    directory = REGISTRY.parent / "datasets" / source["dataset_id"] / "versions" / source["version"]
    out = []
    for kind, drop, count, mixing in (("present", absent, n_present, True), ("absent", present, n_absent, False)):
        path = directory / f"excluded-records-{SPLIT_TAG}-{source['split']}-keep-{kind}.txt"
        text = "".join(f"{i}\n" for i in sorted(excluded | drop))
        if path.exists():
            assert path.read_text() == text, path
        else:
            path.write_text(text)
            path.chmod(0o664)
        part = {k: v for k, v in source.items() if mixing or k != "online_mix"}
        part.update(exclude_records=str(path), weight=source["weight"] * count / (n_present + n_absent))
        out.append(part)
    return out


def is_ts(source):
    return source.get("task") == "ts_asr" or "online_mix" in source or source["version"].endswith("-utts") \
        and source["dataset_id"] in {"emilia2_mix", "alimeeting_sdm", "aishell4", "ami_sdm", "magicdata_ramc"}


def main():
    shared = RUN / "shared"
    catalog_dir = shared / "catalog"
    # The catalog is frozen once written: a rerun reads the copy, not the live registry.
    fresh = not catalog_dir.exists()
    if fresh:
        catalog_dir.mkdir()
        for path in sorted((REGISTRY / "catalog").glob("*.yaml")):
            shutil.copy2(path, catalog_dir / path.name)
    entries = {}
    for path in sorted(catalog_dir.glob("*.yaml")):
        for entry in yaml.safe_load(path.read_text()) or []:
            if isinstance(entry, dict) and "dataset_id" in entry:
                entries[(entry["dataset_id"], entry["version"])] = entry
    registry_keys = set(entries)
    control = yaml.safe_load((CONTROL / "train-data.yaml").read_text())
    frozen = [json.loads(line) for line in open(control["catalog"]) if line.strip()]
    only = [e for e in frozen if (e["dataset_id"], e["version"]) not in registry_keys]
    if fresh:
        (catalog_dir / "zz-frozen-only-zh-nosimilar-control-002.jsonl").write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in only))
        roots = json.loads((REGISTRY / "roots.json").read_text())
        roots.update(json.loads(Path(control["roots"]).read_text()))
        (shared / "roots.json").write_text(json.dumps(roots, ensure_ascii=False, indent=2) + "\n")
    roots = json.loads((shared / "roots.json").read_text())

    def base():
        data = yaml.safe_load((CONTROL / "train-data.yaml").read_text())
        data["catalog"], data["roots"] = str(catalog_dir), str(shared / "roots.json")
        hits = [s for s in data["train"] if (s["dataset_id"], s["version"]) == EMILIA2]
        assert hits, "no Emilia2 dialogue source"
        for s in hits:
            s["exclude_records"] = str(EXCLUDE)
        return data

    a = base()
    assert abs(sum(s["weight"] for s in a["train"]) - 100) < 1e-6
    b = base()
    for s in b["train"]:
        s["weight"] *= (100 - TS_SHARE) / 100
    ts = yaml.safe_load(TS_RECIPE.read_text())
    added = []
    for s in ts["train"]:
        if is_ts(s):
            added += split_real_windows(s, entries, roots) if s["dataset_id"] == "ts_real_train" and "online_mix" in s else [dict(s)]
    total = sum(s["weight"] for s in added)
    for s in added:
        s["weight"] *= TS_SHARE / total
    assert not {(s["dataset_id"], s["version"]) for s in added} & {(s["dataset_id"], s["version"]) for s in b["train"]}
    b["train"] += added
    for key in TS_SECTIONS:
        assert key not in b, key
        b[key] = ts[key]
    b["record_index"] = "lance"
    assert abs(sum(s["weight"] for s in b["train"]) - 100) < 1e-6
    for name, data, head in (
        ("a", a, "# A 组（对照）：zh-nosimilar-control-acp-20261016 prepare 002 的冻结配方，只把 Emilia2 对话的排除名单换成 0.80 拆人上限版本；"
                 "catalog 与 roots 见 scripts/make_recipes.py。由 scripts/make_recipes.py 生成。\n"),
        ("b", b, "# B 组：A 组全部数据源缩到 80，加 TS-robust 配方 v7set23 的目标说话人任务数据源共 20（相对权重、在线混合参数和排除名单不变）"
                 "与 TS 在线混合用的干扰人 / 冲激响应 / 底噪 / 非语音源；批处理、回放、增强、噪声源与 A 组相同。由 scripts/make_recipes.py 生成。\n"),
    ):
        (RUN / "configs" / f"train-data-{name}.yaml").write_text(head + yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    print(json.dumps({"registry_entries": len(registry_keys), "frozen_only_entries": len(only),
                      "a_sources": len(a["train"]), "b_sources": len(b["train"]), "ts_sources": len(added),
                      "ts_source_total_in_v7set23": round(total, 3)}))


if __name__ == "__main__":
    main()
