#!/usr/bin/env python3
"""Move about 25 hours of 6-speaker dialog from the Emilia training records into eval.

The moved audio is whole recordings: every training record from that recording is
deleted, including dialog files that are not 6+ and any long files. Five-speaker
clips stay out of this bucket. Each language is topped up so dev and test both
keep a no-enrollment split and an enrollment split.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

TRAIN = Path("/ai_sds_wuzz/DATA_ASR/Derived/target-sot-emilia-20261004")
EVAL = Path("/ai_sds_wuzz/DATA_ASR/Derived/target-sot-supervision-20261004")
MOVED_HOURS = 25.0
K0_SHARE = 0.40
DEV_SHARE = 10.0 / 35.0


def sha1(text):
    return hashlib.sha1(text.encode()).hexdigest()


def rid_of(cut_id):
    return cut_id.split(":")[1]


def load_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def existing_six_hours():
    summary = json.loads((EVAL / "summary.json").read_text())
    hours = {"zh": 0.0, "en": 0.0}
    for component in summary["components"]:
        name = component["name"]
        if "6plus_plain" not in name:
            continue
        hours["zh" if "_zh_" in name else "en"] += component["hours"]
    return hours


def scan_training():
    per_rid = {}
    for path in sorted((TRAIN / "records").glob("k*_*.jsonl.gz")):
        kind = "long" if "_long_" in path.name else "dialog"
        for record in load_jsonl(path):
            ref = record["audio_slots"][0]["ref"]
            rid = rid_of(ref["cut_id"])
            row = per_rid.setdefault(rid, {
                "langs": set(),
                "six_k0": 0.0,
                "six_kpos": 0.0,
                "other": 0.0,
            })
            hours = ref["duration"] / 3600.0
            if kind == "dialog" and record["labels"]["speaker_count"] >= 6 and ref["duration"] <= 600.0:
                row["langs"].add(record["language"])
                key = "six_k0" if record["labels"]["recipe_k"] == 0 else "six_kpos"
                row[key] += hours
            else:
                row["other"] += hours
    return per_rid


def targets_for(added):
    k0 = added * K0_SHARE
    kpos = added - k0
    return {
        "k0": {"dev": k0 * DEV_SHARE, "test": k0 * (1.0 - DEV_SHARE)},
        "kpos": {"dev": kpos * DEV_SHARE, "test": kpos * (1.0 - DEV_SHARE)},
    }


def pack(candidates, dev_target, test_target):
    chosen = {}
    filled = {"dev": 0.0, "test": 0.0}
    for cap, slack in ((0.5, 0.4), (1.0, 0.5), (2.0, 1.0), (8.0, 2.0), (1e9, 4.0)):
        pool = [
            row for row in candidates
            if row["rid"] not in chosen and row["other"] <= cap * max(row["six"], 1e-6)
        ]
        pool.sort(key=lambda row: (row["six"], sha1(row["rid"])))
        for row in pool:
            options = []
            for split, target in (("dev", dev_target), ("test", test_target)):
                need = target - filled[split]
                if need > 0.02 and row["six"] <= need + slack:
                    options.append((need, split))
            if not options:
                continue
            split = max(options, key=lambda item: item[0])[1]
            chosen[row["rid"]] = split
            filled[split] += row["six"]
        if filled["dev"] >= dev_target - 0.15 and filled["test"] >= test_target - 0.15:
            break
    return chosen, filled


def select(per_rid, existing):
    gap = existing["en"] - existing["zh"]
    extra = (MOVED_HOURS - gap) / 2.0
    added = {"zh": gap + extra, "en": extra}
    if min(added.values()) <= 0 or abs(sum(added.values()) - MOVED_HOURS) > 1e-6:
        raise SystemExit(f"bad allocation {added}")
    plan = {}
    chosen = {}
    for language, hours in added.items():
        goal = targets_for(hours)
        plan[language] = goal
        for kind, key in (("k0", "six_k0"), ("kpos", "six_kpos")):
            candidates = []
            for rid, row in per_rid.items():
                if row["langs"] != {language}:
                    continue
                if row[key] <= 0 or (row["six_k0"] > 0 and row["six_kpos"] > 0):
                    continue
                candidates.append({"rid": rid, "six": row[key], "other": row["other"]})
            picked, filled = pack(candidates, goal[kind]["dev"], goal[kind]["test"])
            for split, target in goal[kind].items():
                if filled[split] < target - 0.2:
                    raise SystemExit(f"short {language} {kind} {split}: {filled[split]:.2f}/{target:.2f}")
            for rid, split in picked.items():
                chosen[rid] = {"language": language, "kind": kind, "split": split}
    return added, plan, chosen


def rewrite_ref(ref, split):
    copied = dict(ref)
    copied["dataset_id"] = "target_sot_eval"
    copied["version"] = "20261004"
    copied["split"] = split
    return copied


def freeze_enrollment(record, catalog_split):
    identities = record["metadata"]["speaker_identities"]
    grouped = defaultdict(list)
    for candidate in record["metadata"].get("enrollment_candidates", []):
        grouped[candidate["speaker_id"]].append(candidate)
    frozen, kept = [], []
    for index in range(1, len(identities) + 1):
        identity = identities[f"S{index}"]
        spans = sorted(grouped.get(identity, []), key=lambda item: item["ref"]["duration"], reverse=True)[:8]
        if not spans or spans[0]["ref"]["duration"] < 1.0:
            continue
        source = spans[0]["ref"]
        length = min(5.0, source["duration"])
        for candidate in spans:
            copied = dict(candidate)
            copied["partition"] = record["metadata"]["partition"]
            copied["ref"] = rewrite_ref(candidate["ref"], catalog_split)
            kept.append(copied)
        frozen.append({
            "speaker_id": identity,
            "ref": rewrite_ref({**source, "duration": length}, catalog_split),
        })
        if len(frozen) == 5:
            break
    return kept, frozen


def to_eval(record, split_name, partition, frozen_pair):
    mixture = record["audio_slots"][0]["ref"]
    candidates, frozen = frozen_pair
    metadata = {
        "sot_output_format": "aligned_utterance_timestamps_v1",
        "partition": partition,
        "speaker_identities": record["metadata"]["speaker_identities"],
        "source_spans": [rewrite_ref(span, split_name) for span in record["metadata"]["source_spans"]],
        "speaker_bucket": "6plus",
        "moved_from": "target_sot_emilia@20261004",
        "recipe": {
            "name": "target-sot-supervision-20261004",
            "k": len(frozen),
            "reps": 1,
            "absent_probability": 0.0,
            "mode": "all",
            "min_seconds": 1.0,
            "max_seconds": 5.0,
        },
    }
    if frozen:
        metadata["enrollment_candidates"] = candidates
        metadata["fixed_enrollment"] = {"mode": "all", "enrollments": frozen}
    suffix = "-enroll" if frozen else ""
    return {
        "schema_version": "audio-record/1.0",
        "id": f"dialog-{partition}-6plus-{record['id']}{suffix}",
        "task": record["task"],
        "audio_slots": [{
            "name": "mixture",
            "purpose": "mixture",
            "ref": rewrite_ref(mixture, split_name),
        }],
        "target": record["target"],
        "language": record["language"],
        "labels": {"speaker_count": record["labels"]["speaker_count"], "recipe_k": len(frozen)},
        "hotwords": [],
        "metadata": metadata,
    }


def needed_cuts(record):
    cuts = {record["audio_slots"][0]["ref"]["cut_id"]}
    for candidate in record["metadata"].get("enrollment_candidates", []):
        cuts.add(candidate["ref"]["cut_id"])
    return cuts


def main():
    import sys
    sys.path.insert(0, "/222042021/lx/open-audio-llm-multi-target/src")
    sys.path.insert(0, "/222042021/lx/audio-data-contract/src")
    from audio_data_contract import AudioRecord
    from open_audio_llm.data.target_sot import EnrollmentConfig, eligible_candidates, parse_target_segments, sample_enrollment

    marker = EVAL / "moved-from-train.json"
    if marker.exists():
        raise SystemExit(f"already moved: {marker}")
    existing = existing_six_hours()
    per_rid = scan_training()
    added, plan, chosen = select(per_rid, existing)
    moved_hours = defaultdict(float)
    collateral = defaultdict(float)
    for rid, spec in chosen.items():
        row = per_rid[rid]
        moved_hours[spec["language"]] += row["six_k0"] if spec["kind"] == "k0" else row["six_kpos"]
        collateral[spec["language"]] += row["other"]
    print("existing", {key: round(value, 2) for key, value in existing.items()}, flush=True)
    print("move", {key: round(value, 2) for key, value in added.items()}, flush=True)
    print("packed", {key: round(value, 2) for key, value in moved_hours.items()}, "collateral",
          {key: round(value, 2) for key, value in collateral.items()}, "recordings", len(chosen), flush=True)

    outputs = defaultdict(list)
    cut_ids = set()
    removed = defaultdict(float)
    removed_speakers = defaultdict(float)
    removed_k = defaultdict(float)
    kept_stats = {}
    config = EnrollmentConfig(max_targets=5, absent_probability=0.0, mode="all", probability=1.0)

    for path in sorted((TRAIN / "records").glob("k*_*.jsonl.gz")):
        kept = []
        hours = 0.0
        group = "long" if "_long_" in path.name else "dialog"
        for record in load_jsonl(path):
            ref = record["audio_slots"][0]["ref"]
            rid = rid_of(ref["cut_id"])
            spec = chosen.get(rid)
            if spec is None:
                kept.append(record)
                hours += ref["duration"] / 3600.0
                continue
            removed[group] += ref["duration"] / 3600.0
            removed_k[record["labels"]["recipe_k"]] += ref["duration"] / 3600.0
            count = record["labels"]["speaker_count"]
            bucket = "6+" if count >= 6 else str(count)
            removed_speakers[(group, record["language"], bucket)] += ref["duration"] / 3600.0
            use = (
                group == "dialog"
                and count >= 6
                and ref["duration"] <= 600.0
                and ((spec["kind"] == "k0" and record["labels"]["recipe_k"] == 0)
                     or (spec["kind"] == "kpos" and record["labels"]["recipe_k"] > 0))
            )
            if not use:
                continue
            partition = spec["split"]
            plain_name = f"dialog_{spec['language']}_{partition}_6plus_plain"
            parse_target_segments(record["target"], ref["duration"])
            plain = to_eval(record, plain_name, partition, ([], []))
            outputs[plain_name].append(plain)
            cut_ids.update(needed_cuts(record))
            if spec["kind"] != "kpos":
                continue
            enroll_name = f"dialog_{spec['language']}_{partition}_6plus_enroll"
            record["metadata"]["partition"] = partition
            candidates, frozen = freeze_enrollment(record, enroll_name)
            if not frozen:
                raise SystemExit(f"no enrollment for {record['id']}")
            enrolled = to_eval(record, enroll_name, partition, (candidates, frozen))
            groups = eligible_candidates(AudioRecord.from_dict(enrolled), config)
            sample_enrollment(AudioRecord.from_dict(enrolled), config, enrolled["id"],
                              fixed=enrolled["metadata"]["fixed_enrollment"])
            if set(groups) != {item["speaker_id"] for item in frozen}:
                raise SystemExit(f"enrollment speakers mismatch {enrolled['id']}")
            outputs[enroll_name].append(enrolled)
            cut_ids.update(needed_cuts(enrolled))
        kept_stats[path.name] = {"records": len(kept), "hours": hours, "records_body": kept}

    copied = {}
    with gzip.open(TRAIN / "audio-index.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["cut_id"] in cut_ids:
                copied[row["cut_id"]] = row
    missing = cut_ids - set(copied)
    if missing:
        raise SystemExit(f"missing audio index rows {len(missing)}")

    for name, records in outputs.items():
        path = EVAL / "records" / f"{name}.jsonl.gz"
        write_jsonl(path, list(load_jsonl(path)) + records)
    index_path = EVAL / "audio-index.jsonl.gz"
    index_rows = list(load_jsonl(index_path))
    present = {row["cut_id"] for row in index_rows}
    for cut_id, row in sorted(copied.items()):
        if cut_id not in present:
            index_rows.append(row)
    write_jsonl(index_path, index_rows)

    for path_name, stats in kept_stats.items():
        write_jsonl(TRAIN / "records" / path_name, stats["records_body"])
        stats.pop("records_body")

    catalog = json.loads((TRAIN / "catalog.jsonl").read_text())
    for artifact in catalog["artifacts"]:
        name = artifact["name"]
        if not name.startswith("records_"):
            continue
        stats = kept_stats.get(name.removeprefix("records_") + ".jsonl.gz")
        if stats is None:
            continue
        artifact["metadata"]["record_count"] = stats["records"]
        catalog["splits"][name.removeprefix("records_")]["statistics"] = {
            "records": stats["records"],
            "hours": round(stats["hours"], 4),
            "reps": 1,
        }
    train_hours = 0.0
    train_records = 0
    for name, split in catalog["splits"].items():
        if name == "train":
            continue
        train_hours += split["statistics"]["hours"]
        train_records += split["statistics"]["records"]
    catalog["splits"]["train"]["statistics"] = {
        "records": train_records,
        "train_hours": round(train_hours, 4),
    }
    (TRAIN / "catalog.jsonl").write_text(json.dumps(catalog, ensure_ascii=False) + "\n")

    summary = json.loads((TRAIN / "summary.json").read_text())
    for (group, language, bucket), hours in removed_speakers.items():
        summary["speakers"][group][language][bucket] -= hours
    for key, hours in removed_k.items():
        summary["by_k"][str(key)] = round(summary["by_k"][str(key)] - hours, 2)
    for group, hours in removed.items():
        summary["by_group"][group] = round(summary["by_group"][group] - hours, 2)
    summary["train_hours"] = round(train_hours, 2)
    summary["moved_6plus_to_eval"] = {
        "mixture_hours": {key: round(value, 4) for key, value in moved_hours.items()},
        "deleted_hours": {key: round(value, 4) for key, value in removed.items()},
        "recordings": len(chosen),
    }
    (TRAIN / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    eval_catalog = json.loads((EVAL / "catalog.jsonl").read_text())
    eval_summary = json.loads((EVAL / "summary.json").read_text())
    component_by_name = {item["name"]: item for item in eval_summary["components"]}
    recounted = {}
    for name in outputs:
        hours = 0.0
        count = 0
        max_duration = 0.0
        for record in load_jsonl(EVAL / "records" / f"{name}.jsonl.gz"):
            duration = record["audio_slots"][0]["ref"]["duration"]
            hours += duration / 3600.0
            count += 1
            max_duration = max(max_duration, duration)
            if duration > 600.0 or record["labels"]["speaker_count"] < 6:
                raise SystemExit(f"bad eval record {record['id']}")
        recounted[name] = {"records": count, "hours": round(hours, 4), "max_duration": max_duration}
        component_by_name[name]["records"] = count
        component_by_name[name]["hours"] = round(hours, 4)
        eval_catalog["splits"][name]["statistics"] = {"records": count, "hours": round(hours, 4)}
        for artifact in eval_catalog["artifacts"]:
            if artifact["name"] == f"records_{name}":
                artifact["metadata"]["record_count"] = count
    for artifact in eval_catalog["artifacts"]:
        if artifact["name"] == "audio_index":
            artifact["metadata"]["records"] = len(index_rows)
    (EVAL / "catalog.jsonl").write_text(json.dumps(eval_catalog, ensure_ascii=False) + "\n")
    (EVAL / "summary.json").write_text(json.dumps(eval_summary, ensure_ascii=False, indent=2) + "\n")

    eval_rids = defaultdict(set)
    train_rids = set()
    for path in (TRAIN / "records").glob("k*_*.jsonl.gz"):
        for record in load_jsonl(path):
            train_rids.add(rid_of(record["audio_slots"][0]["ref"]["cut_id"]))
    for path in (EVAL / "records").glob("*.jsonl.gz"):
        partition = "dev" if "_dev_" in path.name else "test"
        for record in load_jsonl(path):
            rid = rid_of(record["metadata"]["source_spans"][0]["cut_id"])
            eval_rids[partition].add(rid)
            if rid in train_rids:
                raise SystemExit(f"training overlap {rid}")
    overlap = eval_rids["dev"] & eval_rids["test"]
    if overlap:
        raise SystemExit(f"dev/test overlap {len(overlap)}")
    report = {
        "added_target_hours": added,
        "packed_mixture_hours": dict(moved_hours),
        "collateral_hours": dict(collateral),
        "deleted_hours": dict(removed),
        "recordings": len(chosen),
        "eval_6plus": recounted,
        "train_hours": train_hours,
    }
    marker.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
