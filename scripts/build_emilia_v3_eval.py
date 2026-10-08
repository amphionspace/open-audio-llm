#!/usr/bin/env python3
"""Held-out dialog and long eval from recordings absent in the 20261004 train set.

Dialog mixtures are whole files of 30 seconds to 10 minutes, split into
2-5 speakers and 6 or more speakers. Five-speaker files stay in the 2-5
group so a clip is not in both groups. Long mixtures are whole files of
10 seconds to 10 minutes. Dev and test do not share a recording, and neither
shares a recording with training.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path("/222042021/lx/open-audio-llm-multi-target/scripts/build_emilia_v3_dialog_long.py")
TRAIN = Path("/ai_sds_wuzz/DATA_ASR/Derived/target-sot-emilia-20261004")
META = Path("/ai_sds_wuzz/DATA_ASR/emilia2/segment_metainfo")
DATA_ROOT = Path("/ai_sds_wuzz/DATA_ASR")
OUT = Path("/ai_sds_wuzz/DATA_ASR/Derived/target-sot-supervision-20261004")
DS = "target_sot_eval"
VER = "20261004"

# Hours per language. 6+ uses every held-out recording, with dev taking
# the same 10:25 share as long and the 2-5 group.
DIALOG_2TO5 = {"dev": 10.0, "test": 25.0}
LONG_HOURS = {"dev": 10.0, "test": 25.0}
SIX_DEV_SHARE = 10.0 / 35.0


def load_builder():
    spec = importlib.util.spec_from_file_location("emilia_build", ROOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_keys():
    rids, files = set(), set()
    with gzip.open(TRAIN / "audio-index.jsonl.gz", "rt") as handle:
        for line in handle:
            parts = json.loads(line)["relative_path"].split("/")
            rid, kind, stem = parts[3], parts[4], parts[5][:-4]
            rids.add(rid)
            files.add((kind, rid, stem))
    return rids, files


def load_heldout(kind, train_rids, train_files):
    minimum = 30.0 if kind == "dialog" else 10.0
    rows = []
    for path in sorted((TRAIN / "work" / kind).glob("*.pkl")):
        with path.open("rb") as handle:
            cached = pickle_load(handle)
        for row in cached:
            duration = row["frames"] / row["sr"]
            if duration < minimum or duration > 600.0:
                continue
            if row["rid"] in train_rids or (kind, row["rid"], row["stem"]) in train_files:
                continue
            item = dict(row)
            item["dur"] = duration
            item["hours"] = duration / 3600.0
            item["nspk"] = len({span[0] for span in row["spans"]}) if kind == "dialog" else 1
            item["kind"] = kind
            rows.append(item)
    return rows


def pickle_load(handle):
    import pickle
    return pickle.load(handle)


def group_by_recording(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["rid"]].append(row)
    return grouped


def ordered(recordings, builder):
    return sorted(recordings, key=builder.sha1)


def take_dialog(dialog, builder):
    by_rid = group_by_recording(dialog)
    used = set()
    chosen = []

    def add(split, bucket, row):
        chosen.append((split, bucket, row))

    for language in ("zh", "en"):
        recordings = [
            rid for rid, rows in by_rid.items()
            if any(row["lang"] == language and row["nspk"] >= 6 for row in rows)
        ]
        total = sum(row["hours"] for rid in recordings for row in by_rid[rid]
                    if row["lang"] == language and row["nspk"] >= 6)
        dev_target = total * SIX_DEV_SHARE
        dev_hours = 0.0
        for rid in ordered(recordings, builder):
            if rid in used:
                continue
            split = "dev" if dev_hours < dev_target else "test"
            hours = 0.0
            for row in by_rid[rid]:
                if row["lang"] != language or row["nspk"] < 6:
                    continue
                add(split, "6plus", row)
                hours += row["hours"]
            used.add(rid)
            if split == "dev":
                dev_hours += hours

    for language in ("zh", "en"):
        filled = {"dev": 0.0, "test": 0.0}
        recordings = [
            rid for rid, rows in by_rid.items()
            if rid not in used and any(row["lang"] == language and 2 <= row["nspk"] <= 5 for row in rows)
        ]
        for rid in ordered(recordings, builder):
            if filled["dev"] >= DIALOG_2TO5["dev"] and filled["test"] >= DIALOG_2TO5["test"]:
                break
            split = "dev" if filled["dev"] < DIALOG_2TO5["dev"] else "test"
            files = [row for row in by_rid[rid] if row["lang"] == language and 2 <= row["nspk"] <= 5]
            row = min(files, key=lambda item: builder.sha1(item["id"]))
            add(split, "2to5", row)
            used.add(rid)
            filled[split] += row["hours"]
    return chosen, by_rid, used


def take_long(long_rows, banned, builder):
    by_rid = group_by_recording(long_rows)
    chosen = []
    for language in ("zh", "en"):
        filled = {"dev": 0.0, "test": 0.0}
        recordings = [
            rid for rid, rows in by_rid.items()
            if rid not in banned and any(row["lang"] == language for row in rows)
        ]
        for rid in ordered(recordings, builder):
            if filled["dev"] >= LONG_HOURS["dev"] and filled["test"] >= LONG_HOURS["test"]:
                break
            split = "dev" if filled["dev"] < LONG_HOURS["dev"] else "test"
            files = [row for row in by_rid[rid] if row["lang"] == language]
            row = min(files, key=lambda item: builder.sha1(item["id"]))
            chosen.append((split, row))
            banned.add(rid)
            filled[split] += row["hours"]
    return chosen


def load_text(kind, rows, builder):
    grouped = defaultdict(set)
    for row in rows:
        grouped[row["tar"]].add(row["member"])
    found = {}
    for tar_name, members in grouped.items():
        with tarfile.open(META / kind / tar_name, "r") as handle:
            for member in handle:
                if not member.isfile() or member.name not in members:
                    continue
                payload = json.loads(handle.extractfile(member).read())
                segments = builder.dialog_segments(payload) if kind == "dialog" else builder.long_segments(payload)
                found[(tar_name, member.name)] = segments
    return found


def ref(split, cut_id, start, duration):
    return {
        "dataset_id": DS,
        "version": VER,
        "split": split,
        "cut_id": cut_id,
        "start": float(start),
        "duration": float(duration),
    }


def cut_id(kind, row):
    return f"emilia2_{kind}:{row['rid']}:{row['stem']}"


def dialog_enrollment(row, by_rid, identities, catalog_split):
    present = set(identities.values())
    grouped = defaultdict(list)
    for other in by_rid[row["rid"]]:
        if other["stem"] == row["stem"]:
            continue
        limit = other["frames"] / other["sr"]
        for speaker, start, end in other["spans"]:
            identity = f"emilia2_dialog:{row['rid']}:{speaker}"
            if identity not in present:
                continue
            start, end = max(0.0, start), min(limit, end)
            if end - start >= 1.0:
                grouped[identity].append((cut_id("dialog", other), start, end))
    frozen, candidates = [], []
    order = [identities[f"S{i + 1}"] for i in range(len(identities))]
    for identity in order:
        spans = sorted(grouped.get(identity, []), key=lambda item: item[2] - item[1], reverse=True)[:8]
        if not spans:
            continue
        source, start, end = spans[0]
        length = min(5.0, end - start)
        if length < 1.0:
            continue
        for source_id, span_start, span_end in spans:
            candidates.append({
                "speaker_id": identity,
                "partition": None,
                "single_speaker": True,
                "ref": ref(catalog_split, source_id, span_start, span_end - span_start),
            })
        frozen.append({"speaker_id": identity, "ref": ref(catalog_split, source, start, length)})
        if len(frozen) == 5:
            break
    return candidates, frozen


def record_dict(record_id, catalog_split, partition, language, duration, mixture, target, identities,
                candidates, frozen, bucket):
    metadata = {
        "sot_output_format": "aligned_utterance_timestamps_v1",
        "partition": partition,
        "speaker_identities": identities,
        "source_spans": [ref(catalog_split, mixture, 0.0, duration)],
        "speaker_bucket": bucket,
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
        for item in candidates:
            item["partition"] = partition
        metadata["enrollment_candidates"] = candidates
        metadata["fixed_enrollment"] = {"mode": "all", "enrollments": frozen}
    return {
        "schema_version": "audio-record/1.0",
        "id": record_id,
        "task": "speaker_attributed_asr",
        "audio_slots": [{
            "name": "mixture",
            "purpose": "mixture",
            "ref": ref(catalog_split, mixture, 0.0, duration),
        }],
        "target": target,
        "language": language,
        "labels": {"speaker_count": len(identities), "recipe_k": len(frozen)},
        "hotwords": [],
        "metadata": metadata,
    }


def turns_of(kind, segments, row):
    if kind == "dialog":
        return [{
            "start": start,
            "end": end,
            "speaker": f"emilia2_dialog:{row['rid']}:{speaker}",
            "text": text,
        } for speaker, start, end, text in segments]
    identity = f"emilia2_long:{row['rid']}:{row['stem']}"
    return [{
        "start": start,
        "end": end,
        "speaker": identity,
        "text": text,
    } for start, end, text in segments]


def main():
    import sys
    sys.path.insert(0, "/222042021/lx/open-audio-llm-multi-target/src")
    sys.path.insert(0, "/222042021/lx/audio-data-contract/src")
    from audio_data_contract import AudioRecord
    from open_audio_llm.data.target_sot import EnrollmentConfig, eligible_candidates, parse_target_segments

    builder = load_builder()
    train_rids, train_files = train_keys()
    print(f"train recordings {len(train_rids)}", flush=True)
    dialog = load_heldout("dialog", train_rids, train_files)
    long_rows = load_heldout("long", train_rids, train_files)
    print(f"heldout dialog {len(dialog)} long {len(long_rows)}", flush=True)
    selected, dialog_by_rid, used = take_dialog(dialog, builder)
    long_selected = take_long(long_rows, used, builder)
    print(f"selected dialog {len(selected)} long {len(long_selected)}", flush=True)

    dialog_text = load_text("dialog", [row for _split, _bucket, row in selected], builder)
    long_text = load_text("long", [row for _split, row in long_selected], builder)
    config = EnrollmentConfig(max_targets=5, absent_probability=0.0, mode="all", probability=1.0)
    index = {}
    outputs = defaultdict(list)
    skipped = defaultdict(int)

    def add_index(kind, row):
        identity = cut_id(kind, row)
        relative = f"emilia2/raw_data/audio/{row['rid']}/{kind}/{row['stem']}.m4a"
        previous = index.get(identity)
        current = {
            "cut_id": identity,
            "root_alias": "legacy_asr",
            "relative_path": relative,
            "sample_rate": int(row["sr"]),
            "channels": int(row["ch"]),
            "duration": row["frames"] / row["sr"],
            "num_frames": int(row["frames"]),
        }
        if previous is not None and previous["relative_path"] != relative:
            raise RuntimeError(f"cut collision {identity}")
        index[identity] = current

    for split, bucket, row in selected:
        segments = dialog_text.get((row["tar"], row["member"]))
        if not segments:
            skipped["dialog_text"] += 1
            continue
        duration = row["frames"] / row["sr"]
        target, identities = builder.render_turns(turns_of("dialog", segments, row), duration)
        count = len(identities)
        actual = "6plus" if count >= 6 else "2to5" if 2 <= count <= 5 else None
        if not target or actual != bucket:
            skipped["dialog_bucket"] += 1
            continue
        parse_target_segments(target, duration)
        add_index("dialog", row)
        mixture = cut_id("dialog", row)
        plain_name = f"dialog_{row['lang']}_{split}_{bucket}_plain"
        plain = record_dict(
            f"dialog-{split}-{bucket}-{row['id']}", plain_name, split, row["lang"], duration, mixture,
            target, identities, [], [], bucket,
        )
        outputs[plain_name].append((plain, row["hours"]))
        enroll_name = f"dialog_{row['lang']}_{split}_{bucket}_enroll"
        candidates, frozen = dialog_enrollment(row, dialog_by_rid, identities, enroll_name)
        if not frozen:
            skipped["dialog_no_enroll"] += 1
            continue
        referenced = {candidate["ref"]["cut_id"] for candidate in candidates}
        for other in dialog_by_rid[row["rid"]]:
            if cut_id("dialog", other) in referenced:
                add_index("dialog", other)
        enrolled = record_dict(
            f"dialog-{split}-{bucket}-{row['id']}-enroll", enroll_name, split, row["lang"], duration, mixture,
            target, identities, candidates, frozen, bucket,
        )
        groups = eligible_candidates(AudioRecord.from_dict(enrolled), config)
        if set(groups) != {item["speaker_id"] for item in frozen}:
            raise RuntimeError(f"enrollment mismatch {enrolled['id']}")
        outputs[f"dialog_{row['lang']}_{split}_{bucket}_enroll"].append((enrolled, row["hours"]))

    for split, row in long_selected:
        segments = long_text.get((row["tar"], row["member"]))
        if not segments:
            skipped["long_text"] += 1
            continue
        duration = row["frames"] / row["sr"]
        target, identities = builder.render_turns(turns_of("long", segments, row), duration)
        if not target or len(identities) != 1:
            skipped["long_empty"] += 1
            continue
        parse_target_segments(target, duration)
        add_index("long", row)
        record = record_dict(
            f"long-{split}-{row['id']}", f"long_{row['lang']}_{split}_plain", split, row["lang"], duration,
            cut_id("long", row), target, identities, [], [], "long",
        )
        outputs[f"long_{row['lang']}_{split}_plain"].append((record, row["hours"]))

    if OUT.exists() and any(OUT.iterdir()):
        for path in OUT.rglob("*"):
            if path.is_file() and path.suffix != ".py":
                path.unlink()
    records = OUT / "records"
    records.mkdir(parents=True, exist_ok=True)
    components = []
    for name in sorted(outputs):
        path = records / f"{name}.jsonl.gz"
        hours = 0.0
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as handle:
            for record, weight in outputs[name]:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                hours += weight
        partition = "dev" if "_dev_" in name else "test"
        language = "zh" if "_zh_" in name else "en"
        components.append({
            "name": name, "records": len(outputs[name]), "hours": round(hours, 4),
            "partition": partition, "language": language, "enrolled": name.endswith("_enroll"),
        })
        print(f"{name} {hours:.2f}h n={len(outputs[name])}", flush=True)

    index_path = OUT / "audio-index.jsonl.gz"
    with gzip.open(index_path, "wt", encoding="utf-8", compresslevel=1) as handle:
        for key in sorted(index):
            handle.write(json.dumps(index[key], ensure_ascii=False, separators=(",", ":")) + "\n")
    artifacts = [{
        "name": "audio_index",
        "kind": "audio-index",
        "root_alias": "legacy_asr",
        "relative_path": index_path.resolve().relative_to(DATA_ROOT).as_posix(),
        "metadata": {"records": len(index)},
    }]
    splits = {}
    for component in components:
        artifact = f"records_{component['name']}"
        artifacts.append({
            "name": artifact,
            "kind": "audio-records",
            "root_alias": "legacy_asr",
            "relative_path": (records / f"{component['name']}.jsonl.gz").resolve().relative_to(DATA_ROOT).as_posix(),
            "metadata": {"record_count": component["records"]},
        })
        splits[component["name"]] = {
            "records_artifacts": [artifact],
            "audio_index_artifact": "audio_index",
            "statistics": {"records": component["records"], "hours": component["hours"]},
        }
    catalog = {
        "schema_version": "dataset-catalog/1.0",
        "dataset_id": DS,
        "version": VER,
        "languages": ["zh", "en"],
        "tasks": ["speaker_attributed_asr"],
        "aliases": [],
        "artifacts": artifacts,
        "splits": splits,
        "provenance": {
            "description": "Held-out Emilia2 dialog and long. Recordings do not overlap training, dev, or test.",
        },
    }
    (OUT / "catalog.jsonl").write_text(json.dumps(catalog, ensure_ascii=False) + "\n")
    (OUT / "summary.json").write_text(json.dumps({
        "components": components,
        "skipped": dict(skipped),
        "train_recordings_excluded": len(train_rids),
    }, ensure_ascii=False, indent=2) + "\n")
    print("skipped", dict(skipped), flush=True)

    eval_rids = defaultdict(set)
    for row in index.values():
        parts = row["relative_path"].split("/")
        rid, kind, stem = parts[3], parts[4], parts[5][:-4]
        if rid in train_rids or (kind, rid, stem) in train_files:
            raise SystemExit(f"training overlap {row['relative_path']}")
    for name, rows in outputs.items():
        partition = "dev" if "_dev_" in name else "test"
        for record, _hours in rows:
            if record["audio_slots"][0]["ref"]["duration"] > 600:
                raise SystemExit(f"over 10 min {record['id']}")
            rid = record["metadata"]["source_spans"][0]["cut_id"].split(":")[1]
            eval_rids[partition].add(rid)
            if name.startswith("dialog") and not name.endswith("_enroll"):
                count = record["labels"]["speaker_count"]
                if "6plus" in name and count < 6:
                    raise SystemExit(record["id"])
                if "2to5" in name and not 2 <= count <= 5:
                    raise SystemExit(record["id"])
    overlap = eval_rids["dev"] & eval_rids["test"]
    if overlap:
        raise SystemExit(f"dev/test recording overlap {len(overlap)}")
    print(f"dev recordings {len(eval_rids['dev'])} test recordings {len(eval_rids['test'])}", flush=True)


if __name__ == "__main__":
    main()
