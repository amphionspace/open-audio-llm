#!/usr/bin/env python3
"""Write held-out target-SOT supervision. Audio files are not copied.

With-enrollment and without-enrollment records are separate files. An
enrollment crop is the first 5 seconds, or the whole span when it is shorter,
of one eligible single-speaker region. The crop is frozen in
``metadata.fixed_enrollment``. Absent speakers are not enrolled.

Emilia2-long dev is 20 hours and test is 50 hours, split evenly across
Chinese and English. Clips come from recordings that were not used in the
2026-09-30 training recipe, and dev and test do not share a recording.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

RECIPE_PATH = Path(__file__).with_name("build_target_sot_recipe.py")
MEETINGS_PATH = Path(
    "/222042021/mingdong/workspace/AmphionData/src/amphiondata/multispeaker/meetings.py"
)
OUT = Path("/ai_sds_wuzz/DATA_ASR/Derived/target-sot-supervision-20261001")
DS = "target_sot_eval"
VER = "20261001"
AISHELL4_TEST_OUT = Path("/ai_sds_wuzz/DATA_ASR/Derived/target-sot-supervision-aishell4-test-20261006")
DURATIONS = (30, 120, 300, 600)
# 20h dev and 50h test, half Chinese and half English.
LONG_HOURS = {"dev": {"zh": 10.0, "en": 10.0}, "test": {"zh": 25.0, "en": 25.0}}
TRAIN_LONG = {"zh": {"k0": 1181.0, "k1": 722.0}, "en": {"k0": 819.0, "k1": 1364.0}}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recipe = load_module("target_sot_recipe", RECIPE_PATH)
meetings = load_module("meeting_windows", MEETINGS_PATH)


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def ref(split, cut_id, start, duration):
    return {
        "dataset_id": DS,
        "version": VER,
        "split": split,
        "cut_id": cut_id,
        "start": float(start),
        "duration": float(duration),
    }


def candidate(partition, speaker_id, cut_id, start, duration, split):
    return {
        "speaker_id": speaker_id,
        "partition": partition,
        "single_speaker": True,
        "ref": ref(split, cut_id, start, duration),
    }


def appearance(identities):
    return [identities[f"S{i + 1}"] for i in range(len(identities))]


def _blocked(start, end, blocked):
    return any(max(start, lo) < min(end, hi) for lo, hi in blocked)


def freeze_enrollment(catalog_split, partition, cut_id, identities, solos, blocked=()):
    """One prefix crop per present speaker, at most five, in first-speech order."""
    candidates = []
    frozen = []
    for identity in appearance(identities):
        spans = solos.get(identity)
        if not spans:
            continue
        stored = [
            (start, end) for start, end in recipe.cap_spans(spans, f"{catalog_split}:{identity}")
            if not _blocked(start, end, blocked)
        ]
        if not stored:
            continue
        start, end = max(stored, key=lambda item: item[1] - item[0])
        length = min(5.0, end - start)
        if length < 1.0 or _blocked(start, start + length, blocked):
            continue
        for span_start, span_end in stored:
            candidates.append(candidate(
                partition, identity, cut_id, span_start, span_end - span_start, catalog_split,
            ))
        frozen.append({
            "speaker_id": identity,
            "ref": ref(catalog_split, cut_id, start, length),
        })
        if len(frozen) == 5:
            break
    return candidates, frozen


def record_dict(record_id, catalog_split, partition, language, duration, mixture_cut, mixture_start,
                target, identities, spans, k, candidates, frozen, extra=None):
    if partition not in {"train", "dev", "test"}:
        raise RuntimeError(f"unexpected partition {partition}")
    metadata = {
        "sot_output_format": "aligned_utterance_timestamps_v1",
        "partition": partition,
        "speaker_identities": identities,
        "source_spans": spans,
        "recipe": {
            "name": f"target-sot-supervision-{VER}",
            "k": k,
            "reps": 1,
            "absent_probability": 0.0,
            "mode": "all",
            "min_seconds": 1.0,
            "max_seconds": 5.0,
        },
    }
    if k:
        metadata["enrollment_candidates"] = candidates
        metadata["fixed_enrollment"] = {"mode": "all", "enrollments": frozen}
    if extra:
        metadata.update(extra)
    return {
        "schema_version": "audio-record/1.0",
        "id": record_id,
        "task": "speaker_attributed_asr",
        "audio_slots": [{
            "name": "mixture",
            "purpose": "mixture",
            "ref": ref(catalog_split, mixture_cut, mixture_start, duration),
        }],
        "target": target,
        "language": language,
        "labels": {"speaker_count": len(identities), "recipe_k": k},
        "hotwords": [],
        "metadata": metadata,
    }


class Sink:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handles = {}
        self.components = {}

    def write(self, name, record, hours, *, partition, language, group, enrolled):
        handle = self._handles.get(name)
        if handle is None:
            path = self.directory / f"{name}.jsonl.gz"
            handle = gzip.open(path, "wt", encoding="utf-8", compresslevel=1)
            self._handles[name] = handle
            self.components[name] = {
                "name": name,
                "path": str(path),
                "partition": partition,
                "language": language,
                "group": group,
                "enrolled": enrolled,
                "records": 0,
                "hours": 0.0,
                "k": defaultdict(int),
            }
        spec = self.components[name]
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        spec["records"] += 1
        spec["hours"] += hours
        spec["k"][record["metadata"]["recipe"]["k"]] += 1

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def adopt_index(index, cut_id, row):
    index.add(
        cut_id, row["root_alias"], row["relative_path"], row["sample_rate"],
        row["channels"], row["duration"], row["num_frames"],
    )


def turns_of(raw_turns, prefix):
    return [{
        "speaker": f"{prefix}:{turn['source_speaker']}",
        "start": float(turn["start"]),
        "end": float(turn["end"]),
        "text": turn.get("text") or "",
    } for turn in raw_turns]


def emit_meeting_window(sink, index, split_name, partition, language, group, kind,
                        window, present, absent):
    del absent  # Evaluation does not enroll speakers who are outside the mixture.
    prefix = f"{kind}:{window['meeting_id']}"
    target, identities = recipe.render_turns(turns_of(window["turns"], prefix), window["duration"])
    if not target:
        return
    cut_id = f"{kind}:{window['meeting_id']}"
    if window.get("index_row") is not None:
        adopt_index(index, cut_id, window["index_row"])
    else:
        index.add_path(cut_id, window["source_audio"])
    mixture_cut = cut_id
    mixture_start = window["origin"]
    if window.get("window_audio"):
        mixture_cut = f"win:{kind}:{window['id']}"
        index.add_path(mixture_cut, window["window_audio"])
        mixture_start = 0.0
    hours = window["duration"] / 3600
    extra = {"meeting_id": window["meeting_id"], "target_seconds": window["duration_label"]}
    if window.get("unsupported_reason"):
        extra["unsupported_reason"] = window["unsupported_reason"]
    base_id = f"{kind}:{window['id']}"
    plain_split = f"{split_name}_{window['duration_label']}s_plain"
    enroll_split = f"{split_name}_{window['duration_label']}s_enroll"
    sink.write(
        plain_split,
        record_dict(
            f"{base_id}:k0", plain_split, partition, language, window["duration"], mixture_cut,
            mixture_start, target, identities,
            [ref(plain_split, cut_id, window["origin"], window["duration"])],
            0, [], [], extra,
        ),
        hours, partition=partition, language=language, group=group, enrolled=False,
    )
    solos = {}
    for speaker, spans in present.items():
        identity = f"{prefix}:{speaker}"
        if identity in identities.values():
            solos[identity] = spans
    blocked = [(window["origin"], window["origin"] + window["duration"])]
    candidates, frozen = freeze_enrollment(
        enroll_split, partition, cut_id, identities, solos, blocked,
    )
    if not frozen:
        return
    sink.write(
        enroll_split,
        record_dict(
            f"{base_id}:k{len(frozen)}", enroll_split, partition, language, window["duration"], mixture_cut,
            mixture_start, target, identities,
            [ref(enroll_split, cut_id, window["origin"], window["duration"])],
            len(frozen), candidates, frozen, extra,
        ),
        hours, partition=partition, language=language, group=group, enrolled=True,
    )


def meeting_groups(windows):
    grouped = defaultdict(list)
    for window in windows:
        grouped[window["meeting_id"]].append(window)
    return grouped


def emit_meetings(sink, index, split_name, partition, language, group, kind, windows):
    grouped = meeting_groups(windows)
    print(f"{split_name} meetings {len(grouped)} windows {len(windows)}", flush=True)
    for meeting_id, rows in grouped.items():
        merged, blockers = recipe.meeting_blocks(rows)
        for window in rows:
            present, absent = recipe.classify_window(window, merged, blockers)
            emit_meeting_window(
                sink, index, split_name, partition, language, group, kind, window, present, absent,
            )


def load_ali_windows(split):
    base = recipe.DATA_ROOT / "Derived/meeting-sot/meeting-long-v1-20260920"
    index_rows = {
        row["cut_id"]: row
        for row in read_jsonl(base / f"alimeeting_sdm-{split}-audio-index.jsonl.gz")
    }
    windows = []
    for dur in DURATIONS:
        for item in read_jsonl(base / f"alimeeting_sdm-{split}_{dur}s.jsonl.gz"):
            slot = item["audio_slots"][0]["ref"]
            windows.append({
                "id": item["id"],
                "meeting_id": item["metadata"]["meeting_id"],
                "origin": float(slot["start"] or 0.0),
                "duration": float(slot["duration"]),
                "duration_label": dur,
                "turns": item["metadata"].get("turns") or [],
                "unsupported_reason": item["metadata"].get("unsupported_reason"),
                "index_row": index_rows[slot["cut_id"]],
            })
    return windows


def load_ali_test_windows():
    base = recipe.DATA_ROOT / "LHOTSE/AliMeeting/data/manifests"
    recordings = {row["id"]: row for row in read_jsonl(base / "alimeeting_sdm_recordings_test.jsonl.gz")}
    by_recording = defaultdict(list)
    for row in read_jsonl(base / "alimeeting_sdm_supervisions_test.jsonl.gz"):
        by_recording[row["recording_id"]].append(row)
    windows = []
    for meeting_id, recording in sorted(recordings.items()):
        rows = by_recording[meeting_id]
        invalid = [
            row["id"] for row in rows
            if not row.get("speaker") or row["duration"] <= 0
            or not 0 <= round(row["start"] * meetings.RATE)
            < round((row["start"] + row["duration"]) * meetings.RATE) <= recording["num_samples"]
        ]
        if invalid:
            raise RuntimeError(f"invalid AliMeeting test supervision: {meeting_id} {invalid[:3]}")
        source = recording["sources"][0]["source"]
        for dur in DURATIONS:
            for number, (left, right) in enumerate(
                meetings.complete_windows(recording["num_samples"], rows, dur)
            ):
                selected = [row for row in rows if left <= round(row["start"] * meetings.RATE) < right]
                _target, turns, _speakers = meetings.render_target(selected, left / meetings.RATE)
                if not any(turn["text"] for turn in turns):
                    continue
                windows.append({
                    "id": f"alimeeting_sdm-test_{dur}s-{meeting_id}-{number:05d}",
                    "meeting_id": meeting_id,
                    "origin": left / meetings.RATE,
                    "duration": (right - left) / meetings.RATE,
                    "duration_label": dur,
                    "turns": turns,
                    "source_audio": source,
                    "unsupported_reason": "indivisible_overlap_exceeds_audio_budget"
                    if (right - left) / meetings.RATE > 650 else None,
                })
    return windows


def load_aishell4_test_windows():
    """Official AISHELL-4 test, cut like AliMeeting test from recordings and supervisions."""
    base = recipe.DATA_ROOT / "LHOTSE/data_aishell4/data/manifests"
    recordings = {row["id"]: row for row in read_jsonl(base / "aishell4_recordings_test.jsonl.gz")}
    by_recording = defaultdict(list)
    for row in read_jsonl(base / "aishell4_supervisions_test.jsonl.gz"):
        item = dict(row)
        item["text"] = item.get("text") or ""
        by_recording[item["recording_id"]].append(item)
    windows = []
    for meeting_id, recording in sorted(recordings.items()):
        rows = by_recording[meeting_id]
        invalid = [
            row["id"] for row in rows
            if not row.get("speaker") or row["duration"] <= 0
            or not 0 <= round(row["start"] * meetings.RATE)
            < round((row["start"] + row["duration"]) * meetings.RATE) <= recording["num_samples"]
        ]
        if invalid:
            raise RuntimeError(f"invalid AISHELL-4 test supervision: {meeting_id} {invalid[:3]}")
        source = recording["sources"][0]["source"]
        for dur in DURATIONS:
            for number, (left, right) in enumerate(
                meetings.complete_windows(recording["num_samples"], rows, dur)
            ):
                selected = [row for row in rows if left <= round(row["start"] * meetings.RATE) < right]
                _target, turns, _speakers = meetings.render_target(selected, left / meetings.RATE)
                if not any(turn["text"] for turn in turns):
                    continue
                windows.append({
                    "id": f"aishell4-test_{dur}s-{meeting_id}-{number:05d}",
                    "meeting_id": meeting_id,
                    "origin": left / meetings.RATE,
                    "duration": (right - left) / meetings.RATE,
                    "duration_label": dur,
                    "turns": turns,
                    "source_audio": source,
                    "unsupported_reason": "indivisible_overlap_exceeds_audio_budget"
                    if (right - left) / meetings.RATE > 650 else None,
                })
    return windows


def load_ami_test_windows():
    root = recipe.DATA_ROOT / "ami_full/sot"
    windows = []
    for dur in DURATIONS:
        for item in read_jsonl(root / "supervision" / f"test_{dur}s.jsonl"):
            audio = item["audio"]
            audio_path = audio if str(audio).startswith("/") else str(root / audio)
            windows.append({
                "id": item["id"],
                "meeting_id": item["meeting_id"],
                "origin": float(item["source_start"]),
                "duration": float(item["duration"]),
                "duration_label": dur,
                "turns": item.get("turns") or [],
                "window_audio": audio_path,
                "source_audio": item["source_audio"],
            })
    return windows


def build_meetings(sink, index):
    for split in ("train", "dev"):
        emit_meetings(
            sink, index, f"alimeeting_{split}", split, "zh", "meeting", "alimeeting",
            load_ali_windows(split),
        )
    emit_meetings(
        sink, index, "alimeeting_test", "test", "zh", "meeting", "alimeeting",
        load_ali_test_windows(),
    )
    emit_meetings(
        sink, index, "ami_test", "test", "en", "meeting", "ami",
        load_ami_test_windows(),
    )
    emit_meetings(
        sink, index, "aishell4_test", "test", "zh", "meeting", "aishell4",
        load_aishell4_test_windows(),
    )


def build_dialog(sink, index):
    for language in ("zh", "en"):
        for split in ("dev", "test"):
            parents = {}
            parent_path = recipe.DATA_ROOT / f"emilia2/dialog/supervision/emilia2_dialog_{language}_{split}.jsonl"
            for item in read_jsonl(parent_path):
                speech = defaultdict(list)
                for turn in item["turns"]:
                    speech[turn["speaker"]].append((float(turn["start"]), float(turn["end"])))
                parents[item["session_id"]] = {
                    "audio": item["audio"],
                    "speech": {speaker: recipe.merge(intervals) for speaker, intervals in speech.items()},
                }
            name = f"dialog_{language}_{split}"
            kept = enrolled = 0
            sot_path = recipe.DATA_ROOT / f"emilia2/dialog/sot/supervision/emilia2_dialog_{language}_{split}.jsonl"
            for item in read_jsonl(sot_path):
                duration = float(item["duration"])
                if "crop_start" not in item:
                    # The sot wav and the original dialog wav are different paths.
                    cut_id = f"dlguncut:{item['session_id']}"
                    index.add_path(cut_id, item["audio"])
                    turns = [{
                        "speaker": f"emilia2_dialog:{item['session_id']}:{turn['speaker']}",
                        "start": turn["start"],
                        "end": turn["end"],
                        "text": turn.get("text") or "",
                    } for turn in item["turns"]]
                    target, identities = recipe.render_turns(turns, duration)
                    if not target:
                        continue
                    sink.write(
                        f"{name}_plain",
                        record_dict(
                            f"dialog:{item['session_id']}:k0", f"{name}_plain", split, language, duration,
                            cut_id, 0.0, target, identities,
                            [ref(f"{name}_plain", cut_id, 0.0, duration)], 0, [], [],
                        ),
                        duration / 3600, partition=split, language=language, group="dialog", enrolled=False,
                    )
                    kept += 1
                    continue
                parent = parents.get(item["source_session_id"])
                if parent is None:
                    raise RuntimeError(f"dialog crop has no parent in {split}: {item['session_id']}")
                lo, hi = float(item["crop_start"]), float(item["crop_end"])
                present = recipe.dialog_candidates(parent, lo, hi, True)
                mix_id = f"dlgmix:{item['session_id']}"
                parent_cut = f"dlg:{item['source_session_id']}"
                index.add_path(mix_id, item["audio"])
                index.add_path(parent_cut, parent["audio"])
                turns = [{
                    "speaker": f"emilia2_dialog:{item['source_session_id']}:{turn['speaker']}",
                    "start": turn["start"],
                    "end": turn["end"],
                    "text": turn.get("text") or "",
                } for turn in item["turns"]]
                target, identities = recipe.render_turns(turns, duration)
                if not target:
                    continue
                solos = {
                    f"emilia2_dialog:{item['source_session_id']}:{speaker}": spans
                    for speaker, spans in present.items()
                    if f"emilia2_dialog:{item['source_session_id']}:{speaker}" in identities.values()
                }
                plain_split = f"{name}_plain"
                enroll_split = f"{name}_enroll"
                candidates, frozen = freeze_enrollment(
                    enroll_split, split, parent_cut, identities, solos, [(lo, hi)],
                )
                if not frozen:
                    sink.write(
                        plain_split,
                        record_dict(
                            f"dialog:{item['session_id']}:k0", plain_split, split, language, duration,
                            mix_id, 0.0, target, identities,
                            [ref(plain_split, parent_cut, lo, hi - lo)], 0, [], [],
                        ),
                        duration / 3600, partition=split, language=language, group="dialog", enrolled=False,
                    )
                    continue
                sink.write(
                    enroll_split,
                    record_dict(
                        f"dialog:{item['session_id']}:k{len(frozen)}", enroll_split, split, language, duration,
                        mix_id, 0.0, target, identities,
                        [ref(enroll_split, parent_cut, lo, hi - lo)],
                        len(frozen), candidates, frozen,
                    ),
                    duration / 3600, partition=split, language=language, group="dialog", enrolled=True,
                )
                enrolled += 1
            print(f"{name} plain_uncut_or_unenrollable {kept} enroll {enrolled}", flush=True)


def long_rows(language):
    path = recipe.DATA_ROOT / f"emilia2/long/sot/supervision/emilia2_long_{language}.jsonl"
    groups = defaultdict(list)
    for number, item in enumerate(read_jsonl(path), 1):
        groups[item["source_id"]].append(item)
        if number % 200000 == 0:
            print(f"long {language} loaded {number}", flush=True)
    whole, cuts = [], []
    for source_id, clips in groups.items():
        identity = f"emilia2_long:{source_id}"
        for clip in clips:
            turns = [{
                "speaker": identity,
                "start": member["start"],
                "end": member["end"],
                "text": member.get("text") or "",
            } for member in clip["members"]]
            target, identities = recipe.render_turns(turns, float(clip["duration"]))
            if not target:
                continue
            row = {
                "id": clip["id"],
                "hours": float(clip["duration"]) / 3600,
                "cut": bool(clip.get("cut")),
                "abs_start": float(clip["abs_start"]),
                "abs_end": float(clip["abs_end"]),
                "duration": float(clip["duration"]),
                "audio": clip["audio"],
                "target": target,
                "identities": identities,
                "identity": identity,
                "source_id": source_id,
            }
            (cuts if row["cut"] else whole).append(row)
    chosen0, _got0 = recipe.select_hours(whole, TRAIN_LONG[language]["k0"])
    compact = defaultdict(list)
    for source_id, clips in groups.items():
        for clip in clips:
            compact[source_id].append((
                clip["id"], bool(clip.get("cut")), float(clip["abs_start"]),
                float(clip["abs_end"]), float(clip["duration"]), clip["audio"],
            ))
    del groups
    enrollable = []
    for row in cuts:
        siblings = []
        for other_id, _other_cut, abs_start, abs_end, duration, audio in compact[row["source_id"]]:
            if other_id == row["id"]:
                continue
            pieces = []
            left_end = min(abs_end, row["abs_start"])
            if left_end - abs_start >= 1.0:
                pieces.append((abs_start, left_end))
            right_start = max(abs_start, row["abs_end"])
            if abs_end - right_start >= 1.0:
                pieces.append((right_start, abs_end))
            for piece_start, piece_end in pieces:
                local_start = max(0.0, piece_start - abs_start)
                local_end = min(duration, piece_end - abs_start)
                if local_end - local_start < 1.0:
                    continue
                siblings.append((f"long:{other_id}", local_start, local_end - local_start, audio))
        if not siblings:
            continue
        siblings.sort(key=lambda item: item[2], reverse=True)
        row["siblings"] = siblings[:8]
        enrollable.append(row)
    chosen1, _got1 = recipe.select_hours(enrollable, TRAIN_LONG[language]["k1"])
    used = {row["source_id"] for row in chosen0} | {row["source_id"] for row in chosen1}
    free = [row for row in enrollable if row["source_id"] not in used]
    print(f"long {language} free enrollable {len(free)} {sum(row['hours'] for row in free):.1f}h", flush=True)
    return free


def take_one_per_source(rows, target, banned):
    chosen, hours, sources = [], 0.0, set()
    for row in sorted(rows, key=lambda item: recipe.sha1(item["id"])):
        if hours >= target:
            break
        if row["source_id"] in banned or row["source_id"] in sources:
            continue
        chosen.append(row)
        sources.add(row["source_id"])
        hours += row["hours"]
    return chosen, hours, sources


def build_long(sink, index):
    for language in ("zh", "en"):
        free = long_rows(language)
        banned = set()
        for split in ("dev", "test"):
            chosen, got, sources = take_one_per_source(free, LONG_HOURS[split][language], banned)
            banned |= sources
            if got + 1e-6 < LONG_HOURS[split][language]:
                raise RuntimeError(f"long {language} {split} only has {got:.2f}h")
            print(f"long {language} {split} {got:.2f}h clips {len(chosen)}", flush=True)
            plain_split = f"long_{language}_{split}_plain"
            enroll_split = f"long_{language}_{split}_enroll"
            for row in chosen:
                cut_id = f"long:{row['id']}"
                index.add_path(cut_id, row["audio"])
                for sibling_id, _start, _duration, audio in row["siblings"]:
                    index.add_path(sibling_id, audio)
                sink.write(
                    plain_split,
                    record_dict(
                        f"long:{row['id']}:k0", plain_split, split, language, row["duration"], cut_id, 0.0,
                        row["target"], row["identities"],
                        [ref(plain_split, cut_id, 0.0, row["duration"])], 0, [], [],
                    ),
                    row["hours"], partition=split, language=language, group="long", enrolled=False,
                )
                # The longest outside sibling is first. Its prefix is the frozen crop.
                stored = row["siblings"][:8]
                sibling_id, start, duration, _audio = stored[0]
                length = min(5.0, duration)
                candidates = [
                    candidate(split, row["identity"], sib_id, sib_start, sib_duration, enroll_split)
                    for sib_id, sib_start, sib_duration, _audio in stored
                ]
                frozen = [{
                    "speaker_id": row["identity"],
                    "ref": ref(enroll_split, sibling_id, start, length),
                }]
                sink.write(
                    enroll_split,
                    record_dict(
                        f"long:{row['id']}:k1", enroll_split, split, language, row["duration"], cut_id, 0.0,
                        row["target"], row["identities"],
                        [ref(enroll_split, cut_id, 0.0, row["duration"])], 1, candidates, frozen,
                    ),
                    row["hours"], partition=split, language=language, group="long", enrolled=True,
                )


def write_catalog(sink, index, *, notes=None, languages=None, include_long_hours=True):
    index_path = OUT / "audio-index.jsonl.gz"
    index.probe()
    with gzip.open(index_path, "wt", encoding="utf-8", compresslevel=1) as stream:
        for cut_id in sorted(index.rows):
            stream.write(json.dumps(index.rows[cut_id], ensure_ascii=False, separators=(",", ":")) + "\n")
    artifacts = [{
        "name": "audio_index",
        "kind": "audio-index",
        "root_alias": "legacy_asr",
        "relative_path": index_path.resolve().relative_to(recipe.DATA_ROOT).as_posix(),
        "metadata": {"records": len(index.rows)},
    }]
    splits = {}
    components = []
    for name, spec in sorted(sink.components.items()):
        path = Path(spec["path"])
        artifact_name = f"records_{name}"
        artifacts.append({
            "name": artifact_name,
            "kind": "audio-records",
            "root_alias": "legacy_asr",
            "relative_path": path.resolve().relative_to(recipe.DATA_ROOT).as_posix(),
            "metadata": {"record_count": spec["records"]},
        })
        splits[name] = {
            "records_artifact": artifact_name,
            "audio_index_artifact": "audio_index",
            "statistics": {"records": spec["records"], "duration_hours": round(spec["hours"], 4)},
        }
        components.append({
            **{key: spec[key] for key in ("name", "partition", "language", "group", "enrolled", "records")},
            "hours": round(spec["hours"], 4),
            "k": {str(k): spec["k"][k] for k in sorted(spec["k"])},
        })
    catalog = {
        "schema_version": "dataset-catalog/1.0",
        "dataset_id": DS,
        "version": VER,
        "languages": ["zh", "en"] if languages is None else languages,
        "tasks": ["speaker_attributed_asr"],
        "aliases": [],
        "artifacts": artifacts,
        "splits": splits,
        "provenance": {
            "description": "Held-out supervision. With-enrollment and without-enrollment files are separate. Audio is referenced, not copied.",
        },
    }
    (OUT / "catalog.jsonl").write_text(json.dumps(catalog, ensure_ascii=False) + "\n")
    summary = {
        "dataset_id": DS,
        "version": VER,
        "catalog": str(OUT / "catalog.jsonl"),
        "roots": str(recipe.SYN_ROOT / "roots.json"),
        **({"long_hours": LONG_HOURS} if include_long_hours else {}),
        "notes": [
            "AliMeeting train and dev reuse the published 30/120/300/600 second windows. Test windows are cut with the same boundary rule and point at the original far-field wav.",
            "AMI-SDM test uses the existing sot windows. Four durations cover the same meetings, so their hours are not additive.",
            "Dialog crops with an outside enrollment span are only in the enroll files. Uncut sessions, and crops without one, are only in the plain files.",
            "Meeting windows are in the plain file, and windows with an outside enrollment span are also in the enroll file.",
            "Emilia2-long dev is 10 Chinese hours plus 10 English hours. Test is 25 plus 25. Each recording contributes one clip, and none of these recordings were used by the training recipe.",
            "The long plain file is the same clips as the enroll file, without enrollment. Frozen enrollment is a prefix of at most 5 seconds.",
        ] if notes is None else notes,
        "components": components,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([
        {key: item[key] for key in ("name", "records", "hours", "k")} for item in components
    ], ensure_ascii=False), flush=True)


def validate(index):
    sys.path.insert(0, "/222042021/lx/open-audio-llm-multi-target/src")
    sys.path.insert(0, "/222042021/lx/audio-data-contract/src")
    from audio_data_contract import AudioRecord, DatasetSpec
    from open_audio_llm.data.target_sot import EnrollmentConfig, parse_target_segments, sample_enrollment

    DatasetSpec.from_dict(json.loads((OUT / "catalog.jsonl").read_text()))
    config = EnrollmentConfig(max_targets=5, probability=1, mode="all", absent_probability=0)
    checked = 0
    for path in sorted((OUT / "records").glob("*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                item = json.loads(line)
                record = AudioRecord.from_dict(item)
                parse_target_segments(record.target, record.audio_slots[0].ref.duration)
                if item["metadata"]["recipe"]["k"]:
                    sample_enrollment(record, config, "eval", fixed=item["metadata"]["fixed_enrollment"])
                checked += 1
        print("validated", path.name, flush=True)
    missing = [row["relative_path"] for row in index.rows.values() if row["num_frames"] is None]
    if missing:
        raise RuntimeError(f"unprobed audio: {missing[:3]}")
    print("validated records", checked, flush=True)


def release_aishell4_test():
    """Write AISHELL-4 test without touching the published 20261001 or 20261004 catalogs."""
    global OUT, VER
    if AISHELL4_TEST_OUT.exists() and any(AISHELL4_TEST_OUT.iterdir()):
        raise SystemExit(f"output already exists: {AISHELL4_TEST_OUT}")
    OUT = AISHELL4_TEST_OUT
    VER = "20261006"
    OUT.mkdir(parents=True, exist_ok=True)
    index = recipe.Index()
    sink = Sink(OUT / "records")
    try:
        emit_meetings(
            sink, index, "aishell4_test", "test", "zh", "meeting", "aishell4",
            load_aishell4_test_windows(),
        )
    finally:
        sink.close()
    write_catalog(
        sink, index, languages=["zh"], include_long_hours=False,
        notes=[
            "AISHELL-4 official test. Windows use the same boundary rule as AliMeeting test and point at the original 8-channel flac.",
            "Four durations cover the same 20 meetings, so their hours are not additive.",
            "Every kept window is in the plain file. Windows with a single-speaker span outside the mixture are also in the enroll file.",
            "Enrollment is a prefix of at most 5 seconds. Absent speakers are not enrolled.",
            "This catalog does not replace target-sot-supervision-20261001 or 20261004.",
        ],
    )
    validate(index)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"output already exists: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    index = recipe.Index()
    sink = Sink(OUT / "records")
    try:
        build_meetings(sink, index)
        build_dialog(sink, index)
        build_long(sink, index)
    finally:
        sink.close()
    write_catalog(sink, index)
    validate(index)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--aishell4-test":
        release_aishell4_test()
    else:
        main()
