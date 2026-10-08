#!/usr/bin/env python3
"""Materialize the 2026-09-30 enrollment recipe as audio-record supervision.

Records point at existing audio. Enrollment candidates are single-speaker
spans of at least 1 second outside the mixture; training crops 1-5 seconds
from those spans. Each record is written once. Integer repeats live in
recipe.json as ``reps``, not as duplicated lines.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf

DS = "target_sot_recipe"
VER = "20260930"
SPLIT = "train"
DATA_ROOT = Path("/ai_sds_wuzz/DATA_ASR")
SYN_ROOT = Path("/222042021/mingdong/data/sot-multispeaker/synthetic-v2-20260915")
OUT = Path("/ai_sds_wuzz/DATA_ASR/Derived/target-sot-recipe-20260930")

# Hour targets are the mixture hours inside each file, before reps.
# Meeting K>=1 and K=0 use reps=2. Dialog/synthetic repeats stay on the file.
TARGETS = {
    # k, name: (hours or None for "all", reps, language, group)
}


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def merge(intervals):
    ordered = sorted((float(a), float(b)) for a, b in intervals if b - a > 1e-4)
    out = []
    for start, end in ordered:
        if not out or start > out[-1][1] + 1e-3:
            out.append([start, end])
        else:
            out[-1][1] = max(out[-1][1], end)
    return [(a, b) for a, b in out]


def solo_outside(own, blockers, lo, hi, need=1.0):
    """Non-overlapped pieces of ``own`` that sit outside [lo, hi]."""
    pieces = []
    for start, end in own:
        cur = start
        for b0, b1 in blockers:
            if b1 <= cur:
                continue
            if b0 >= end:
                break
            if b0 - cur >= need:
                pieces.append((cur, b0))
            cur = max(cur, b1)
            if cur >= end:
                break
        if end - cur >= need:
            pieces.append((cur, end))
    outside = []
    for start, end in pieces:
        if end <= lo + 1e-6 or start >= hi - 1e-6:
            if end - start >= need:
                outside.append((start, end))
            continue
        if lo - start >= need:
            outside.append((start, lo))
        if end - hi >= need:
            outside.append((hi, end))
    return outside


def intersects(intervals, lo, hi):
    return any(not (end <= lo or start >= hi) for start, end in intervals)


def render_turns(turns, duration):
    """Timestamped SOT lines. Speaker ids in ``turns`` are stable identities."""
    cleaned = []
    for turn in turns:
        text = " ".join(str(turn.get("text") or "").split())
        if not text:
            continue
        start, end = float(turn["start"]), float(turn["end"])
        if not (math.isfinite(start) and math.isfinite(end)) or end - start < 0.01:
            continue
        cleaned.append((start, end, str(turn["speaker"]), text))
    cleaned.sort(key=lambda row: (row[0], row[2]))
    order = []
    for _, _, speaker, _ in cleaned:
        if speaker not in order:
            order.append(speaker)
    label = {speaker: f"S{i + 1}" for i, speaker in enumerate(order)}
    limit = math.floor((float(duration) + 1e-6) * 100) / 100
    last_end = {}
    lines = []
    for start, end, speaker, text in cleaned:
        name = label[speaker]
        left, right = round(start, 2), round(min(end, limit), 2)
        if right > limit:
            right = limit
        previous = last_end.get(name, -1.0)
        if left < previous:
            left = previous
        if right <= left:
            continue
        if left < 0 or right - left < 0.01:
            continue
        lines.append(f"[{name}][{left:.2f}-{right:.2f}] {text}")
        last_end[name] = right
    identities = {label[speaker]: speaker for speaker in order if any(
        line.startswith(f"[{label[speaker]}]") for line in lines
    )}
    return "\n".join(lines), identities


def ref(cut_id, start, duration, channel=None):
    item = {
        "dataset_id": DS,
        "version": VER,
        "split": SPLIT,
        "cut_id": cut_id,
        "start": float(start),
        "duration": float(duration),
    }
    if channel is not None:
        item["channel"] = int(channel)
    return item


def candidate(speaker_id, cut_id, start, duration, channel=None):
    return {
        "speaker_id": speaker_id,
        "partition": "train",
        "single_speaker": True,
        "ref": ref(cut_id, start, duration, channel),
    }


def choose(rows, count, salt):
    if not rows or count <= 0:
        return []
    ordered = sorted(rows, key=lambda row: row["key"])
    start = int(sha1(salt), 16) % len(ordered)
    return [ordered[(start + i) % len(ordered)] for i in range(min(count, len(ordered)))]


def select_hours(rows, target):
    total = sum(row["hours"] for row in rows)
    if target is None or total <= target + 1e-9:
        return rows, total
    ordered = sorted(rows, key=lambda row: sha1(row["id"]))
    chosen, acc = [], 0.0
    for row in ordered:
        if acc >= target:
            break
        chosen.append(row)
        acc += row["hours"]
    return chosen, acc


class Index:
    def __init__(self):
        self.rows = {}

    def add(self, cut_id, root_alias, relative_path, sample_rate, channels, duration, num_frames):
        row = {
            "cut_id": cut_id,
            "root_alias": root_alias,
            "relative_path": relative_path,
            "sample_rate": int(sample_rate),
            "channels": int(channels),
            "duration": float(duration),
            "num_frames": int(num_frames),
        }
        previous = self.rows.get(cut_id)
        if previous is None:
            self.rows[cut_id] = row
            return
        if (previous["root_alias"], previous["relative_path"]) != (root_alias, relative_path):
            raise RuntimeError(f"cut_id collision: {cut_id}")

    def add_path(self, cut_id, path):
        resolved = Path(path).resolve()
        root = DATA_ROOT.resolve()
        if not str(resolved).startswith(str(root)):
            raise RuntimeError(f"audio is outside DATA_ASR: {path}")
        relative = resolved.relative_to(root).as_posix()
        previous = self.rows.get(cut_id)
        if previous is not None:
            if previous["relative_path"] != relative:
                raise RuntimeError(f"cut_id collision: {cut_id}")
            return
        self.rows[cut_id] = {
            "cut_id": cut_id,
            "root_alias": "legacy_asr",
            "relative_path": relative,
            "sample_rate": None,
            "channels": None,
            "duration": None,
            "num_frames": None,
        }

    def probe(self):
        pending = [row for row in self.rows.values() if row["num_frames"] is None]
        print(f"probing {len(pending)} audio headers", flush=True)
        from concurrent.futures import ThreadPoolExecutor

        def read(row):
            path = DATA_ROOT / row["relative_path"]
            try:
                info = sf.info(path)
            except Exception as exc:
                raise RuntimeError(f"cannot read audio header {path}: {exc}") from exc
            return row["cut_id"], info.samplerate, info.channels, info.duration, info.frames

        with ThreadPoolExecutor(max_workers=32) as pool:
            for number, result in enumerate(pool.map(read, pending), 1):
                cut_id, sample_rate, channels, duration, frames = result
                row = self.rows[cut_id]
                row.update(sample_rate=sample_rate, channels=channels, duration=duration, num_frames=frames)
                if number % 20000 == 0:
                    print("probed", number, flush=True)


class Writer:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._handles = {}
        self.components = {}

    def write(self, name, record, hours, *, k, reps, language, group):
        handle = self._handles.get(name)
        if handle is None:
            path = self.directory / f"{name}.jsonl.gz"
            handle = gzip.open(path, "wt", encoding="utf-8", compresslevel=1)
            self._handles[name] = handle
            self.components[name] = {
                "name": name,
                "path": str(path),
                "k": k,
                "reps": reps,
                "language": language,
                "group": group,
                "records": 0,
                "hours": 0.0,
            }
        spec = self.components[name]
        if spec["reps"] != reps or spec["k"] != k:
            raise RuntimeError(f"inconsistent recipe for {name}")
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        spec["records"] += 1
        spec["hours"] += hours

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


def record_dict(record_id, language, duration, cut_id, start, target, identities, spans,
                candidates, k, reps, absent_probability, extra=None):
    metadata = {
        "sot_output_format": "aligned_utterance_timestamps_v1",
        "partition": "train",
        "speaker_identities": identities,
        "source_spans": spans,
        "recipe": {
            "name": "target-sot-20260930",
            "k": k,
            "reps": reps,
            "absent_probability": absent_probability,
            "mode": "random" if k else "all",
            "min_seconds": 1.0,
            "max_seconds": 5.0,
        },
    }
    if k:
        metadata["enrollment_candidates"] = candidates
    if extra:
        metadata.update(extra)
    return {
        "schema_version": "audio-record/1.0",
        "id": record_id,
        "task": "speaker_attributed_asr",
        "audio_slots": [{
            "name": "mixture",
            "purpose": "mixture",
            "ref": ref(cut_id, start, duration),
        }],
        "target": target,
        "language": language,
        "labels": {"speaker_count": len(identities), "recipe_k": k},
        "hotwords": [],
        "metadata": metadata,
    }


def cap_spans(spans, salt, limit=8):
    ranked = sorted(spans, key=lambda item: item[1] - item[0], reverse=True)[: max(limit * 2, limit)]
    rows = [{"key": f"{start:.3f}-{end:.3f}", "start": start, "end": end} for start, end in ranked]
    return [(row["start"], row["end"]) for row in choose(rows, limit, salt)]


def build_pools():
    by_speaker = defaultdict(list)
    by_lang = defaultdict(list)
    pools = {
        "aishell": SYN_ROOT / "pools/aishell.jsonl.gz",
        "aishell2": SYN_ROOT / "pools/aishell2.jsonl.gz",
        "kespeech": SYN_ROOT / "pools/kespeech.jsonl.gz",
        "librispeech": SYN_ROOT / "pools/librispeech.jsonl.gz",
        "commonvoice_en_clean": SYN_ROOT / "pools/commonvoice_en_clean.jsonl.gz",
    }
    seen = set()
    for name, path in pools.items():
        count = 0
        with gzip.open(path, "rt") as stream:
            for line in stream:
                if '"split":"train"' not in line and '"split": "train"' not in line:
                    continue
                item = json.loads(line)
                if item.get("split") != "train":
                    continue
                duration = float(item["duration"])
                if duration < 1.0:
                    continue
                speaker = item["speaker"]
                utt = {
                    "key": item["source_id"],
                    "source_id": item["source_id"],
                    "dataset_id": item["dataset_id"],
                    "duration": duration,
                    "start": float(item.get("start") or 0.0),
                    "channel": item.get("channel"),
                    "text": item.get("text") or "",
                    "language": item.get("language") or ("en" if name in {"librispeech", "commonvoice_en_clean"} else "zh"),
                    "audio": item["audio"],
                    "sample_rate": int(item["sample_rate"]),
                }
                by_speaker[speaker].append(utt)
                count += 1
        print(f"pool {name} utts>={1} {count}", flush=True)
    for speaker, utts in by_speaker.items():
        lang = utts[0]["language"]
        if speaker not in seen:
            by_lang[lang].append(speaker)
            seen.add(speaker)
    for lang in by_lang:
        by_lang[lang].sort()
    print("speakers", {lang: len(items) for lang, items in by_lang.items()}, flush=True)
    return by_speaker, by_lang


def register_utt(index, utt):
    cut_id = f"pool:{utt['dataset_id']}:{utt['source_id']}"
    audio = utt["audio"]
    index.add(
        cut_id,
        audio["root_alias"],
        audio["relative_path"],
        utt["sample_rate"],
        1,
        utt["duration"],
        int(round(utt["duration"] * utt["sample_rate"])),
    )
    return cut_id


def absent_ids(by_lang, languages, present, salt, count=4):
    pool = []
    for language in languages:
        pool.extend(by_lang.get(language, []))
    if not pool:
        return []
    chosen = []
    base = int(sha1(salt), 16)
    for step in range(count * 8):
        speaker = pool[(base + step * 9973) % len(pool)]
        if speaker in present or speaker in chosen:
            continue
        chosen.append(speaker)
        if len(chosen) == count:
            break
    return chosen


def utt_candidates(index, by_speaker, speaker, banned, salt, count):
    rows = [utt for utt in by_speaker.get(speaker, []) if utt["source_id"] not in banned]
    chosen = choose(rows, count, salt)
    out = []
    for utt in chosen:
        cut_id = register_utt(index, utt)
        out.append(candidate(speaker, cut_id, utt["start"], utt["duration"], utt["channel"]))
    return out


def load_syn_index():
    path = SYN_ROOT / "indexes/train.jsonl.gz"
    rows = {}
    with gzip.open(path, "rt") as stream:
        for line in stream:
            item = json.loads(line)
            rows[item["cut_id"]] = item
    print("synthetic index", len(rows), flush=True)
    return rows


def build_synthetic(index, writer, by_speaker, by_lang, limit=None):
    syn_index = load_syn_index()
    files = sorted((SYN_ROOT / "train").glob("*/*/*/*/records.jsonl.gz"))
    seen = 0
    reps_of = {5: {"zh": 3, "en": 3, "zh-en": 1}}
    for path in files:
        with gzip.open(path, "rt") as stream:
            for line in stream:
                item = json.loads(line)
                seen += 1
                if limit is not None and seen > limit:
                    print("synthetic stopped at limit", flush=True)
                    return
                speakers = item["metadata"]["speakers"]
                k = len(speakers)
                language = item.get("language") or path.parts[-5]
                if language not in {"zh", "en", "zh-en"}:
                    raise RuntimeError(f"unexpected language {language}")
                mixture_cut = item["audio_slots"][0]["ref"]["cut_id"]
                mixture = syn_index[mixture_cut]
                our_cut = f"syn:{mixture_cut}"
                index.add(
                    our_cut, mixture["root_alias"], mixture["relative_path"],
                    mixture["sample_rate"], mixture["channels"], mixture["duration"],
                    mixture["num_frames"],
                )
                duration = float(item["audio_slots"][0]["ref"]["duration"])
                segments = sorted(item["metadata"]["segments"], key=lambda row: (row["start"], row["speaker"]))
                turns = []
                spans = []
                banned = set()
                for segment in segments:
                    source = segment["source"]
                    banned.add(source["source_id"])
                    utt = {
                        "key": source["source_id"],
                        "source_id": source["source_id"],
                        "dataset_id": source["dataset_id"],
                        "duration": float(source["duration"]),
                        "start": float(source.get("start") or 0.0),
                        "channel": source.get("channel"),
                        "text": segment.get("text") or "",
                        "language": source.get("language") or language,
                        "audio": source["audio"],
                        "sample_rate": int(source.get("sample_rate") or 16000),
                    }
                    cut_id = register_utt(index, utt)
                    spans.append(ref(cut_id, utt["start"], utt["duration"], utt["channel"]))
                    turns.append({
                        "speaker": source["speaker"],
                        "start": float(segment["start"]),
                        "end": float(segment["start"]) + float(segment["duration"]),
                        "text": segment.get("text") or "",
                    })
                target, identities = render_turns(turns, duration)
                if not target:
                    continue
                present = list(identities.values())
                k = len(present)
                candidates = []
                for speaker in present:
                    candidates.extend(utt_candidates(
                        index, by_speaker, speaker, banned, f"{item['id']}:{speaker}", 3,
                    ))
                languages = ["zh", "en"] if language == "zh-en" else [language]
                for speaker in absent_ids(by_lang, languages, set(present), item["id"], 4):
                    candidates.extend(utt_candidates(
                        index, by_speaker, speaker, set(), f"{item['id']}:absent:{speaker}", 1,
                    ))
                reps = reps_of.get(k, {}).get(language, 1)
                name = f"k{k}_syn_{language.replace('-', '')}"
                writer.write(
                    name,
                    record_dict(
                        f"{item['id']}:k{k}", language, duration, our_cut, 0.0, target,
                        identities, spans, candidates, k, reps, 0.2,
                    ),
                    duration / 3600,
                    k=k, reps=reps, language=language, group="synthetic",
                )
                if seen % 100000 == 0:
                    print("synthetic", seen, flush=True)
    print("synthetic records", seen, flush=True)


def build_short(index, writer, by_speaker, limit=None):
    plan = [
        ("zh", "aishell", 80.0),
        ("zh", "aishell2", 180.0),
        ("zh", "kespeech", 140.0),
        ("en", "librispeech", 160.0),
        ("en", "commonvoice_en_clean", 240.0),
    ]
    buckets = {key: [] for _, key, _ in plan}
    for speaker, utts in by_speaker.items():
        if len(utts) < 2:
            continue
        dataset_id = utts[0]["dataset_id"]
        if dataset_id not in buckets:
            continue
        for utt in utts:
            buckets[dataset_id].append((speaker, utt))
    written = 0
    for language, dataset_id, hours in plan:
        rows = []
        for speaker, utt in buckets[dataset_id]:
            rows.append({
                "id": f"{dataset_id}:{utt['source_id']}",
                "hours": utt["duration"] / 3600,
                "speaker": speaker,
                "utt": utt,
            })
        chosen, got = select_hours(rows, hours)
        print(f"short {dataset_id} selected {got:.2f}h of target {hours}", flush=True)
        for row in chosen:
            if limit is not None and written >= limit:
                return
            utt = row["utt"]
            speaker = row["speaker"]
            others = [item for item in by_speaker[speaker] if item["source_id"] != utt["source_id"]]
            picks = choose(others, 3, row["id"])
            if not picks:
                continue
            cut_id = register_utt(index, utt)
            candidates = []
            for other in picks:
                other_id = register_utt(index, other)
                candidates.append(candidate(speaker, other_id, other["start"], other["duration"], other["channel"]))
            duration = utt["duration"]
            target, identities = render_turns([{
                "speaker": speaker,
                "start": 0.0,
                "end": duration,
                "text": utt["text"],
            }], duration)
            if not target:
                continue
            writer.write(
                f"k1_short_{language}",
                record_dict(
                    f"short:{row['id']}:k1", language, duration, cut_id, utt["start"], target,
                    identities, [ref(cut_id, utt["start"], duration, utt["channel"])],
                    candidates, 1, 1, 0.0,
                ),
                duration / 3600,
                k=1, reps=1, language=language, group="short",
            )
            written += 1


def load_dialog_parents(language):
    path = DATA_ROOT / f"emilia2/dialog/supervision/emilia2_dialog_{language}_train.jsonl"
    sessions = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            item = json.loads(line)
            speech = defaultdict(list)
            for turn in item["turns"]:
                speech[turn["speaker"]].append((turn["start"], turn["end"]))
            sessions[item["session_id"]] = {
                "audio": item["audio"],
                "duration": float(item["duration"]),
                "sr": int(item.get("sr") or 16000),
                "speech": {speaker: merge(intervals) for speaker, intervals in speech.items()},
            }
            if line_number % 50000 == 0:
                print(f"dialog parent {language}", line_number, flush=True)
    print(f"dialog parent {language} sessions", len(sessions), flush=True)
    return sessions


def dialog_candidates(session, lo, hi, present_only):
    grouped = {}
    for speaker, own in session["speech"].items():
        inside = intersects(own, lo, hi)
        if present_only is True and not inside:
            continue
        if present_only is False and inside:
            continue
        blockers = merge([
            piece for other, intervals in session["speech"].items() if other != speaker for piece in intervals
        ])
        solos = solo_outside(own, blockers, lo, hi)
        if solos:
            grouped[speaker] = solos
    return grouped


def build_dialog(index, writer, limit=None):
    plans = {
        1: {"zh": (None, 3), "en": (None, 1)},
        2: {"zh": (None, 1), "en": (2010.0, 1)},
        3: {"zh": (None, 1), "en": (2104.0, 1)},
        4: {"zh": (None, 3), "en": (1541.0, 1)},
        5: {"zh": (None, 3), "en": (None, 1)},
    }
    for language in ("zh", "en"):
        parents = load_dialog_parents(language)
        buckets = defaultdict(list)
        path = DATA_ROOT / f"emilia2/dialog/sot/supervision/emilia2_dialog_{language}_train.jsonl"
        kept = skipped = 0
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if limit is not None and line_number > limit:
                    break
                item = json.loads(line)
                duration = float(item["duration"])
                if "crop_start" not in item:
                    cut_id = f"dlg:{item['session_id']}"
                    index.add_path(cut_id, item["audio"])
                    turns = [{
                        "speaker": f"emilia2_dialog:{item['session_id']}:{turn['speaker']}",
                        "start": turn["start"],
                        "end": turn["end"],
                        "text": turn.get("text") or "",
                    } for turn in item["turns"]]
                    target, identities = render_turns(turns, duration)
                    if not target:
                        skipped += 1
                        continue
                    writer.write(
                        f"k0_dialog_{language}",
                        record_dict(
                            f"dialog:{item['session_id']}:k0", language, duration, cut_id, 0.0,
                            target, identities, [ref(cut_id, 0.0, duration)], [], 0, 1, 0.0,
                        ),
                        duration / 3600, k=0, reps=1, language=language, group="dialog",
                    )
                    kept += 1
                    continue
                parent_id = item["source_session_id"]
                session = parents.get(parent_id)
                if session is None:
                    skipped += 1
                    continue
                lo, hi = float(item["crop_start"]), float(item["crop_end"])
                present = dialog_candidates(session, lo, hi, True)
                absent = dialog_candidates(session, lo, hi, False)
                n = len(present)
                if n < 1:
                    skipped += 1
                    continue
                k = min(n, 5)
                mix_id = f"dlgmix:{item['session_id']}"
                parent_cut = f"dlg:{parent_id}"
                turns = [{
                    "speaker": f"emilia2_dialog:{parent_id}:{turn['speaker']}",
                    "start": turn["start"],
                    "end": turn["end"],
                    "text": turn.get("text") or "",
                } for turn in item["turns"]]
                target, identities = render_turns(turns, duration)
                if not target:
                    skipped += 1
                    continue
                buckets[k].append({
                    "id": item["session_id"],
                    "hours": duration / 3600,
                    "item": item,
                    "target": target,
                    "identities": identities,
                    "present": present,
                    "absent": absent,
                    "lo": lo,
                    "hi": hi,
                    "duration": duration,
                    "mix_id": mix_id,
                    "parent_cut": parent_cut,
                    "parent_id": parent_id,
                    "parent_audio": session["audio"],
                })
                if line_number % 50000 == 0:
                    print(f"dialog sot {language}", line_number, flush=True)
        print(f"dialog {language} k0_or_scanned kept_k0_part {kept} skipped {skipped}", flush=True)
        for k, rows in sorted(buckets.items()):
            target_hours, reps = plans[k][language]
            chosen, got = select_hours(rows, target_hours)
            print(f"dialog {language} k{k} {got:.2f}h reps {reps} items {len(chosen)}", flush=True)
            for row in chosen:
                index.add_path(row["mix_id"], row["item"]["audio"])
                index.add_path(row["parent_cut"], row["parent_audio"])
                candidates = []
                for speaker, spans in row["present"].items():
                    identity = f"emilia2_dialog:{row['parent_id']}:{speaker}"
                    for start, end in cap_spans(spans, f"{row['id']}:{speaker}"):
                        candidates.append(candidate(identity, row["parent_cut"], start, end - start))
                for speaker, spans in row["absent"].items():
                    identity = f"emilia2_dialog:{row['parent_id']}:{speaker}"
                    for start, end in cap_spans(spans, f"{row['id']}:absent:{speaker}"):
                        candidates.append(candidate(identity, row["parent_cut"], start, end - start))
                writer.write(
                    f"k{k}_dialog_{language}",
                    record_dict(
                        f"dialog:{row['id']}:k{k}", language, row["duration"], row["mix_id"], 0.0,
                        row["target"], row["identities"],
                        [ref(row["parent_cut"], row["lo"], row["hi"] - row["lo"])],
                        candidates, k, reps, 0.2,
                    ),
                    row["hours"], k=k, reps=reps, language=language, group="dialog",
                )
        del parents
        del buckets


def build_long(index, writer, limit=None):
    targets = {"zh": {"k0": 1181.0, "k1": 722.0}, "en": {"k0": 819.0, "k1": 1364.0}}
    for language in ("zh", "en"):
        path = DATA_ROOT / f"emilia2/long/sot/supervision/emilia2_long_{language}.jsonl"
        groups = defaultdict(list)
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if limit is not None and line_number > limit:
                    break
                item = json.loads(line)
                groups[item["source_id"]].append({
                    "id": item["id"],
                    "cut": bool(item.get("cut")),
                    "abs_start": float(item["abs_start"]),
                    "abs_end": float(item["abs_end"]),
                    "duration": float(item["duration"]),
                    "audio": item["audio"],
                    "members": item["members"],
                })
                if line_number % 100000 == 0:
                    print(f"long load {language}", line_number, flush=True)
        whole, cuts = [], []
        for source_id, clips in groups.items():
            identity = f"emilia2_long:{source_id}"
            for clip in clips:
                cut_id = f"long:{clip['id']}"
                turns = [{
                    "speaker": identity,
                    "start": member["start"],
                    "end": member["end"],
                    "text": member.get("text") or "",
                } for member in clip["members"]]
                target, identities = render_turns(turns, clip["duration"])
                if not target:
                    continue
                row = {
                    "id": clip["id"],
                    "hours": clip["duration"] / 3600,
                    "clip": clip,
                    "cut_id": cut_id,
                    "target": target,
                    "identities": identities,
                    "identity": identity,
                    "source_id": source_id,
                }
                if clip["cut"]:
                    cuts.append(row)
                else:
                    whole.append(row)
        chosen, got = select_hours(whole, targets[language]["k0"])
        print(f"long {language} k0 {got:.2f}h items {len(chosen)}", flush=True)
        for row in chosen:
            clip = row["clip"]
            index.add_path(row["cut_id"], clip["audio"])
            writer.write(
                f"k0_long_{language}",
                record_dict(
                    f"long:{row['id']}:k0", language, clip["duration"], row["cut_id"], 0.0,
                    row["target"], row["identities"], [ref(row["cut_id"], 0.0, clip["duration"])],
                    [], 0, 1, 0.0,
                ),
                row["hours"], k=0, reps=1, language=language, group="long",
            )
        enrollable = []
        by_source = groups
        for row in cuts:
            clip = row["clip"]
            siblings = []
            for other in by_source[row["source_id"]]:
                if other["id"] == clip["id"]:
                    continue
                pieces = []
                left_end = min(other["abs_end"], clip["abs_start"])
                if left_end - other["abs_start"] >= 1.0:
                    pieces.append((other["abs_start"], left_end))
                right_start = max(other["abs_start"], clip["abs_end"])
                if other["abs_end"] - right_start >= 1.0:
                    pieces.append((right_start, other["abs_end"]))
                for abs_start, abs_end in pieces:
                    if abs_end - abs_start < 1.0:
                        continue
                    local_start = abs_start - other["abs_start"]
                    local_end = abs_end - other["abs_start"]
                    local_end = min(local_end, other["duration"])
                    local_start = max(0.0, local_start)
                    if local_end - local_start < 1.0:
                        continue
                    siblings.append((f"long:{other['id']}", local_start, local_end - local_start, other["audio"]))
            if not siblings:
                continue
            siblings.sort(key=lambda item: item[2], reverse=True)
            row["siblings"] = siblings[:8]
            enrollable.append(row)
        chosen, got = select_hours(enrollable, targets[language]["k1"])
        print(f"long {language} k1 {got:.2f}h items {len(chosen)}", flush=True)
        for row in chosen:
            clip = row["clip"]
            index.add_path(row["cut_id"], clip["audio"])
            for cut_id, _, _, audio in row["siblings"]:
                index.add_path(cut_id, audio)
            candidates = [
                candidate(row["identity"], cut_id, start, duration)
                for cut_id, start, duration, _ in row["siblings"]
            ]
            writer.write(
                f"k1_long_{language}",
                record_dict(
                    f"long:{row['id']}:k1", language, clip["duration"], row["cut_id"], 0.0,
                    row["target"], row["identities"], [ref(row["cut_id"], 0.0, clip["duration"])],
                    candidates, 1, 1, 0.0,
                ),
                row["hours"], k=1, reps=1, language=language, group="long",
            )
        del groups


def meeting_blocks(windows):
    speech = defaultdict(list)
    for window in windows:
        origin = window["origin"]
        for turn in window["turns"]:
            speech[turn["source_speaker"]].append((origin + turn["start"], origin + turn["end"]))
    merged = {speaker: merge(intervals) for speaker, intervals in speech.items()}
    blockers = {
        speaker: merge([
            piece for other, intervals in merged.items() if other != speaker for piece in intervals
        ])
        for speaker in merged
    }
    return merged, blockers


def classify_window(window, merged, blockers):
    lo = window["origin"]
    hi = lo + window["duration"]
    present, absent = {}, {}
    for speaker, own in merged.items():
        solos = solo_outside(own, blockers[speaker], lo, hi)
        if not solos:
            continue
        if intersects(own, lo, hi):
            present[speaker] = solos
        else:
            absent[speaker] = solos
    return present, absent


def iter_meeting_windows(kind):
    durations = (30, 120, 300, 600)
    if kind == "aishell4":
        base = DATA_ROOT / "Derived/meeting-sot/meeting-long-v1-20260920"
        index_path = base / "aishell4-train-audio-index.jsonl.gz"
        for dur in durations:
            path = base / f"aishell4-train_{dur}s.jsonl.gz"
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    item = json.loads(line)
                    ref_item = item["audio_slots"][0]["ref"]
                    yield {
                        "id": item["id"],
                        "meeting_id": item["metadata"]["meeting_id"],
                        "origin": float(ref_item["start"] or 0.0),
                        "duration": float(ref_item["duration"]),
                        "language": item.get("language") or "zh",
                        "turns": [{
                            "source_speaker": turn["source_speaker"],
                            "start": float(turn["start"]),
                            "end": float(turn["end"]),
                            "text": turn.get("text") or "",
                        } for turn in item["metadata"].get("turns") or []],
                        "audio_cut": item["metadata"]["meeting_id"],
                        "audio_index": index_path,
                        "dataset_hint": "aishell4",
                    }
    elif kind == "alimeeting":
        base = DATA_ROOT / "Derived/meeting-sot/meeting-long-v1-20260920"
        index_path = base / "alimeeting_sdm-train-audio-index.jsonl.gz"
        for dur in durations:
            path = base / f"alimeeting_sdm-train_{dur}s.jsonl.gz"
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    item = json.loads(line)
                    ref_item = item["audio_slots"][0]["ref"]
                    yield {
                        "id": item["id"],
                        "meeting_id": item["metadata"]["meeting_id"],
                        "origin": float(ref_item["start"] or 0.0),
                        "duration": float(ref_item["duration"]),
                        "language": item.get("language") or "zh",
                        "turns": [{
                            "source_speaker": turn["source_speaker"],
                            "start": float(turn["start"]),
                            "end": float(turn["end"]),
                            "text": turn.get("text") or "",
                        } for turn in item["metadata"].get("turns") or []],
                        "audio_cut": item["metadata"]["meeting_id"],
                        "audio_index": index_path,
                        "dataset_hint": "alimeeting",
                    }
    else:
        roots = {
            "ramc": DATA_ROOT / "MagicData-RAMC/sot",
            "ami": DATA_ROOT / "ami_full/sot",
            "chime6": DATA_ROOT / "chime6/sot",
        }
        languages = {"ramc": "zh", "ami": "en", "chime6": "en"}
        root = roots[kind]
        for dur in durations:
            path = root / "supervision" / f"train_{dur}s.jsonl"
            with path.open() as stream:
                for line in stream:
                    item = json.loads(line)
                    audio = item["audio"]
                    audio_path = audio if audio.startswith("/") else str(root / audio)
                    yield {
                        "id": item["id"],
                        "meeting_id": item["meeting_id"],
                        "origin": float(item["source_start"]),
                        "duration": float(item["duration"]),
                        "language": languages[kind],
                        "turns": [{
                            "source_speaker": turn["source_speaker"],
                            "start": float(turn["start"]),
                            "end": float(turn["end"]),
                            "text": turn.get("text") or "",
                        } for turn in item.get("turns") or []],
                        "window_audio": audio_path,
                        "source_audio": item["source_audio"],
                        "dataset_hint": kind,
                    }


def ensure_meeting_index(index, cache, window):
    if "audio_index" in window:
        key = window["audio_index"]
        if key not in cache:
            loaded = {}
            with gzip.open(key, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    loaded[row["cut_id"]] = row
            cache[key] = loaded
        row = cache[key][window["audio_cut"]]
        cut_id = f"meet:{window['dataset_hint']}:{window['meeting_id']}"
        index.add(
            cut_id, row["root_alias"], row["relative_path"], row["sample_rate"],
            row["channels"], row["duration"], row["num_frames"],
        )
        return cut_id, cut_id
    source_id = f"meet:{window['dataset_hint']}:{window['meeting_id']}"
    window_id = f"win:{window['dataset_hint']}:{window['id']}"
    index.add_path(source_id, window["source_audio"])
    index.add_path(window_id, window["window_audio"])
    return window_id, source_id


def build_meetings(index, writer, limit=None):
    cache = {}
    for kind in ("aishell4", "alimeeting", "ramc", "ami", "chime6"):
        grouped = defaultdict(list)
        for number, window in enumerate(iter_meeting_windows(kind), 1):
            if limit is not None and number > limit:
                break
            grouped[window["meeting_id"]].append(window)
        print(kind, "meetings", len(grouped), "windows", sum(len(v) for v in grouped.values()), flush=True)
        buckets = defaultdict(list)
        k0_rows = []
        for meeting_id, windows in grouped.items():
            merged, blockers = meeting_blocks(windows)
            for window in windows:
                present, absent = classify_window(window, merged, blockers)
                mixture_cut, source_cut = ensure_meeting_index(index, cache, window)
                prefix = f"{kind}:{meeting_id}"
                turns = [{
                    "speaker": f"{prefix}:{turn['source_speaker']}",
                    "start": turn["start"],
                    "end": turn["end"],
                    "text": turn["text"],
                } for turn in window["turns"]]
                target, identities = render_turns(turns, window["duration"])
                span_start = window["origin"] if mixture_cut == source_cut else 0.0
                row = {
                    "id": window["id"],
                    "hours": window["duration"] / 3600,
                    "language": window["language"],
                    "duration": window["duration"],
                    "target": target,
                    "identities": identities,
                    "mixture_cut": mixture_cut,
                    "source_cut": source_cut,
                    "span_start": span_start,
                    "origin": window["origin"],
                    "present": present,
                    "absent": absent,
                    "prefix": prefix,
                }
                k0_rows.append(row)
                if present:
                    buckets[min(len(present), 5)].append(row)
        for row in k0_rows:
            writer.write(
                f"k0_{kind}",
                record_dict(
                    f"{kind}:{row['id']}:k0", row["language"], row["duration"], row["mixture_cut"],
                    row["span_start"], row["target"], row["identities"],
                    [ref(row["source_cut"], row["origin"], row["duration"])],
                    [], 0, 2, 0.0,
                ),
                row["hours"], k=0, reps=2, language=row["language"], group="meeting",
            )
        for k, rows in sorted(buckets.items()):
            print(f"meeting {kind} k{k} {sum(r['hours'] for r in rows):.2f}h items {len(rows)}", flush=True)
            for row in rows:
                candidates = []
                # Spans are on the source timeline. Window-relative mixtures still
                # use the source file for enrollment.
                for speaker, spans in row["present"].items():
                    identity = f"{row['prefix']}:{speaker}"
                    for start, end in cap_spans(spans, f"{row['id']}:{speaker}"):
                        candidates.append(candidate(identity, row["source_cut"], start, end - start))
                for speaker, spans in row["absent"].items():
                    identity = f"{row['prefix']}:{speaker}"
                    for start, end in cap_spans(spans, f"{row['id']}:absent:{speaker}"):
                        candidates.append(candidate(identity, row["source_cut"], start, end - start))
                writer.write(
                    f"k{k}_{kind}",
                    record_dict(
                        f"{kind}:{row['id']}:k{k}", row["language"], row["duration"], row["mixture_cut"],
                        row["span_start"], row["target"], row["identities"],
                        [ref(row["source_cut"], row["origin"], row["duration"])],
                        candidates, k, 2, 0.2,
                    ),
                    row["hours"], k=k, reps=2, language=row["language"], group="meeting",
                )


def write_catalog(writer, index):
    records_dir = OUT / "records"
    index_path = OUT / "audio-index.jsonl.gz"
    index.probe()
    print("writing audio index", len(index.rows), flush=True)
    with gzip.open(index_path, "wt", encoding="utf-8", compresslevel=1) as stream:
        for cut_id in sorted(index.rows):
            stream.write(json.dumps(index.rows[cut_id], ensure_ascii=False, separators=(",", ":")) + "\n")
    artifacts = [{
        "name": "audio_index",
        "kind": "audio-index",
        "root_alias": "legacy_asr",
        "relative_path": index_path.resolve().relative_to(DATA_ROOT).as_posix(),
        "metadata": {"records": len(index.rows)},
    }]
    names = []
    components = []
    for name, spec in sorted(writer.components.items()):
        path = Path(spec["path"])
        artifact_name = f"records_{name}"
        names.append(artifact_name)
        artifacts.append({
            "name": artifact_name,
            "kind": "audio-records",
            "root_alias": "legacy_asr",
            "relative_path": path.resolve().relative_to(DATA_ROOT).as_posix(),
            "metadata": {"record_count": spec["records"]},
        })
        train_hours = spec["hours"] * spec["reps"]
        components.append({**spec, "hours": round(spec["hours"], 4), "train_hours": round(train_hours, 4)})
    catalog = {
        "schema_version": "dataset-catalog/1.0",
        "dataset_id": DS,
        "version": VER,
        "languages": ["zh", "en", "zh-en"],
        "tasks": ["speaker_attributed_asr"],
        "aliases": [],
        "artifacts": artifacts,
        "splits": {
            "train": {
                "records_artifacts": names,
                "audio_index_artifact": "audio_index",
                "statistics": {
                    "records": sum(item["records"] for item in components),
                    "train_hours": round(sum(item["train_hours"] for item in components), 4),
                },
            }
        },
        "recipe_parameters": {"components": components},
        "provenance": {
            "description": "Enrollment recipe supervision. Audio is referenced, not copied. reps repeats a file at train time.",
        },
    }
    catalog_path = OUT / "catalog.jsonl"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False) + "\n")
    recipe = {
        "dataset_id": DS,
        "version": VER,
        "catalog": str(catalog_path),
        "roots": str(SYN_ROOT / "roots.json"),
        "audio_index": str(index_path),
        "notes": [
            "Each record is stored once. Integer repeats are the reps field, applied when sampling.",
            "Enrollment refs are single-speaker spans of at least 1 second. Crop to 1-5 seconds at train time.",
            "metadata.recipe.k is the assigned enrollment count. Absent identities are included for dialog, meetings and synthetic.",
            "Current trainer EnrollmentConfig.max_targets still accepts only 1, 2 or 3.",
        ],
        "components": components,
        "by_k": {},
        "by_group": {},
    }
    by_k, by_group = defaultdict(float), defaultdict(float)
    for item in components:
        by_k[item["k"]] += item["train_hours"]
        by_group[item["group"]] += item["train_hours"]
    recipe["by_k"] = {str(k): round(by_k[k], 2) for k in sorted(by_k)}
    recipe["by_group"] = {key: round(value, 2) for key, value in sorted(by_group.items())}
    recipe["train_hours"] = round(sum(by_k.values()), 2)
    (OUT / "recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"by_k": recipe["by_k"], "by_group": recipe["by_group"], "train_hours": recipe["train_hours"]}, ensure_ascii=False), flush=True)


def validate_samples():
    sys.path.insert(0, str(Path("/222042021/lx/open-audio-llm-multi-target/src")))
    sys.path.insert(0, str(Path("/222042021/lx/audio-data-contract/src")))
    from audio_data_contract import AudioRecord, load_catalog
    from open_audio_llm.data.target_sot import EnrollmentConfig, eligible_candidates

    load_catalog(OUT / "catalog.jsonl")
    config = EnrollmentConfig()
    checked = 0
    for path in sorted((OUT / "records").glob("*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            item = json.loads(stream.readline())
        record = AudioRecord.from_dict(item)
        if record.metadata["recipe"]["k"]:
            groups = eligible_candidates(record, config)
            if not groups:
                raise RuntimeError(f"no enrollment candidates: {path}")
        checked += 1
    print("validated files", checked, flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()
    if args.output is not None:
        OUT = args.output
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"output already exists: {OUT}")
    index = Index()
    writer = Writer(OUT / "records")
    try:
        by_speaker, by_lang = build_pools()
        build_synthetic(index, writer, by_speaker, by_lang, args.limit)
        build_short(index, writer, by_speaker, args.limit)
        del by_speaker, by_lang
        build_dialog(index, writer, args.limit)
        build_long(index, writer, args.limit)
        build_meetings(index, writer, args.limit)
    finally:
        writer.close()
    write_catalog(writer, index)
    if not args.skip_validate:
        validate_samples()


if __name__ == "__main__":
    main()
