#!/usr/bin/env python3
"""Build dialog and long enrollment records from Emilia2 segment metadata.

Mixture hours are counted once. Repeats stay at 1. Windows cut from files
longer than 10 minutes keep the last 60 seconds out of every window so the
same file can still supply enrollment. Records point at the source m4a.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

DS = "target_sot_emilia"
VER = "20261004"
SPLIT = "train"
RECIPE = "target-sot-emilia-20261004"
DATA_ROOT = Path("/ai_sds_wuzz/DATA_ASR")
META = DATA_ROOT / "emilia2/segment_metainfo"
AUDIO = DATA_ROOT / "emilia2/raw_data/audio"
OUT = DATA_ROOT / "Derived/target-sot-emilia-20261004"
ROOTS = Path("/222042021/mingdong/data/sot-multispeaker/synthetic-v2-20260915/roots.json")
SCAN_VERSION = "1"
KEEP_STATUS = {"ok", "reused"}
WORKERS = 24

# Per language. Dialog sums to 9020 h. Long bins sum to 2042 h; the published
# 2043 h target rounds the same percentages.
DIALOG_BANDS = (
    ("30-60", 30.0, 60.0, 451.0),
    ("1-2", 60.0, 120.0, 722.0),
    ("2-3", 120.0, 180.0, 1001.0),
    ("3-5", 180.0, 300.0, 2336.0),
    ("5-6", 300.0, 360.0, 902.0),
    ("6-7", 360.0, 420.0, 902.0),
    ("7-8", 420.0, 480.0, 902.0),
    ("8-9", 480.0, 540.0, 902.0),
    ("9-10", 540.0, 600.0, 902.0),
)
LONG_BANDS = (
    ("10-30", 10.0, 30.0, 102.0),
    ("30-60", 30.0, 60.0, 176.0),
    ("1-2", 60.0, 120.0, 227.0),
    ("2-3", 120.0, 180.0, 184.0),
    ("3-5", 180.0, 300.0, 333.0),
    ("5-6", 300.0, 360.0, 204.0),
    ("6-7", 360.0, 420.0, 204.0),
    ("7-8", 420.0, 480.0, 204.0),
    ("8-9", 480.0, 540.0, 204.0),
    ("9-10", 540.0, 600.0, 204.0),
)
DIALOG_CUT = tuple(band for band in DIALOG_BANDS if band[1] >= 180.0)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def hours_of(frames: int, sample_rate: int) -> float:
    return frames / sample_rate / 3600.0


def probe_header(path: str):
    import av

    container = av.open(path)
    try:
        stream = container.streams.audio[0]
        sample_rate = int(stream.rate or stream.codec_context.sample_rate)
        channels = int(stream.channels or stream.codec_context.channels or 1)
        if stream.duration is None or stream.time_base is None or sample_rate <= 0:
            return None
        if stream.time_base.numerator == 1 and stream.time_base.denominator == sample_rate:
            frames = int(stream.duration)
        else:
            frames = int(round(float(stream.duration * stream.time_base) * sample_rate))
    finally:
        container.close()
    if frames <= 0 or channels <= 0:
        return None
    return sample_rate, channels, frames


def dialog_segments(obj):
    rows = []
    for member in obj.get("members") or []:
        if str(member.get("text_status") or "") not in KEEP_STATUS:
            continue
        text = " ".join(str(member.get("text") or "").split())
        speaker = str(member.get("speaker") or "").strip()
        try:
            start, end = float(member["start"]), float(member["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not speaker or not text or not (math.isfinite(start) and math.isfinite(end)):
            continue
        if end - start < 0.01:
            continue
        rows.append((speaker, start, end, text))
    return rows


def long_segments(obj):
    rows = []
    for member in obj.get("members") or []:
        text = " ".join(str(member.get("text") or "").split())
        try:
            start, end = float(member["start"]), float(member["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not text or not (math.isfinite(start) and math.isfinite(end)) or end - start < 0.01:
            continue
        rows.append((start, end, text))
    return rows


def language_of(obj):
    langs = obj.get("languages")
    if langs == ["zh"]:
        return "zh"
    if langs == ["en"]:
        return "en"
    return None


def bucket(duration: float, bands) -> str | None:
    if duration > 600.0:
        return None
    if 540.0 <= duration <= 600.0:
        return "9-10"
    for name, lo, hi, _target in bands:
        if name == "9-10":
            continue
        if lo <= duration < hi:
            return name
    return None


def scan_tar(kind: str, tar_path: str, dest: str):
    stats = {"rows": 0, "no_audio": 0, "bad_lang": 0, "invalid": 0, "no_text": 0, "header": 0, "bad_json": 0, "error": ""}
    rows = []
    try:
        with tarfile.open(tar_path, "r") as handle:
            for member in handle:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                try:
                    payload = json.loads(handle.extractfile(member).read())
                except Exception:
                    stats["bad_json"] += 1
                    continue
                if payload.get("is_valid") is not True:
                    stats["invalid"] += 1
                    continue
                lang = language_of(payload)
                if lang is None:
                    stats["bad_lang"] += 1
                    continue
                rid = str(payload.get("recording_id") or "").strip()
                item_id = str(payload.get("id") or "").strip()
                if not rid or not item_id:
                    stats["bad_json"] += 1
                    continue
                stem = Path(member.name).stem
                audio = AUDIO / rid / kind / f"{stem}.m4a"
                if not audio.is_file():
                    stats["no_audio"] += 1
                    continue
                if kind == "dialog":
                    segments = dialog_segments(payload)
                    spans = tuple((speaker, start, end) for speaker, start, end, _text in segments)
                else:
                    segments = long_segments(payload)
                    spans = tuple((start, end) for start, end, _text in segments)
                if not spans:
                    stats["no_text"] += 1
                    continue
                header = probe_header(str(audio))
                if header is None:
                    stats["header"] += 1
                    continue
                sample_rate, channels, frames = header
                rows.append({
                    "id": item_id,
                    "rid": rid,
                    "lang": lang,
                    "frames": frames,
                    "sr": sample_rate,
                    "ch": channels,
                    "tar": Path(tar_path).name,
                    "member": member.name,
                    "stem": stem,
                    "spans": spans,
                })
    except Exception as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
        return stats
    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(rows, handle, protocol=4)
    os.replace(temporary, destination)
    stats["rows"] = len(rows)
    return stats


def select_hours(rows, target):
    total = sum(hours_of(row["frames"], row["sr"]) for row in rows)
    if total <= target + 1e-9:
        return list(rows), total
    ordered = sorted(rows, key=lambda row: sha1(row["id"]))
    chosen, acc = [], 0.0
    for row in ordered:
        if acc >= target:
            break
        chosen.append(row)
        acc += hours_of(row["frames"], row["sr"])
    return chosen, acc


def overlaps_window(spans, lo, hi, dialog: bool) -> bool:
    for span in spans:
        start, end = (span[1], span[2]) if dialog else (span[0], span[1])
        if min(end, hi) - max(start, lo) >= 0.01:
            return True
    return False


def next_content(spans, cursor, dialog: bool):
    starts = []
    for span in spans:
        start, end = (span[1], span[2]) if dialog else (span[0], span[1])
        if end > cursor + 0.01:
            starts.append(max(start, cursor))
    return min(starts) if starts else None


def plan_dialog_cuts(files, needs):
    planned = []
    ordered = sorted(files, key=lambda row: sha1(row["id"]))
    for item in ordered:
        if all(value <= 1e-4 for value in needs.values()):
            break
        duration = item["frames"] / item["sr"]
        usable = duration - 60.0
        cursor = 0.0
        for _guard in range(10000):
            room = usable - cursor
            if room < 180.0:
                break
            options = []
            for name, lo, hi, _target in DIALOG_CUT:
                if needs[name] <= 1e-4 or room < lo:
                    continue
                length = min(hi - 1e-4, room)
                if length < lo:
                    continue
                options.append((needs[name], name, length))
            if not options:
                break
            _score, band, length = max(options)
            end = cursor + length
            if not overlaps_window(item["spans"], cursor, end, True):
                nxt = next_content(item["spans"], cursor, True)
                if nxt is None or nxt >= usable - 180.0:
                    break
                cursor = nxt
                continue
            planned.append((item, cursor, end, band))
            needs[band] -= length / 3600.0
            cursor = end
    return planned


def plan_long_cuts(files, target):
    planned, acc = [], 0.0
    for item in sorted(files, key=lambda row: sha1(row["id"])):
        if acc >= target:
            break
        length = 10.0 + (int(sha1(item["id"]), 16) % 2000) / 100.0
        if item["frames"] / item["sr"] < length + 1.0:
            continue
        if not overlaps_window(item["spans"], 0.0, length, False):
            continue
        planned.append((item, 0.0, length, "10-30"))
        acc += length / 3600.0
    return planned, acc


def load_text_tar(kind: str, tar_path: str, members: list, dest: str):
    wanted = set(members)
    found = {}
    with tarfile.open(tar_path, "r") as handle:
        for member in handle:
            if not member.isfile() or member.name not in wanted:
                continue
            payload = json.loads(handle.extractfile(member).read())
            found[member.name] = dialog_segments(payload) if kind == "dialog" else long_segments(payload)
    destination = Path(dest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(found, handle, protocol=4)
    os.replace(temporary, destination)
    return len(found)


def render_turns(turns, duration):
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
    for _start, _end, speaker, _text in cleaned:
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
        if right <= left or left < 0 or right - left < 0.01:
            continue
        lines.append(f"[{name}][{left:.2f}-{right:.2f}] {text}")
        last_end[name] = right
    identities = {
        label[speaker]: speaker
        for speaker in order
        if any(line.startswith(f"[{label[speaker]}]") for line in lines)
    }
    return "\n".join(lines), identities


def ref(cut_id, start, duration):
    return {
        "dataset_id": DS,
        "version": VER,
        "split": SPLIT,
        "cut_id": cut_id,
        "start": float(start),
        "duration": float(duration),
    }


def choose(rows, count, salt):
    if not rows or count <= 0:
        return []
    ordered = sorted(rows, key=lambda row: row["key"])
    start = int(sha1(salt), 16) % len(ordered)
    return [ordered[(start + i) % len(ordered)] for i in range(min(count, len(ordered)))]


def cap_spans(spans, salt, limit=8):
    ranked = sorted(spans, key=lambda item: item[1] - item[0], reverse=True)[: max(limit * 2, limit)]
    rows = [{"key": f"{start:.3f}-{end:.3f}", "start": start, "end": end} for start, end in ranked]
    return [(row["start"], row["end"]) for row in choose(rows, limit, salt)]


def pieces_outside(start, end, lo, hi, file_end):
    start, end = max(0.0, start), min(file_end, end)
    if end - start < 1.0:
        return []
    if end <= lo or start >= hi:
        return [(start, end)]
    pieces = []
    if lo - start >= 1.0:
        pieces.append((start, lo))
    if end - hi >= 1.0:
        pieces.append((hi, end))
    return pieces


def record_dict(record_id, language, duration, cut_id, start, target, identities, candidates, k, absent):
    metadata = {
        "sot_output_format": "aligned_utterance_timestamps_v1",
        "partition": "train",
        "speaker_identities": identities,
        "source_spans": [ref(cut_id, start, duration)],
        "recipe": {
            "name": RECIPE,
            "k": k,
            "reps": 1,
            "absent_probability": absent,
            "mode": "random" if k else "all",
            "min_seconds": 1.0,
            "max_seconds": 5.0,
        },
    }
    if k:
        metadata["enrollment_candidates"] = candidates
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


def cut_id_of(kind, item):
    return f"emilia2_{kind}:{item['rid']}:{item['stem']}"


def audio_relative(kind, item):
    return f"emilia2/raw_data/audio/{item['rid']}/{kind}/{item['stem']}.m4a"


def add_index(index, kind, item):
    cut_id = cut_id_of(kind, item)
    relative = audio_relative(kind, item)
    previous = index.get(cut_id)
    row = {
        "cut_id": cut_id,
        "root_alias": "legacy_asr",
        "relative_path": relative,
        "sample_rate": int(item["sr"]),
        "channels": int(item["ch"]),
        "duration": item["frames"] / item["sr"],
        "num_frames": int(item["frames"]),
    }
    if previous is None:
        index[cut_id] = row
        return
    if previous["relative_path"] != relative or previous["num_frames"] != row["num_frames"]:
        raise RuntimeError(f"cut_id collision: {cut_id}")


def dialog_candidates(item, start, end, by_rid, items, index):
    lo, hi = start, start + (end - start)
    file_end = item["frames"] / item["sr"]
    groups = defaultdict(list)
    for other in by_rid[item["rid"]]:
        other_end = other["frames"] / other["sr"]
        same = other["id"] == item["id"]
        for speaker, span_start, span_end in other["spans"]:
            identity = f"emilia2_dialog:{item['rid']}:{speaker}"
            if same:
                pieces = pieces_outside(span_start, span_end, lo, hi, file_end)
            else:
                left, right = max(0.0, span_start), min(other_end, span_end)
                pieces = [(left, right)] if right - left >= 1.0 else []
            for left, right in pieces:
                groups[identity].append((left, right, other))
    capped = {}
    for identity, spans in groups.items():
        chosen = cap_spans([(left, right) for left, right, _other in spans], f"{item['id']}:{lo:.3f}:{identity}")
        if not chosen:
            continue
        # cap_spans drops the source file. Map back by approximate edges.
        located = []
        for left, right in chosen:
            for span_left, span_right, other in spans:
                if abs(span_left - left) < 1e-6 and abs(span_right - right) < 1e-6:
                    located.append((other, left, right - left))
                    break
        if located:
            capped[identity] = located
    return capped


def materialize_candidates(capped, index, kind):
    candidates = []
    for identity, located in capped.items():
        for other, start, duration in located:
            add_index(index, kind, other)
            candidates.append({
                "speaker_id": identity,
                "partition": "train",
                "single_speaker": True,
                "ref": ref(cut_id_of(kind, other), start, duration),
            })
    return candidates


def turns_from_dialog(segments, start, end, rid):
    turns = []
    for speaker, span_start, span_end, text in segments:
        left, right = max(span_start, start), min(span_end, end)
        if right - left < 0.01:
            continue
        turns.append({
            "start": left - start,
            "end": right - start,
            "speaker": f"emilia2_dialog:{rid}:{speaker}",
            "text": text,
        })
    return turns


def turns_from_long(segments, start, end, identity):
    turns = []
    for span_start, span_end, text in segments:
        left, right = max(span_start, start), min(span_end, end)
        if right - left < 0.01:
            continue
        turns.append({"start": left - start, "end": right - start, "speaker": identity, "text": text})
    return turns


def snap_range(frames, sample_rate, start, end, reserve_samples):
    limit = frames - reserve_samples
    if limit <= 0:
        return None
    start_n = min(max(int(round(start * sample_rate)), 0), limit)
    end_n = min(max(int(round(end * sample_rate)), start_n), limit)
    if end_n - start_n < int(0.01 * sample_rate):
        return None
    return start_n, end_n


def run_pool(jobs, function, label):
    if not jobs:
        return
    done = 0
    errors = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(function, *job): job for job in jobs}
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
            except Exception as exc:
                errors += 1
                print(f"{label} failed {futures[future][1]}: {exc}", flush=True)
                if errors > 30:
                    raise
                continue
            if isinstance(result, dict) and result.get("error"):
                errors += 1
                print(f"{label} error {futures[future][1]}: {result['error']}", flush=True)
                if errors > 30:
                    raise RuntimeError(result["error"])
            if done % 200 == 0 or done == len(futures):
                print(f"{label} {done}/{len(futures)}", flush=True)


def scan_all():
    version_path = OUT / "work" / "VERSION"
    if version_path.is_file() and version_path.read_text().strip() == SCAN_VERSION:
        print("reusing scan cache", flush=True)
        return
    if (OUT / "work").exists():
        for path in (OUT / "work").rglob("*"):
            if path.is_file():
                path.unlink()
    jobs = []
    for kind in ("dialog", "long"):
        for tar in sorted((META / kind).glob("*.tar")):
            dest = OUT / "work" / kind / f"{tar.name}.pkl"
            jobs.append((kind, str(tar), str(dest)))
    print(f"scanning {len(jobs)} tars", flush=True)
    run_pool(jobs, scan_tar, "scan")
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_text(SCAN_VERSION + "\n")


def load_items():
    loaded = {"dialog": [], "long": []}
    seen = set()
    duplicates = 0
    for kind in ("dialog", "long"):
        files = sorted((OUT / "work" / kind).glob("*.pkl"))
        print(f"loading {kind} {len(files)} caches", flush=True)
        for path in files:
            with path.open("rb") as handle:
                rows = pickle.load(handle)
            for row in rows:
                key = (kind, row["rid"], row["stem"])
                if key in seen or (kind, row["id"]) in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                seen.add((kind, row["id"]))
                loaded[kind].append(row)
    print(f"loaded dialog {len(loaded['dialog'])} long {len(loaded['long'])} duplicates {duplicates}", flush=True)
    return loaded


def summarize_supply(items):
    supply = {"dialog": defaultdict(lambda: defaultdict(float)), "long": defaultdict(lambda: defaultdict(float))}
    for kind, bands in (("dialog", DIALOG_BANDS), ("long", LONG_BANDS)):
        for item in items[kind]:
            duration = item["frames"] / item["sr"]
            name = "gt10" if duration > 600 else bucket(duration, bands)
            if name is None:
                name = "other"
            supply[kind][item["lang"]][name] += hours_of(item["frames"], item["sr"])
    for kind in ("dialog", "long"):
        for lang in ("zh", "en"):
            parts = ", ".join(f"{name} {supply[kind][lang][name]:.1f}" for name in [band[0] for band in (DIALOG_BANDS if kind == "dialog" else LONG_BANDS)] + ["gt10", "other"])
            print(f"supply {kind} {lang}: {parts}", flush=True)
    dialog_zh = sum(supply["dialog"]["zh"][band[0]] for band in DIALOG_BANDS)
    dialog_en = sum(supply["dialog"]["en"][band[0]] for band in DIALOG_BANDS)
    if not (6500 <= dialog_zh <= 7800 and 8500 <= dialog_en <= 11000):
        raise SystemExit(f"dialog supply outside expected range: zh {dialog_zh:.1f} en {dialog_en:.1f}")
    return supply


def plan(items):
    selected = {"dialog": [], "long": []}
    for lang in ("zh", "en"):
        native = defaultdict(list)
        cuts_from = []
        for item in items["dialog"]:
            if item["lang"] != lang:
                continue
            duration = item["frames"] / item["sr"]
            if duration > 600:
                cuts_from.append(item)
                continue
            name = bucket(duration, DIALOG_BANDS)
            if name is not None:
                native[name].append(item)
        needs = {}
        for name, _lo, _hi, target in DIALOG_BANDS:
            chosen, got = select_hours(native[name], target)
            for item in chosen:
                selected["dialog"].append((item, 0.0, item["frames"] / item["sr"], name, False))
            needs[name] = target - got
            print(f"dialog {lang} {name}: native {got:.2f} / {target:.2f}", flush=True)
        cut_needs = {name: max(0.0, needs[name]) for name, _lo, _hi, _target in DIALOG_CUT}
        windows = plan_dialog_cuts(cuts_from, cut_needs)
        for item, start, end, band in windows:
            selected["dialog"].append((item, start, end, band, True))
        print(f"dialog {lang} cuts {sum(end - start for _item, start, end, _band in windows) / 3600:.2f} h across {len(windows)} windows", flush=True)
        for name, _lo, _hi, _target in DIALOG_CUT:
            print(f"dialog {lang} {name} still short {cut_needs[name]:.2f} h", flush=True)

        native = defaultdict(list)
        cuts_from = []
        for item in items["long"]:
            if item["lang"] != lang:
                continue
            duration = item["frames"] / item["sr"]
            if duration > 600:
                cuts_from.append(item)
                continue
            name = bucket(duration, LONG_BANDS)
            if name is not None:
                native[name].append(item)
        for name, _lo, _hi, target in LONG_BANDS:
            if name == "10-30":
                continue
            chosen, got = select_hours(native[name], target)
            for item in chosen:
                selected["long"].append((item, 0.0, item["frames"] / item["sr"], name, False))
            print(f"long {lang} {name}: native {got:.2f} / {target:.2f}", flush=True)
        chosen, got = select_hours(native["10-30"], 102.0)
        for item in chosen:
            selected["long"].append((item, 0.0, item["frames"] / item["sr"], "10-30", False))
        windows, cut_hours = plan_long_cuts(cuts_from, max(0.0, 102.0 - got))
        for item, start, end, band in windows:
            selected["long"].append((item, start, end, band, True))
        print(f"long {lang} 10-30 native {got:.2f} cuts {cut_hours:.2f}", flush=True)
    print(f"planned dialog {len(selected['dialog'])} long {len(selected['long'])}", flush=True)
    return selected


def load_texts(selected):
    text_root = OUT / "work" / "text"
    if text_root.exists():
        for path in text_root.rglob("*"):
            if path.is_file():
                path.unlink()
    grouped = defaultdict(set)
    for kind in ("dialog", "long"):
        for item, _start, _end, _band, _cut in selected[kind]:
            grouped[(kind, item["tar"])].add(item["member"])
    jobs = []
    for (kind, tar_name), members in grouped.items():
        dest = text_root / kind / f"{tar_name}.pkl"
        jobs.append((kind, str(META / kind / tar_name), sorted(members), str(dest)))
    print(f"reading text from {len(jobs)} tars", flush=True)
    run_pool(jobs, load_text_tar, "text")
    texts = {}
    for kind in ("dialog", "long"):
        for path in (text_root / kind).glob("*.pkl"):
            with path.open("rb") as handle:
                found = pickle.load(handle)
            for member, segments in found.items():
                texts[(kind, path.name[:-4], member)] = segments
    print(f"text members {len(texts)}", flush=True)
    return texts


def assert_loader_duration(path, frames, sample_rate, channels, start_n, dur_n):
    sys.path.insert(0, "/222042021/lx/audio-data-contract/src")
    sys.path.insert(0, "/222042021/lx/open-audio-llm-multi-target/src")
    from io import BytesIO

    import soundfile as sf
    from lhotse import AudioSource, Recording

    from open_audio_llm.data.catalog_dataset import load_mono
    from open_audio_llm.data.pyav_backend import install_pyav_backend

    install_pyav_backend()
    declared = dur_n / sample_rate
    recording = Recording(
        id="probe",
        sources=[AudioSource("file", list(range(channels)), path)],
        sampling_rate=sample_rate,
        num_samples=frames,
        duration=frames / sample_rate,
    )
    cut = recording.to_cut().truncate(offset=start_n / sample_rate, duration=declared)
    audio = load_mono(cut, 16000)
    buffer = BytesIO()
    sf.write(buffer, audio, 16000, format="WAV", subtype="FLOAT")
    got = sf.info(BytesIO(buffer.getvalue())).duration
    if abs(got - declared) > 1 / 16000:
        raise SystemExit(
            f"duration mismatch {path}: declared {declared:.9f} got {got:.9f} diff {got - declared:.9f}"
        )
    print(f"duration ok {Path(path).name} {declared:.4f}s", flush=True)


def check_duration_contract(items):
    sample = next(item for item in items["long"] if 20 * item["sr"] < item["frames"] < 180 * item["sr"])
    path = str(AUDIO / sample["rid"] / "long" / f"{sample['stem']}.m4a")
    assert_loader_duration(path, sample["frames"], sample["sr"], sample["ch"], 0, sample["frames"])
    start_n = 5 * sample["sr"]
    dur_n = 10 * sample["sr"]
    assert_loader_duration(path, sample["frames"], sample["sr"], sample["ch"], start_n, dur_n)
    long_cut = next(item for item in items["dialog"] if item["frames"] > 700 * item["sr"])
    path = str(AUDIO / long_cut["rid"] / "dialog" / f"{long_cut['stem']}.m4a")
    assert_loader_duration(path, long_cut["frames"], long_cut["sr"], long_cut["ch"], 0, 20 * long_cut["sr"])


def write_records(items, selected, texts):
    sys.path.insert(0, "/222042021/lx/open-audio-llm-multi-target/src")
    sys.path.insert(0, "/222042021/lx/audio-data-contract/src")
    from audio_data_contract import AudioRecord
    from open_audio_llm.data.target_sot import EnrollmentConfig, eligible_candidates, parse_target_segments

    by_rid = defaultdict(list)
    for item in items["dialog"]:
        by_rid[item["rid"]].append(item)
    index = {}
    records_dir = OUT / "records"
    if records_dir.exists():
        for path in records_dir.glob("*.jsonl.gz"):
            path.unlink()
    records_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    components = {}
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"hours": 0.0, "records": 0, "native": 0.0, "cut": 0.0})))
    speakers = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    skipped = defaultdict(int)
    config = EnrollmentConfig(max_targets=5)
    examples = []

    def emit(kind, lang, k, record, duration, band, cut):
        name = f"k{k}_{kind}_{lang}"
        handle = handles.get(name)
        if handle is None:
            handle = gzip.open(records_dir / f"{name}.jsonl.gz", "wt", encoding="utf-8", compresslevel=1)
            handles[name] = handle
            components[name] = {"name": name, "k": k, "reps": 1, "language": lang, "group": kind, "records": 0, "hours": 0.0}
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        components[name]["records"] += 1
        components[name]["hours"] += duration / 3600.0
        bucket_stats = stats[kind][lang][band]
        bucket_stats["hours"] += duration / 3600.0
        bucket_stats["records"] += 1
        bucket_stats["cut" if cut else "native"] += duration / 3600.0
        count = record["labels"]["speaker_count"]
        key = "6+" if count >= 6 else str(count)
        speakers[kind][lang][key] += duration / 3600.0
        if len(examples) < 6:
            examples.append((kind, cut, record))
        if components[name]["records"] % 20000 == 0:
            print(f"wrote {name} {components[name]['records']}", flush=True)

    try:
        for item, start, end, band, cut in selected["dialog"]:
            segments = texts.get(("dialog", item["tar"], item["member"]))
            if not segments:
                skipped["dialog_text"] += 1
                continue
            if cut:
                snapped = snap_range(item["frames"], item["sr"], start, end, int(round(60 * item["sr"])))
            else:
                snapped = (0, item["frames"])
            if snapped is None:
                skipped["dialog_snap"] += 1
                continue
            start_n, end_n = snapped
            offset = start_n / item["sr"]
            duration = (end_n - start_n) / item["sr"]
            turns = turns_from_dialog(segments, offset, offset + duration, item["rid"])
            target, identities = render_turns(turns, duration)
            if not target or not identities:
                skipped["dialog_empty"] += 1
                continue
            add_index(index, "dialog", item)
            capped = dialog_candidates(item, offset, offset + duration, by_rid, items, index)
            present = set(identities.values())
            enrolled = present & set(capped)
            k = min(len(identities), 5) if enrolled else 0
            candidates = materialize_candidates(capped, index, "dialog") if k else []
            record_id = f"emilia2-dialog-{item['id']}" if not cut else f"emilia2-dialog-{item['id']}-w{start_n}"
            record = record_dict(record_id, item["lang"], duration, cut_id_of("dialog", item), offset, target, identities, candidates, k, 0.2 if k else 0.0)
            parsed = parse_target_segments(target, duration)
            if {row["speaker_id"] for row in parsed} != set(identities):
                raise RuntimeError(f"identity mismatch {record_id}")
            if k:
                groups = eligible_candidates(AudioRecord.from_dict(record), config)
                if not groups or not enrolled <= set(groups):
                    raise RuntimeError(f"enrollment mismatch {record_id}")
            emit("dialog", item["lang"], k, record, duration, band, cut)
        identity_cache = {}
        for item, start, end, band, cut in selected["long"]:
            segments = texts.get(("long", item["tar"], item["member"]))
            if not segments:
                skipped["long_text"] += 1
                continue
            if cut:
                snapped = snap_range(item["frames"], item["sr"], start, end, item["sr"])
            else:
                snapped = (0, item["frames"])
            if snapped is None:
                skipped["long_snap"] += 1
                continue
            start_n, end_n = snapped
            offset = start_n / item["sr"]
            duration = (end_n - start_n) / item["sr"]
            identity = f"emilia2_long:{item['rid']}:{item['stem']}"
            turns = turns_from_long(segments, offset, offset + duration, identity)
            target, identities = render_turns(turns, duration)
            if not target or list(identities.values()) != [identity]:
                skipped["long_empty"] += 1
                continue
            add_index(index, "long", item)
            k = 0
            candidates = []
            if cut:
                tail = item["frames"] / item["sr"] - (offset + duration)
                if tail >= 1.0:
                    k = 1
                    candidates = [{
                        "speaker_id": identity,
                        "partition": "train",
                        "single_speaker": True,
                        "ref": ref(cut_id_of("long", item), offset + duration, tail),
                    }]
            record_id = f"emilia2-long-{item['id']}" if not cut else f"emilia2-long-{item['id']}-w{start_n}"
            record = record_dict(record_id, item["lang"], duration, cut_id_of("long", item), offset, target, identities, candidates, k, 0.0)
            parse_target_segments(target, duration)
            if k:
                groups = eligible_candidates(AudioRecord.from_dict(record), config)
                if set(groups) != {identity}:
                    raise RuntimeError(f"long enrollment mismatch {record_id}")
            identity_cache[record_id] = True
            emit("long", item["lang"], k, record, duration, band, cut)
    finally:
        for handle in handles.values():
            handle.close()
    print("skipped", dict(skipped), flush=True)
    return index, components, stats, speakers, examples


def write_catalog(index, components, stats, speakers):
    index_path = OUT / "audio-index.jsonl.gz"
    print(f"writing audio index {len(index)}", flush=True)
    with gzip.open(index_path, "wt", encoding="utf-8", compresslevel=1) as handle:
        for cut_id in sorted(index):
            handle.write(json.dumps(index[cut_id], ensure_ascii=False, separators=(",", ":")) + "\n")
    artifacts = [{
        "name": "audio_index",
        "kind": "audio-index",
        "root_alias": "legacy_asr",
        "relative_path": index_path.resolve().relative_to(DATA_ROOT).as_posix(),
        "metadata": {"records": len(index)},
    }]
    names = []
    component_rows = []
    for name, spec in sorted(components.items()):
        path = OUT / "records" / f"{name}.jsonl.gz"
        artifact_name = f"records_{name}"
        names.append(artifact_name)
        artifacts.append({
            "name": artifact_name,
            "kind": "audio-records",
            "root_alias": "legacy_asr",
            "relative_path": path.resolve().relative_to(DATA_ROOT).as_posix(),
            "metadata": {"record_count": spec["records"]},
        })
        component_rows.append({
            **spec,
            "path": str(path),
            "hours": round(spec["hours"], 4),
            "train_hours": round(spec["hours"], 4),
        })
    catalog = {
        "schema_version": "dataset-catalog/1.0",
        "dataset_id": DS,
        "version": VER,
        "languages": ["zh", "en"],
        "tasks": ["speaker_attributed_asr"],
        "aliases": [],
        "artifacts": artifacts,
        "splits": {"train": {
            "records_artifacts": names,
            "audio_index_artifact": "audio_index",
            "statistics": {
                "records": sum(item["records"] for item in component_rows),
                "train_hours": round(sum(item["train_hours"] for item in component_rows), 4),
            },
        }},
        "recipe_parameters": {"components": component_rows},
        "provenance": {
            "description": "Emilia2 dialog and long mixture hours, counted once. reps is 1.",
        },
    }
    catalog_path = OUT / "catalog.jsonl"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False) + "\n")
    by_k, by_group = defaultdict(float), defaultdict(float)
    for item in component_rows:
        by_k[item["k"]] += item["train_hours"]
        by_group[item["group"]] += item["train_hours"]
    recipe = {
        "dataset_id": DS,
        "version": VER,
        "catalog": str(catalog_path),
        "roots": str(ROOTS),
        "audio_index": str(index_path),
        "notes": [
            "只生成了 dialog 和 long。会议、合成和 short 仍用 20260930 配方。",
            "每条记录写一次，reps 都是 1，时长不按注册重复计算。",
            "dialog 超过 10 分钟的文件从可用前缀切窗，文件末尾 60 秒不进任何窗。",
            "long 的 10–30 秒来自超过 10 分钟文件开头的一个窗口，其余留给这条 long 的注册。",
        ],
        "components": component_rows,
        "by_k": {str(k): round(by_k[k], 2) for k in sorted(by_k)},
        "by_group": {key: round(value, 2) for key, value in sorted(by_group.items())},
        "train_hours": round(sum(by_k.values()), 2),
    }
    (OUT / "recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2) + "\n")

    def freeze(tree):
        if isinstance(tree, defaultdict):
            return {key: freeze(value) for key, value in tree.items()}
        if isinstance(tree, dict):
            return {key: freeze(value) if isinstance(value, dict) else value for key, value in tree.items()}
        return tree

    summary = {
        "targets": {
            "dialog": {band[0]: band[3] for band in DIALOG_BANDS},
            "long": {band[0]: band[3] for band in LONG_BANDS},
        },
        "achieved": freeze(stats),
        "speakers": freeze(speakers),
        "by_k": recipe["by_k"],
        "by_group": recipe["by_group"],
        "train_hours": recipe["train_hours"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"by_k": recipe["by_k"], "by_group": recipe["by_group"], "train_hours": recipe["train_hours"]}, ensure_ascii=False), flush=True)


def verify_examples(examples):
    if len(examples) < 4:
        raise SystemExit(f"too few examples to verify: {len(examples)}")
    for _kind, _cut, record in examples[:4]:
        slot = record["audio_slots"][0]["ref"]
        # Resolved from the index file just written.
        sys.path.insert(0, "/222042021/lx/open-audio-llm-multi-target/src")
        sys.path.insert(0, "/222042021/lx/audio-data-contract/src")
        from open_audio_llm.data.catalog_dataset import load_mono
        from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver
        from open_audio_llm.data.target_sot import parse_target_segments
        from audio_data_contract import AudioRef

        resolver = LhotseCatalogAudioResolver(OUT / "catalog.jsonl", ROOTS)
        cut = resolver.get_cut(AudioRef.from_dict(slot))
        audio = load_mono(cut, 16000)
        from io import BytesIO
        import soundfile as sf
        buffer = BytesIO()
        sf.write(buffer, audio, 16000, format="WAV", subtype="FLOAT")
        got = sf.info(BytesIO(buffer.getvalue())).duration
        if abs(got - slot["duration"]) > 1 / 16000:
            raise SystemExit(f"record duration mismatch {record['id']}: {got} vs {slot['duration']}")
        parse_target_segments(record["target"], slot["duration"])
        print(f"record ok {record['id']} {slot['duration']:.3f}s k={record['labels']['recipe_k']}", flush=True)


def run_self_test():
    path = "/ai_sds_wuzz/DATA_ASR/emilia2/raw_data/audio/1ee0153ea12bf52d/long/1ee0153ea12bf52d_long_00002.m4a"
    header = probe_header(path)
    if header is None:
        raise SystemExit("self-test header failed")
    sample_rate, channels, frames = header
    assert_loader_duration(path, frames, sample_rate, channels, 0, frames)
    assert_loader_duration(path, frames, sample_rate, channels, 5 * sample_rate, 10 * sample_rate)
    print("self-test passed", flush=True)


def main():
    if "--self-test" in sys.argv:
        run_self_test()
        return
    if OUT.resolve() == (DATA_ROOT / "Derived/target-sot-recipe-20260930").resolve():
        raise SystemExit("refusing to write the published recipe")
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, str(Path(__file__).resolve()), "--self-test"])
    scan_all()
    items = load_items()
    summarize_supply(items)
    selected = plan(items)
    texts = load_texts(selected)
    check_duration_contract(items)
    index, components, stats, speakers, examples = write_records(items, selected, texts)
    write_catalog(index, components, stats, speakers)
    verify_examples(examples)
    for kind in ("dialog", "long"):
        for lang in ("zh", "en"):
            parts = []
            for band, _lo, _hi, target in (DIALOG_BANDS if kind == "dialog" else LONG_BANDS):
                got = stats[kind][lang][band]["hours"]
                parts.append(f"{band} {got:.1f}/{target:.0f}")
            print(f"final {kind} {lang}: {', '.join(parts)}", flush=True)


if __name__ == "__main__":
    main()
