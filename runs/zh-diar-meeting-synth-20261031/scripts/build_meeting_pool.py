"""Timeline templates and speech material for online meeting synthesis.

Training splits only; the zh245 holdout meetings (never trained on since unified-diarization-asr-20261001) are left out.

templates  Real meeting timelines: every annotated turn (speaker, start, end, characters) of AliMeeting train,
           AISHELL-4 train and AISHELL-5 train, taken from the timestamped meeting records and deduplicated across
           their window sizes (and, for AISHELL-5, across its four far microphones). Window records name a span of
           one template; training fills its turns with other speech (open_audio_llm.data.meeting_online).
material   Spontaneous single-speaker utterances cut at their annotated boundaries from per-speaker headset tracks
           (AliMeeting train near field, AISHELL-5 train DA microphones), shifted by the track's measured lag to the
           annotation. Each (meeting, speaker) becomes one FLAC of its utterances in order, MATERIAL_GAP s apart, so
           a run of consecutive utterances is one read. Every row records how much of it other speakers overlap in
           the source meeting (their speech leaks into a headset) and whether a teacher, not a person, labelled it.
"""
import argparse
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from amphiondata.multispeaker.farfield import envelope_lag

RATE = 16000
MATERIAL_GAP = 0.1  # silence between utterances inside a material track (s)
FADE = round(0.005 * RATE)  # edge fade of each cut utterance
# A lag beyond this, with a decisive correlation peak, means the headset track does not follow the annotation
# (the far-field simulation builder's rule, near-farsim-v1-20261009).
MAX_LAG, DECISIVE = 0.1, 0.02
WINDOWS = (30, 120, 300)  # window lengths (s), as the real meeting training splits (600 s left out: token budget)
CHARS = re.compile(r"[\W_]+")
SOURCES = {
    "alimeeting": ("alimeeting_sdm_meeting_sot", "real-meetings-v2-20261010", "alimeeting_sdm"),
    "aishell4": ("aishell4_meeting_sot", "real-meetings-v2-20261010", "aishell4"),
    "aishell5": ("aishell5_meeting_sot", "far-mic-v1-20261010", "aishell5"),
}


def chars(text):
    return len(CHARS.sub("", text))


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def meeting_turns(version_dir, prefix, holdout):
    """{meeting: sorted [(speaker, start, end, text, origin)]} on the source timeline, deduplicated."""
    turns = defaultdict(dict)
    for path in sorted(version_dir.glob(f"{prefix}-train_*s.jsonl.gz")):
        for record in read_rows(path):
            meta = record["metadata"]
            if meta["meeting_id"] in holdout:
                continue
            offset = record["audio_slots"][0]["ref"]["start"]
            for t in meta["turns"]:
                start, end = round(offset + t["start"], 4), round(offset + t["end"], 4)
                turns[meta["meeting_id"]][(t["source_speaker"], start, end)] = (t["text"], t.get("origin", "human"))
    return {m: sorted((s, a, b, text, origin) for (s, a, b), (text, origin) in rows.items())
            for m, rows in turns.items()}


def overlap_fraction(start, end, others):
    """Share of [start, end] covered by the union of ``others`` (sorted (start, end))."""
    covered, cursor = 0.0, start
    for a, b in others:
        if b <= cursor:
            continue
        if a >= end:
            break
        a = max(a, cursor)
        covered += min(b, end) - a
        cursor = min(b, end)
    return covered / (end - start) if end > start else 0.0


def cut_material(job):
    """Write one FLAC per speaker of a meeting and return its utterance rows."""
    meeting, corpus, tracks, turns, out = job["meeting"], job["corpus"], job["tracks"], job["turns"], Path(job["out"])
    by_speaker = defaultdict(list)
    for speaker, start, end, text, origin in turns:
        by_speaker[speaker].append((start, end, text, origin))
    rows, qa = [], {"meeting": meeting, "corpus": corpus, "speakers": {}}
    for speaker, own in sorted(by_speaker.items()):
        path = tracks.get(speaker)
        if path is None:
            qa["speakers"][speaker] = {"failed": "no headset track"}
            continue
        track, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == RATE, path
        track = np.ascontiguousarray(track[:, 0])
        lag, peak, at_zero = envelope_lag(track, [(a, b) for a, b, _, _ in own])
        if lag is None or (abs(lag) >= MAX_LAG and peak - at_zero > DECISIVE):
            qa["speakers"][speaker] = {"failed": f"lag {lag}", "peak": peak, "at_zero": at_zero}
            continue
        shift = lag if peak - at_zero > DECISIVE else 0.0
        others = sorted((a, b) for s, a, b, _, _ in turns if s != speaker)
        pieces, cursor, kept = [], 0, 0
        for index, (start, end, text, origin) in enumerate(own):
            a, b = round((start + shift) * RATE), round((end + shift) * RATE)
            if a < 0 or b > len(track) or b - a < 2 * FADE or not text.strip():
                continue
            clip = track[a:b].copy()
            clip[:FADE] *= np.linspace(0, 1, FADE, dtype=np.float32)
            clip[-FADE:] *= np.linspace(1, 0, FADE, dtype=np.float32)
            rows.append({"corpus": corpus, "meeting": meeting, "speaker": speaker, "index": index,
                         "offset": round(cursor / RATE, 5), "duration": round(len(clip) / RATE, 5),
                         "text": text, "chars": chars(text), "origin": origin,
                         "overlap": round(overlap_fraction(start, end, others), 3),
                         "source_start": start, "source_end": end})
            pieces += [clip, np.zeros(round(MATERIAL_GAP * RATE), dtype=np.float32)]
            cursor += len(clip) + round(MATERIAL_GAP * RATE)
            kept += 1
        if not kept:
            qa["speakers"][speaker] = {"failed": "no usable utterance"}
            continue
        relative = f"material/{corpus}/{meeting}/{speaker}.flac"
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, np.concatenate(pieces), RATE, format="FLAC", subtype="PCM_16")
        for row in rows:
            if row["speaker"] == speaker and "relative_path" not in row:
                row["relative_path"] = relative
        qa["speakers"][speaker] = {"lag_s": lag, "shift_s": shift, "utterances": kept, "seconds": round(cursor / RATE, 1)}
    return rows, qa


def alimeeting_tracks(near_dir, meeting, speakers):
    found = {}
    for speaker in speakers:
        paths = sorted(near_dir.glob(f"{meeting}_*_SPK{speaker}.wav"))
        if len(paths) == 1:
            found[speaker] = str(paths[0])
    return found


def aishell5_tracks(train_dir, meeting):
    """Headset (DA*) tracks of a session, keyed by the speaker named in each one's TextGrid tier."""
    session = train_dir / meeting.rsplit("-", 1)[1]
    found = {}
    for grid in sorted(session.glob("DA*.TextGrid")):
        raw = grid.read_bytes()
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8")
        names = re.findall(r'name = "([^"]+)"', text)
        if len(names) == 1 and grid.with_suffix(".wav").is_file():
            found[names[0]] = str(grid.with_suffix(".wav"))
    return found


def windows(meeting, corpus, turns, duration, length):
    """Window records over one template: half-overlapping spans, each holding only turns that lie inside it."""
    starts = list(np.arange(0.0, max(duration - length, 0.0) + 1e-6, length / 2))
    if duration > length and duration - length - starts[-1] > 1.0:
        starts.append(duration - length)
    for number, t0 in enumerate(starts):
        t1 = min(t0 + length, duration)
        inside = [[s, round(a - t0, 3), round(b - t0, 3), chars(text)] for s, a, b, text, _ in turns
                  if a >= t0 and b <= t1]
        if inside:
            yield number, round(float(t0), 3), round(float(t1 - t0), 3), inside


def write_jsonl_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", type=Path, default=Path("/workspace/data/datasets"))
    p.add_argument("--alimeeting-near", type=Path, required=True)
    p.add_argument("--aishell5-train", type=Path, required=True)
    p.add_argument("--holdout", type=Path, required=True, help="zh245-holdout.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset-id", default="meeting_synth_zh")
    p.add_argument("--version", required=True)
    p.add_argument("--root-alias", required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="first N meetings per corpus (smoke test)")
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    holdout = set(json.loads(args.holdout.read_text())["meetings"])

    from audio_data_contract import ArtifactRef, AudioRecord, AudioRef, AudioSlot, DatasetSpec, Split, write_records

    templates, durations = {}, {}
    for corpus, (dataset_id, version, prefix) in SOURCES.items():
        version_dir = args.datasets / dataset_id / "versions" / version
        for meeting, turns in meeting_turns(version_dir, prefix, holdout).items():
            templates[(corpus, meeting)] = turns
        for row in read_rows(version_dir / f"{prefix}-train-audio-index.jsonl.gz"):
            meeting = row["cut_id"].rsplit("_DX", 1)[0] if corpus == "aishell5" else row["cut_id"]
            durations[(corpus, meeting)] = row["num_frames"] / row["sample_rate"]
    if args.limit:
        templates = {k: v for k, v in templates.items()
                     if sorted(m for c, m in templates if c == k[0]).index(k[1]) < args.limit}
    print({c: sum(k[0] == c for k in templates) for c in SOURCES}, flush=True)

    jobs = []
    for (corpus, meeting), turns in sorted(templates.items()):
        speakers = sorted({t[0] for t in turns})
        if corpus == "alimeeting":
            tracks = alimeeting_tracks(args.alimeeting_near, meeting, speakers)
        elif corpus == "aishell5":
            tracks = aishell5_tracks(args.aishell5_train, meeting)
        else:
            continue
        jobs.append({"meeting": meeting, "corpus": corpus, "tracks": tracks, "turns": turns, "out": str(args.output)})
    material, qa = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for rows, report in pool.map(cut_material, jobs, chunksize=1):
            material += rows
            qa.append(report)
    print(f"material: {len(material)} utterances from {len(qa)} meetings", flush=True)

    write_jsonl_gz(args.output / "material.jsonl.gz", material)
    write_jsonl_gz(args.output / "templates.jsonl.gz", (
        {"corpus": c, "meeting": m, "duration": durations[(c, m)], "turns": [list(t[:3]) + [chars(t[3]), t[4]] for t in turns]}
        for (c, m), turns in sorted(templates.items())))
    # The training loader reads each track by its catalog audio-index entry.
    tracks = sorted({(r["corpus"], r["meeting"], r["speaker"], r["relative_path"]) for r in material})
    index_rows = []
    for corpus, meeting, speaker, relative in tracks:
        info = sf.info(args.output / relative)
        index_rows.append({"cut_id": f"{corpus}/{meeting}/{speaker}", "root_alias": args.root_alias,
                           "relative_path": relative, "sample_rate": RATE, "channels": 1,
                           "num_frames": info.frames, "duration": info.frames / RATE})
    write_jsonl_gz(args.output / "material-audio-index.jsonl.gz", index_rows)

    artifacts = [ArtifactRef("material", "jsonl-metadata", args.root_alias, "material.jsonl.gz"),
                 ArtifactRef("material-audio-index", "jsonl-metadata", args.root_alias, "material-audio-index.jsonl.gz"),
                 ArtifactRef("templates", "jsonl-metadata", args.root_alias, "templates.jsonl.gz")]
    splits = {"material": Split({"meeting_material": ("material",), "audio_index": ("material-audio-index",)})}
    counts = {}
    for length in WINDOWS:
        split = f"train_{length}s"
        records = []
        for (corpus, meeting), turns in sorted(templates.items()):
            for number, t0, span, inside in windows(meeting, corpus, turns, durations[(corpus, meeting)], length):
                records.append(AudioRecord(
                    f"{args.dataset_id}-{split}-{corpus}-{meeting}-{number:05d}", "speaker_attributed_asr",
                    (AudioSlot("mixture", AudioRef(args.dataset_id, args.version, split, f"template/{corpus}/{meeting}",
                                                   start=t0, duration=span)),),
                    "", language="zh",
                    labels={"speaker_count": len({t[0] for t in inside}), "target_seconds": length,
                            "template_corpus": corpus},
                    metadata={"meeting_template": {"corpus": corpus, "meeting": meeting, "start": t0, "turns": inside}}))
        write_records(records, args.output / f"{split}.jsonl.gz")
        artifacts.append(ArtifactRef(f"{split}-records", "audio-records", args.root_alias, f"{split}.jsonl.gz",
                                     metadata={"record_count": len(records)}))
        splits[split] = Split({"records": (f"{split}-records",)})
        counts[split] = Counter(r.labels["template_corpus"] for r in records)
    spec = DatasetSpec(args.dataset_id, args.version, ("zh",), ("speaker_attributed_asr",), tuple(artifacts), splits,
                       provenance={"builder": "runs/zh-diar-meeting-synth-20261031/scripts/build_meeting_pool.py",
                                   "sources": {c: f"{d}@{v}" for c, (d, v, _) in SOURCES.items()},
                                   "holdout_excluded": sorted(holdout)})
    (args.output / "catalog.jsonl").write_text(json.dumps(spec.to_dict(), ensure_ascii=False) + "\n")
    (args.output / "roots.json").write_text(json.dumps({args.root_alias: str(args.output)}) + "\n")

    hours = defaultdict(float)
    for r in material:
        hours[r["corpus"]] += r["duration"] / 3600
    report = {
        "templates": Counter(c for c, _ in templates),
        "template_hours": {c: round(sum(durations[k] for k in templates if k[0] == c) / 3600, 1) for c in SOURCES},
        "material_utterances": Counter(r["corpus"] for r in material),
        "material_hours": {k: round(v, 1) for k, v in hours.items()},
        "material_clean_hours": {c: round(sum(r["duration"] for r in material if r["corpus"] == c and r["overlap"] == 0) / 3600, 1)
                                 for c in hours},
        "windows": {k: dict(v) for k, v in counts.items()},
        "failed_tracks": [{"meeting": q["meeting"], "speaker": s, **v} for q in qa for s, v in q["speakers"].items()
                          if "failed" in v],
        "per_meeting": qa,
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_meeting"}, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
