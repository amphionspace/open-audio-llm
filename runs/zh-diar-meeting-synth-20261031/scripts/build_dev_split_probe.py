"""Why does the model hear extra speakers on re-voiced dev windows? Interventions on the windows of 2-3 speakers.

v1 near fills over-count with the size of the source conversation (2 voices: 22% from 2-person meetings, 73% from
4-person ones), and within a window leak-heavy utterances are not split more. So the cause sits at window level: faint
voices of people without a label, or the model recalling a training meeting it has memorized. All variants are close-talk:

  no_foreign     the v1 plan's voices, using only utterances nobody outside the window talks over in the source
  thinned        the v1 plan with utterances dropped at random down to no_foreign's seconds (same speech density, leak kept)
  unseen_src4    the fill with Eval speech (never trained on) from a 4-person meeting: the leak without the memory
  unseen_src2    the same from a 2-person meeting (windows of 2 speakers only): neither
  real_gated     the window's real speech through the same rendering (control)
  real_plus_leak real_gated with two strangers' real activity, -22 dB under each speaker and only while that speaker talks:
                 is the leak alone enough?

Record ids "<variant>/<panel id>"; per-window seconds of foreign leak in the catalog's split-probe-stats.jsonl."""
import argparse
import glob
import gzip
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from audio_data_contract import resolve_split

from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver
from open_audio_llm.data.meeting_online import (
    BACKCHANNEL_CHARS,
    CHARS_PATTERN,
    MaterialSpeaker,
    OnlineMeetingConfig,
    _start_cursor,
    fill_turns,
    load_material,
    meeting_target,
    plan_meeting,
    read_material_rows,
    render_meeting,
)

RATE = 16000
LEAK_DB = -22.0  # Median crosstalk of the AliMeeting material tracks (tasks/03-data-check attempt 001).
FADE = round(0.005 * RATE)  # scripts/build_meeting_pool.py cuts training material with these fades.
FOUR_PERSON = ("R8001_M8004", "R8003_M8001", "R8007_M8010", "R8007_M8011")
TWO_PERSON = ("R8009_M8018", "R8009_M8019", "R8009_M8020")
LINE = re.compile(r"\[(S\d+)\]\[(\d+\.\d\d)-(\d+\.\d\d)\] (.+)")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def merged_seconds(spans):
    total, end = 0.0, -1e9
    for a, b in sorted(spans):
        total += max(0.0, b - max(a, end))
        end = max(end, b)
    return total


def foreign_seconds(placed, row_of, by_meeting):
    keys = {item[3].key for item in placed}
    spans = []
    for start, _, _, voice, index in placed:
        row = row_of(voice, index)
        for o in by_meeting[row["meeting"]]:
            if o["speaker"] != row["speaker"] and (o["corpus"], o["meeting"], o["speaker"]) not in keys:
                a, b = max(o["source_start"], row["source_start"]), min(o["source_end"], row["source_end"])
                if a < b:
                    spans.append((start + a - row["source_start"], start + b - row["source_start"]))
    return merged_seconds(spans)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-records", type=Path, required=True)
    p.add_argument("--near-dir", type=Path, required=True)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--catalog-dir", type=Path, required=True)
    p.add_argument("--dataset-id", default="meeting_synth_fidelity")
    p.add_argument("--version", required=True)
    p.add_argument("--seed", default="fidelity-20261031")
    args = p.parse_args()

    data = yaml.safe_load(args.data_config.read_text())
    config = replace(OnlineMeetingConfig(**data["online_meeting"]), corpus_weights={"alimeeting": 1.0})
    near_config = replace(config, dry_prob=1.0, snr_db=(80.0, 80.0))
    resolver = LhotseCatalogAudioResolver(data["catalog"], data["roots"])
    m = config.material
    tracks = {}
    for path in resolve_split(resolver.catalog, m["dataset_id"], m["version"], m["split"], "audio_index", resolver.roots):
        for row in read_material_rows([path]):
            tracks[row["cut_id"]] = str(Path(resolver.roots[row["root_alias"]]) / row["relative_path"])
    train_rows = [r for r in read_material_rows(resolve_split(
        resolver.catalog, m["dataset_id"], m["version"], m["split"], "meeting_material", resolver.roots)) if r["corpus"] == "alimeeting"]
    track_of = lambda row: tracks[f"{row['corpus']}/{row['meeting']}/{row['speaker']}"]  # noqa: E731
    material = load_material(train_rows, config, track_of)

    records = [json.loads(line) for line in gzip.open(args.panel_records, "rt")]
    # Eval material: the panel's turns on the source timeline, one row per turn.
    by_source = defaultdict(dict)
    for r in records:
        first = r["metadata"]["source_record"]["start"]
        for t in r["metadata"]["turns"]:
            a, b = round(first + t["start"], 4), round(first + t["end"], 4)
            by_source[r["metadata"]["meeting_id"]][(t["source_speaker"], a, b)] = t["text"]
    eval_rows = []
    for meeting, turns in by_source.items():
        counts = defaultdict(int)
        for (speaker, a, b), text in sorted(turns.items()):
            eval_rows.append({"corpus": "eval", "meeting": meeting, "speaker": speaker, "index": counts[speaker], "offset": a,
                              "duration": round(b - a, 5), "text": text, "chars": len(CHARS_PATTERN.sub("", text)),
                              "origin": "human", "overlap": 0.0, "source_start": a, "source_end": b})
            counts[speaker] += 1
    near_path = lambda meeting, speaker: glob.glob(f"{args.near_dir}/{meeting}_N_SPK{speaker}.wav")[0]  # noqa: E731
    eval_track = lambda row: near_path(row["meeting"], row["speaker"])  # noqa: E731
    eval_config = replace(config, corpus_weights={"eval": 1.0})
    eval_material = load_material(eval_rows, eval_config, eval_track)["eval"]
    conversation = {c[0].key[1]: c for c in eval_material}

    rows_by_meeting = defaultdict(list)
    for r in train_rows + eval_rows:
        rows_by_meeting[r["meeting"]].append(r)
    voice_rows = {}
    for r in train_rows + eval_rows:
        voice_rows.setdefault((r["corpus"], r["meeting"], r["speaker"]), []).append(r)
    for rows in voice_rows.values():
        rows.sort(key=lambda r: r["index"])
    row_of = lambda voice, index: voice_rows[voice.key][int(np.searchsorted(voice_starts[voice.key], voice.starts[index]))]  # noqa: E731
    voice_starts = {k: np.array([r["source_start"] for r in rows]) for k, rows in voice_rows.items()}

    def filtered_voice(voice, keys):
        kept = []
        for row in voice_rows[voice.key]:
            leak = any(o["speaker"] != row["speaker"] and (o["corpus"], o["meeting"], o["speaker"]) not in keys
                       and o["source_start"] < row["source_end"] and row["source_start"] < o["source_end"]
                       for o in rows_by_meeting[row["meeting"]])
            if not leak:
                kept.append(row)
        return MaterialSpeaker(voice.key, voice.track, kept, config) if kept else None

    def clip_of(voice, index, duration):
        a = round(float(voice.offsets[index]) * RATE)
        clip = sf.read(voice.track, start=a, frames=round(duration * RATE), dtype="float32", always_2d=True)[0][:, 0].copy()
        if voice.key[0] == "eval" and len(clip) >= 2 * FADE:
            clip[:FADE] *= np.linspace(0, 1, FADE, dtype=np.float32)
            clip[-FADE:] *= np.linspace(1, 0, FADE, dtype=np.float32)
        return clip

    def render_placed(placed, length, rng):
        dry, spans = defaultdict(lambda: np.zeros(length, dtype=np.float32)), defaultdict(list)
        for begin, duration, speaker, voice, index in placed:
            clip = clip_of(voice, index, duration)
            b = round(begin * RATE)
            clip = clip[: length - b]
            dry[speaker][b:b + len(clip)] = clip
            spans[speaker].append((b, b + len(clip)))
        names = sorted(dry)
        return render_meeting(rng, [dry[n] for n in names], [spans[n] for n in names], length, near_config, rooms=[],
                              load_rir=None, noise_loader=lambda _: np.zeros(RATE, dtype=np.float32), sampling_rate=RATE)

    headsets = {}

    def headset(meeting, speaker):
        if (meeting, speaker) not in headsets:
            headsets[(meeting, speaker)] = sf.read(near_path(meeting, speaker), dtype="float32", always_2d=True)[0][:, 0]
        return headsets[(meeting, speaker)]

    def window_of(track, first, length):
        piece = track[first:first + length]
        return np.pad(piece, (0, max(0, length - len(piece))))

    variants = ["no_foreign", "thinned", "unseen_src4", "unseen_src2", "real_gated", "real_plus_leak"]
    for variant in variants:
        (args.output / variant).mkdir(parents=True, exist_ok=True)
    out_records, index_rows, stats = defaultdict(list), [], []
    for r in records:
        meeting, rid = r["metadata"]["meeting_id"], r["id"]
        turns = sorted(r["metadata"]["turns"], key=lambda t: t["start"])
        count = len({t["source_speaker"] for t in turns})
        if count not in (2, 3):
            continue
        length = round(r["audio_slots"][0]["ref"]["duration"] * RATE)
        template = [[t["source_speaker"], t["start"], t["end"], len(CHARS_PATTERN.sub("", t["text"]))] for t in turns]
        by_speaker = defaultdict(list)
        for speaker, a, b, chars in template:
            by_speaker[str(speaker)].append((float(a), float(b), int(chars)))
        audio = {}
        info = {"id": rid, "speakers": count}
        # The plan as built in v1.
        _, base = plan_meeting(random.Random(f"{args.seed}:fill:alimeeting:{rid}"), template, length / RATE, material, config)
        keys = {item[3].key for item in base}
        info["v1_seconds"] = round(sum(item[1] for item in base), 2)
        info["v1_foreign"] = round(foreign_seconds(base, row_of, rows_by_meeting), 2)
        # no_foreign: same voices, leak-free utterances, same part of the conversation.
        rng = random.Random(f"{args.seed}:no_foreign:{rid}")
        clean = []
        for speaker in by_speaker:
            mine = [item for item in base if item[2] == speaker]
            if not mine:
                continue
            voice = filtered_voice(mine[0][3], keys)
            if voice is None or not voice.eligible.any():
                continue
            cursor = _start_cursor(voice, float(mine[0][3].starts[mine[0][4]]))
            for start, index in fill_turns(rng, sorted(by_speaker[speaker]), length / RATE, voice, cursor, config):
                clean.append((start, float(voice.durations[index]), speaker, voice, index))
        clean.sort(key=lambda item: (item[0], item[2]))
        info["no_foreign_seconds"] = round(sum(item[1] for item in clean), 2)
        info["no_foreign_voices"] = len({item[2] for item in clean})
        # thinned: the v1 plan down to the same seconds, every speaker keeping an utterance.
        rng = random.Random(f"{args.seed}:thinned:{rid}")
        kept = list(base)
        order = list(range(len(base)))
        rng.shuffle(order)
        seconds = sum(item[1] for item in base)
        target = sum(item[1] for item in clean)
        dropped = set()
        left = defaultdict(int)
        for item in base:
            left[item[2]] += 1
        for k in order:
            if seconds <= target:
                break
            if left[base[k][2]] > 1:
                dropped.add(k)
                left[base[k][2]] -= 1
                seconds -= base[k][1]
        kept = [item for k, item in enumerate(base) if k not in dropped]
        info["thinned_seconds"] = round(seconds, 2)
        info["thinned_foreign"] = round(foreign_seconds(kept, row_of, rows_by_meeting), 2)
        plans = {"no_foreign": clean, "thinned": kept}
        # Unseen speech: another meeting of Eval.
        rng = random.Random(f"{args.seed}:unseen_src4:{rid}")
        sources = {"unseen_src4": FOUR_PERSON}
        if count == 2:
            sources["unseen_src2"] = TWO_PERSON
        for variant, names in sources.items():
            pool = {"eval": [conversation[n] for n in names if n != meeting]}
            _, plan = plan_meeting(random.Random(f"{args.seed}:{variant}:{rid}"), template, length / RATE, pool, eval_config)
            plans[variant] = plan
            info[f"{variant}_seconds"] = round(sum(item[1] for item in plan), 2)
            info[f"{variant}_foreign"] = round(foreign_seconds(plan, row_of, rows_by_meeting), 2)
        for variant, plan in plans.items():
            signal = render_placed(plan, length, random.Random(f"{args.seed}:render:{variant}:{rid}"))
            audio[variant] = (signal, meeting_target(plan)[0])
        # The real window, with and without strangers.
        first = round(r["metadata"]["source_record"]["start"] * RATE)
        dry, spans = [], []
        for speaker in sorted(by_speaker):
            own = [(round(a * RATE), min(round(b * RATE), length)) for a, b, _ in by_speaker[speaker]]
            window = window_of(headset(meeting, speaker), first, length)
            gated = np.zeros(length, dtype=np.float32)
            for begin, end in own:
                gated[begin:end] = window[begin:end]
            dry.append(gated)
            spans.append(own)
        audio["real_gated"] = (render_meeting(random.Random(f"{args.seed}:render:real_gated:{rid}"), dry, spans, length, near_config,
                                              rooms=[], load_rir=None, noise_loader=lambda _: np.zeros(RATE, dtype=np.float32),
                                              sampling_rate=RATE), r["target"])
        # Two strangers talking as they really did in another 4-person meeting; each headset hears them while its owner talks.
        rng = random.Random(f"{args.seed}:real_plus_leak:{rid}")
        stranger_meeting = rng.choice([n for n in FOUR_PERSON if n != meeting])
        strangers = rng.sample(conversation[stranger_meeting], 2)
        offset = rng.randrange(max(1, min(len(headset(*s.key[1:])) for s in strangers) - length))
        stranger = np.zeros(length, dtype=np.float32)
        stranger_mask = np.zeros(length, dtype=bool)
        for s in strangers:
            track = window_of(headset(*s.key[1:]), offset, length)
            for a, b in zip(s.starts, s.starts + s.durations):
                x, y = max(round(a * RATE) - offset, 0), min(round(b * RATE) - offset, length)
                if x < y:
                    stranger[x:y] += track[x:y]
                    stranger_mask[x:y] = True
        leaky, heard = [], 0
        for gated, own in zip(dry, spans):
            mask = np.zeros(length, dtype=bool)
            for begin, end in own:
                mask[begin:end] = True
            voiced = mask & stranger_mask
            own_level = float(np.sqrt(np.mean(gated[mask].astype(np.float64) ** 2))) if mask.any() else 0.0
            level = float(np.sqrt(np.mean(stranger[voiced].astype(np.float64) ** 2))) if voiced.any() else 0.0
            if level > 0 and own_level > 0:
                leaky.append(gated + np.where(mask, stranger, 0.0).astype(np.float32) * (own_level * 10 ** (LEAK_DB / 20) / level))
                heard += int(voiced.sum())
            else:
                leaky.append(gated)
        info["real_plus_leak_heard_seconds"] = round(heard / RATE, 2)
        audio["real_plus_leak"] = (render_meeting(random.Random(f"{args.seed}:render:real_plus_leak:{rid}"), leaky, spans, length, near_config,
                                                  rooms=[], load_rir=None, noise_loader=lambda _: np.zeros(RATE, dtype=np.float32),
                                                  sampling_rate=RATE), r["target"])
        stats.append(info)
        for variant, (signal, target) in audio.items():
            cut = f"{variant}/{rid}"
            sf.write(args.output / f"{cut}.wav", signal, RATE, subtype="PCM_16")
            row = json.loads(json.dumps(r))
            row["id"] = cut
            row["audio_slots"][0]["ref"].update({"dataset_id": args.dataset_id, "version": args.version, "split": variant, "cut_id": cut,
                                                 "start": 0.0, "duration": len(signal) / RATE})
            row["target"] = target
            if target != r["target"]:
                row["metadata"]["turns"] = [
                    {"source_id": f"{cut}-{k}", "speaker": label, "source_speaker": label, "start": float(a), "end": float(b), "text": text}
                    for k, (label, a, b, text) in enumerate(LINE.fullmatch(line).groups() for line in target.splitlines())]
            out_records[variant].append(row)
            index_rows.append({"channels": 1, "cut_id": cut, "duration": len(signal) / RATE, "num_frames": len(signal),
                               "relative_path": f"{cut}.wav", "root_alias": "meeting_synth_fidelity", "sample_rate": RATE})
        print(rid, json.dumps(info), flush=True)

    cat = args.catalog_dir
    (cat / "records").mkdir(parents=True, exist_ok=True)
    (cat / "eval-set").mkdir(parents=True, exist_ok=True)
    with gzip.open(cat / "audio-index.jsonl.gz", "wt") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in index_rows)
    artifacts = [{"name": "audio-index", "kind": "audio-index", "root_alias": "meeting_synth_fidelity_catalog",
                  "relative_path": "audio-index.jsonl.gz", "sha256": sha256_file(cat / "audio-index.jsonl.gz")}]
    splits = {}
    for variant, rows in out_records.items():
        path = cat / "records" / f"{variant}.jsonl.gz"
        with gzip.open(path, "wt") as stream:
            stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        artifacts.append({"name": f"{variant}-records", "kind": "audio-records", "root_alias": "meeting_synth_fidelity_catalog",
                          "relative_path": f"records/{variant}.jsonl.gz", "sha256": sha256_file(path), "metadata": {"record_count": len(rows)}})
        splits[variant] = {"artifacts": {"records": [f"{variant}-records"], "audio_index": ["audio-index"]},
                           "statistics": {"records": len(rows)}}
    entry = {"schema_version": "dataset-catalog/2.0", "dataset_id": args.dataset_id, "version": args.version, "languages": ["zh"],
             "tasks": ["speaker_attributed_asr"], "aliases": [], "artifacts": artifacts, "splits": splits,
             "provenance": {"builder": "runs/zh-diar-meeting-synth-20261031/scripts/build_dev_split_probe.py", "usage": "evaluation_only",
                            "online_meeting": data["online_meeting"]}}
    (cat / "catalog.yaml").write_text(yaml.safe_dump([entry], sort_keys=False, allow_unicode=True))
    (cat / "roots.json").write_text(json.dumps({"meeting_synth_fidelity": str(args.output.resolve()),
                                                "meeting_synth_fidelity_catalog": str(cat.resolve())}, indent=1) + "\n")
    eval_set = {"task": "diarization", "tags": ["meeting", "multi-speaker", "timestamps"], "sampling_rate": RATE,
                "groups": [{"name": v, "section": "多数人排查", "labels": [v], "aggregate": "macro", "description": v} for v in out_records],
                "evaluation": [{"dataset_id": args.dataset_id, "version": args.version, "split": v, "task": "speaker_attributed_asr", "label": v,
                                "tags": ["zh", "meeting", "multi-speaker", "timestamps"], "display": v, "description": v} for v in out_records]}
    (cat / "eval-set" / "eval-set.yaml").write_text(yaml.safe_dump(eval_set, sort_keys=False, allow_unicode=True))
    with (cat / "split-probe-stats.jsonl").open("w") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in stats)
    print(json.dumps({v: len(rows) for v, rows in out_records.items()}))


if __name__ == "__main__":
    main()
