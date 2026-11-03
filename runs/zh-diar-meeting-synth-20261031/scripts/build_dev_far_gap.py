"""Why is text worse on far re-voiced windows (no-overlap 10.2) than on the same dev speech through the same acoustics
(syn_acoustic 6.3), when close-talk differs far less (6.2 against 4.3)? Both variants take the 138 dev windows through the
synthesizer's room, levels and noise (scripts/build_dev_fidelity.py with synth-v1-fidelity-alimeeting.yaml):

  shuffled_far  each turn swapped for the same speaker's closest-duration turn from elsewhere in the meeting, placed at the
                original start: real voices and layout, no exchange between speakers. near_shuffled was 5.8 close-talk.
  unseen_far    the fill with Eval speech (never trained on) from another meeting: fill machinery without seen material.

Record ids "<variant>/<panel id>"."""
import argparse
import glob
import gzip
import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from open_audio_llm.data.catalog_dataset import choose_from_groups, load_mono, load_rir_item, source_weights
from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver
from open_audio_llm.data.meeting_online import (
    CHARS_PATTERN,
    OnlineMeetingConfig,
    group_rooms,
    load_material,
    meeting_target,
    plan_meeting,
    render_meeting,
)

RATE = 16000
FADE = round(0.005 * RATE)  # scripts/build_meeting_pool.py cuts training material with these fades.
LINE = re.compile(r"\[(S\d+)\]\[(\d+\.\d\d)-(\d+\.\d\d)\] (.+)")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
    config = replace(OnlineMeetingConfig(**data["online_meeting"]), corpus_weights={"eval": 1.0})
    resolver = LhotseCatalogAudioResolver(data["catalog"], data["roots"])
    items = [item for s in config.rir_sources for item in resolver.iter_recordings(s["dataset_id"], s["version"], s["split"])]
    rooms = group_rooms(items, lambda item: item.id, config.min_room_positions)
    noise_groups = [list(resolver.iter_recordings(s["dataset_id"], s["version"], s["split"])) for s in config.noise_sources]
    weights = source_weights(list(config.noise_sources))
    noise = lambda r: load_mono(choose_from_groups(r, noise_groups, weights), RATE)  # noqa: E731
    load_rir = lambda item: load_rir_item(item, RATE)  # noqa: E731

    records = [json.loads(line) for line in gzip.open(args.panel_records, "rt")]
    by_source = defaultdict(dict)
    for r in records:
        first = r["metadata"]["source_record"]["start"]
        for t in r["metadata"]["turns"]:
            by_source[r["metadata"]["meeting_id"]][(t["source_speaker"], round(first + t["start"], 4), round(first + t["end"], 4))] = (t["text"], r["id"])
    eval_rows = []
    for meeting, turns in by_source.items():
        counts = defaultdict(int)
        for (speaker, a, b), (text, _) in sorted(turns.items()):
            eval_rows.append({"corpus": "eval", "meeting": meeting, "speaker": speaker, "index": counts[speaker], "offset": a,
                              "duration": round(b - a, 5), "text": text, "chars": len(CHARS_PATTERN.sub("", text)),
                              "origin": "human", "overlap": 0.0, "source_start": a, "source_end": b})
            counts[speaker] += 1
    near_path = lambda meeting, speaker: glob.glob(f"{args.near_dir}/{meeting}_N_SPK{speaker}.wav")[0]  # noqa: E731
    conversations = load_material(eval_rows, config, lambda row: near_path(row["meeting"], row["speaker"]))["eval"]
    headsets = {}

    def headset(meeting, speaker):
        if (meeting, speaker) not in headsets:
            headsets[(meeting, speaker)] = sf.read(near_path(meeting, speaker), dtype="float32", always_2d=True)[0][:, 0]
        return headsets[(meeting, speaker)]

    def cut(track, a, frames):
        clip = track[a:a + frames].copy()
        if len(clip) >= 2 * FADE:
            clip[:FADE] *= np.linspace(0, 1, FADE, dtype=np.float32)
            clip[-FADE:] *= np.linspace(1, 0, FADE, dtype=np.float32)
        return clip

    variants = ("shuffled_far", "unseen_far")
    for variant in variants:
        (args.output / variant).mkdir(parents=True, exist_ok=True)
    out_records, index_rows = defaultdict(list), []
    for r in records:
        meeting, rid = r["metadata"]["meeting_id"], r["id"]
        length = round(r["audio_slots"][0]["ref"]["duration"] * RATE)
        turns = sorted(r["metadata"]["turns"], key=lambda t: t["start"])
        audio = {}
        # shuffled_far: same speakers and places, other turns of the meeting. A turn stops before the speaker's next one.
        rng = random.Random(f"{args.seed}:shuffled_far:{rid}")
        pool = defaultdict(list)
        for (speaker, a, b), (text, window) in by_source[meeting].items():
            if window != rid:
                pool[speaker].append((b - a, a, text))
        dry, spans, lines, used = defaultdict(lambda: np.zeros(length, dtype=np.float32)), defaultdict(list), [], set()
        for k, t in enumerate(turns):
            speaker = t["source_speaker"]
            limit = min([u["start"] for u in turns[k + 1:] if u["source_speaker"] == speaker] + [length / RATE]) - t["start"] - 0.05
            want = t["end"] - t["start"]
            options = [c for c in pool[speaker] if c[1] not in used and c[0] <= limit] or [c for c in pool[speaker] if c[0] <= limit]
            if not options:
                continue
            duration, a, text = min(options, key=lambda c: abs(c[0] - want))
            used.add(a)
            clip = cut(headset(meeting, speaker), round(a * RATE), round(duration * RATE))
            begin = round(t["start"] * RATE)
            clip = clip[: length - begin]
            dry[speaker][begin:begin + len(clip)] = clip
            spans[speaker].append((begin, begin + len(clip)))
            lines.append((t["start"], t["start"] + duration, speaker, text))
        lines.sort()
        labels = {}
        for _, _, speaker, _ in lines:
            labels.setdefault(speaker, f"S{len(labels) + 1}")
        target = "\n".join(f"[{labels[s]}][{a:.2f}-{b:.2f}] {text}" for a, b, s, text in lines if round(a, 2) < round(b, 2))
        names = sorted(dry)
        audio["shuffled_far"] = (render_meeting(random.Random(f"{args.seed}:render:shuffled_far:{rid}"), [dry[n] for n in names],
                                                [spans[n] for n in names], length, config, rooms=rooms, load_rir=load_rir,
                                                noise_loader=noise, sampling_rate=RATE), target)
        # unseen_far: the fill from another meeting of Eval.
        template = [[t["source_speaker"], t["start"], t["end"], len(CHARS_PATTERN.sub("", t["text"]))] for t in turns]
        others = [c for c in conversations if c[0].key[1] != meeting]
        _, placed = plan_meeting(random.Random(f"{args.seed}:unseen_far:{rid}"), template, length / RATE, {"eval": others}, config)
        dry, spans = defaultdict(lambda: np.zeros(length, dtype=np.float32)), defaultdict(list)
        for begin, duration, speaker, voice, index in placed:
            clip = cut(headset(*voice.key[1:]), round(float(voice.offsets[index]) * RATE), round(duration * RATE))
            b = round(begin * RATE)
            clip = clip[: length - b]
            dry[speaker][b:b + len(clip)] = clip
            spans[speaker].append((b, b + len(clip)))
        names = sorted(dry)
        audio["unseen_far"] = (render_meeting(random.Random(f"{args.seed}:render:unseen_far:{rid}"), [dry[n] for n in names],
                                              [spans[n] for n in names], length, config, rooms=rooms, load_rir=load_rir,
                                              noise_loader=noise, sampling_rate=RATE), meeting_target(placed)[0])
        for variant, (signal, target) in audio.items():
            name = f"{variant}/{rid}"
            sf.write(args.output / f"{name}.wav", signal, RATE, subtype="PCM_16")
            row = json.loads(json.dumps(r))
            row["id"] = name
            row["audio_slots"][0]["ref"].update({"dataset_id": args.dataset_id, "version": args.version, "split": variant, "cut_id": name,
                                                 "start": 0.0, "duration": len(signal) / RATE})
            row["target"] = target
            row["metadata"]["turns"] = [
                {"source_id": f"{name}-{k}", "speaker": label, "source_speaker": label, "start": float(a), "end": float(b), "text": text}
                for k, (label, a, b, text) in enumerate(LINE.fullmatch(line).groups() for line in target.splitlines())]
            out_records[variant].append(row)
            index_rows.append({"channels": 1, "cut_id": name, "duration": len(signal) / RATE, "num_frames": len(signal),
                               "relative_path": f"{name}.wav", "root_alias": "meeting_synth_fidelity", "sample_rate": RATE})
        print(rid, flush=True)

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
             "provenance": {"builder": "runs/zh-diar-meeting-synth-20261031/scripts/build_dev_far_gap.py", "usage": "evaluation_only",
                            "online_meeting": data["online_meeting"]}}
    (cat / "catalog.yaml").write_text(yaml.safe_dump([entry], sort_keys=False, allow_unicode=True))
    (cat / "roots.json").write_text(json.dumps({"meeting_synth_fidelity": str(args.output.resolve()),
                                                "meeting_synth_fidelity_catalog": str(cat.resolve())}, indent=1) + "\n")
    eval_set = {"task": "diarization", "tags": ["meeting", "multi-speaker", "timestamps"], "sampling_rate": RATE,
                "groups": [{"name": v, "section": "远场文字差距排查", "labels": [v], "aggregate": "macro", "description": v} for v in out_records],
                "evaluation": [{"dataset_id": args.dataset_id, "version": args.version, "split": v, "task": "speaker_attributed_asr", "label": v,
                                "tags": ["zh", "meeting", "multi-speaker", "timestamps"], "display": v, "description": v} for v in out_records]}
    (cat / "eval-set" / "eval-set.yaml").write_text(yaml.safe_dump(eval_set, sort_keys=False, allow_unicode=True))
    print(json.dumps({v: len(rows) for v, rows in out_records.items()}))


if __name__ == "__main__":
    main()
