"""Fidelity check of the online meeting synthesizer on the 138 AliMeeting dev selection-panel windows.

Every variant runs the training code (open_audio_llm.data.meeting_online, with the configuration under test and the
training material, rooms and noise) on the dev windows, so each can be scored against a real counterpart:

  syn_acoustic         the real dev speech and timing (headsets gated to their own turns) through the synthesizer's
                       room, levels and noise; reference unchanged. Real counterpart: far_mean (the panel audio).
  syn_fill_near_<c>    the dev windows' real timelines voiced with training material of corpus <c>, close-talk
                       (no room, no noise); new reference. Real counterpart: near_gated (same timing, real speech).
  syn_fill_far_<c>     the same plan (same voices and words) through the room, levels and noise; new reference.
                       Real counterpart: far_mean.

Record ids "<variant>/<panel id>" (AmphionEval needs unique ids)."""
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
from audio_data_contract import resolve_split

from open_audio_llm.data.catalog_dataset import choose_from_groups, load_mono, load_rir_item, source_weights
from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver
from open_audio_llm.data.meeting_online import (
    CHARS_PATTERN,
    OnlineMeetingConfig,
    group_rooms,
    load_material,
    meeting_target,
    plan_meeting,
    read_material_rows,
    render_meeting,
)

RATE = 16000
LINE = re.compile(r"\[(S\d+)\]\[(\d+\.\d\d)-(\d+\.\d\d)\] (.+)")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gate(track, spans, length):
    out = np.zeros(length, dtype=np.float32)
    for begin, end in spans:
        out[begin:end] = track[begin:end]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-records", type=Path, required=True)
    p.add_argument("--near-dir", type=Path, required=True)
    p.add_argument("--data-config", type=Path, required=True, help="YAML whose online_meeting block is under test")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--catalog-dir", type=Path, required=True)
    p.add_argument("--dataset-id", default="meeting_synth_fidelity")
    p.add_argument("--version", required=True)
    p.add_argument("--corpora", nargs="+", default=["alimeeting", "aishell5"])
    p.add_argument("--seed", default="fidelity-20261031")
    args = p.parse_args()

    data = yaml.safe_load(args.data_config.read_text())
    config = OnlineMeetingConfig(**data["online_meeting"])
    resolver = LhotseCatalogAudioResolver(data["catalog"], data["roots"])
    m = config.material
    tracks = {}
    for path in resolve_split(resolver.catalog, m["dataset_id"], m["version"], m["split"], "audio_index",
                              resolver.roots):
        for row in read_material_rows([path]):
            tracks[row["cut_id"]] = str(Path(resolver.roots[row["root_alias"]]) / row["relative_path"])
    material_rows = list(read_material_rows(resolve_split(
        resolver.catalog, m["dataset_id"], m["version"], m["split"], "meeting_material", resolver.roots)))
    track_of = lambda row: tracks[f"{row['corpus']}/{row['meeting']}/{row['speaker']}"]  # noqa: E731
    materials = {c: load_material(material_rows, replace(config, corpus_weights={c: 1.0}), track_of)
                 for c in args.corpora}
    items = [item for s in config.rir_sources for item in resolver.iter_recordings(s["dataset_id"], s["version"], s["split"])]
    rooms = group_rooms(items, lambda item: item.id, config.min_room_positions)
    noise_groups = [list(resolver.iter_recordings(s["dataset_id"], s["version"], s["split"])) for s in config.noise_sources]
    weights = source_weights(list(config.noise_sources))
    noise = lambda r: load_mono(choose_from_groups(r, noise_groups, weights), RATE)  # noqa: E731
    load_rir = lambda item: load_rir_item(item, RATE)  # noqa: E731
    near_config = replace(config, dry_prob=1.0, snr_db=(80.0, 80.0))

    records = [json.loads(line) for line in gzip.open(args.panel_records, "rt")]
    variants = ["syn_acoustic"] + [f"syn_fill_{kind}_{c}" for c in args.corpora for kind in ("near", "far")]
    for variant in variants:
        (args.output / variant).mkdir(parents=True, exist_ok=True)
    headsets = {}
    out_records, index_rows, stats = defaultdict(list), [], defaultdict(list)
    for r in records:
        meeting, rid = r["metadata"]["meeting_id"], r["id"]
        start = r["metadata"]["source_record"]["start"]
        length = round(r["audio_slots"][0]["ref"]["duration"] * RATE)
        turns = sorted(r["metadata"]["turns"], key=lambda t: t["start"])
        by_speaker = defaultdict(list)
        for t in turns:
            by_speaker[t["source_speaker"]].append((round(t["start"] * RATE), min(round(t["end"] * RATE), length)))
        audio = {}
        # syn_acoustic: real headsets, gated, through the synthesizer's acoustics.
        dry, spans = [], []
        for speaker, own in sorted(by_speaker.items()):
            if (meeting, speaker) not in headsets:
                (path,) = glob.glob(f"{args.near_dir}/{meeting}_N_SPK{speaker}.wav")
                headsets[(meeting, speaker)] = sf.read(path, dtype="float32", always_2d=True)[0][:, 0]
            track = headsets[(meeting, speaker)]
            first = round(start * RATE)
            window = np.pad(track[first:first + length], (0, max(0, length - len(track[first:first + length]))))
            dry.append(gate(window, own, length))
            spans.append(own)
        rng = random.Random(f"{args.seed}:syn_acoustic:{rid}")
        info = {}
        audio["syn_acoustic"] = (render_meeting(rng, dry, spans, length, config, rooms=rooms, load_rir=load_rir,
                                                noise_loader=noise, sampling_rate=RATE, info=info), r["target"])
        stats["syn_acoustic"].append(info)
        template = [[t["source_speaker"], t["start"], t["end"], len(CHARS_PATTERN.sub("", t["text"]))] for t in turns]
        for corpus in args.corpora:
            rng = random.Random(f"{args.seed}:fill:{corpus}:{rid}")
            _, placed = plan_meeting(rng, template, length / RATE, materials[corpus], config)
            target, _ = meeting_target(placed)
            dry, spans = defaultdict(lambda: np.zeros(length, dtype=np.float32)), defaultdict(list)
            for begin, duration, speaker, voice, index in placed:
                a = round(voice.offsets[index] * RATE)
                clip = sf.read(voice.track, start=a, frames=round(duration * RATE), dtype="float32", always_2d=True)[0][:, 0]
                b = round(begin * RATE)
                clip = clip[: length - b]
                dry[speaker][b:b + len(clip)] = clip
                spans[speaker].append((b, b + len(clip)))
            names = sorted(dry)
            stats[f"fill_{corpus}"].append({"placed": len(placed), "template_turns": len(template),
                                            "template_seconds": sum(t[2] - t[1] for t in template),
                                            "placed_seconds": sum(d for _, d, *_ in placed)})
            for kind, chosen in (("near", near_config), ("far", config)):
                rng = random.Random(f"{args.seed}:{kind}:{corpus}:{rid}")
                info = {}
                audio[f"syn_fill_{kind}_{corpus}"] = (render_meeting(
                    rng, [dry[n] for n in names], [spans[n] for n in names], length, chosen, rooms=rooms,
                    load_rir=load_rir, noise_loader=noise, sampling_rate=RATE, info=info), target)
                stats[f"syn_fill_{kind}_{corpus}"].append(info)
        for variant, (signal, target) in audio.items():
            cut = f"{variant}/{rid}"
            sf.write(args.output / f"{cut}.wav", signal, RATE, subtype="PCM_16")
            row = json.loads(json.dumps(r))
            row["id"] = cut
            row["audio_slots"][0]["ref"].update({"dataset_id": args.dataset_id, "version": args.version,
                                                 "split": variant, "cut_id": cut, "start": 0.0,
                                                 "duration": len(signal) / RATE})
            row["target"] = target
            if target != r["target"]:
                # The reference lines are the turns now (speaker labels as in the target).
                row["metadata"]["turns"] = [
                    {"source_id": f"{cut}-{k}", "speaker": label, "source_speaker": label, "start": float(a),
                     "end": float(b), "text": text}
                    for k, (label, a, b, text) in enumerate(LINE.fullmatch(line).groups()
                                                            for line in target.splitlines())]
            out_records[variant].append(row)
            index_rows.append({"channels": 1, "cut_id": cut, "duration": len(signal) / RATE, "num_frames": len(signal),
                               "relative_path": f"{cut}.wav", "root_alias": "meeting_synth_fidelity", "sample_rate": RATE})
        print(rid, flush=True)

    cat = args.catalog_dir
    (cat / "records").mkdir(parents=True, exist_ok=True)
    with gzip.open(cat / "audio-index.jsonl.gz", "wt") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in index_rows)
    artifacts = [{"name": "audio-index", "kind": "audio-index", "root_alias": "meeting_synth_fidelity_catalog",
                  "relative_path": "audio-index.jsonl.gz", "sha256": sha256_file(cat / "audio-index.jsonl.gz")}]
    splits = {}
    for variant, rows in out_records.items():
        path = cat / "records" / f"{variant}.jsonl.gz"
        with gzip.open(path, "wt") as stream:
            stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        artifacts.append({"name": f"{variant}-records", "kind": "audio-records",
                          "root_alias": "meeting_synth_fidelity_catalog", "relative_path": f"records/{variant}.jsonl.gz",
                          "sha256": sha256_file(path), "metadata": {"record_count": len(rows)}})
        splits[variant] = {"artifacts": {"records": [f"{variant}-records"], "audio_index": ["audio-index"]},
                           "statistics": {"records": len(rows)}}
    entry = {"schema_version": "dataset-catalog/2.0", "dataset_id": args.dataset_id, "version": args.version,
             "languages": ["zh"], "tasks": ["speaker_attributed_asr"], "aliases": [], "artifacts": artifacts,
             "splits": splits, "provenance": {"builder": "runs/zh-diar-meeting-synth-20261031/scripts/build_dev_fidelity.py",
                                              "usage": "evaluation_only", "online_meeting": data["online_meeting"]}}
    (cat / "catalog.yaml").write_text(yaml.safe_dump([entry], sort_keys=False, allow_unicode=True))
    (cat / "roots.json").write_text(json.dumps({"meeting_synth_fidelity": str(args.output),
                                                "meeting_synth_fidelity_catalog": str(cat)}, indent=1) + "\n")
    summary = {}
    for key, rows in stats.items():
        if key.startswith("fill_"):
            summary[key] = {"placed_over_template_seconds": round(sum(r["placed_seconds"] for r in rows)
                                                                  / sum(r["template_seconds"] for r in rows), 3),
                            "placed_utterances": sum(r["placed"] for r in rows),
                            "template_turns": sum(r["template_turns"] for r in rows)}
        else:
            summary[key] = {"snr_db_mean": round(float(np.mean([r["snr_db"] for r in rows])), 2)}
    (cat / "fidelity-stats.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
