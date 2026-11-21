"""Target-speaker scenes made only of the synthetic spelling data's TTS voices, with twins (question 7).

Every train_ts record of context_correction_synth_zh@train-v1-20261114 (a Qwen3-TTS clone of one aishell3 / kespeech
speaker, enrolled with that speaker's real utterance) becomes one scene in which other cloned voices also talk, and a
share of the records are its twin: the same scene with another voice enrolled, transcript empty. This is the
alt-twins recipe of ts_scene_pool (audio-data-contract scripts/ts_robust/build_ts_scene_pool.py --alt-twins), built
for the cloned voices: in training a TTS voice was always the target (question 6), so the model transcribed any TTS
voice; here a TTS voice is the target in some scenes and somebody else in the others.

The records only name spans of the existing clips (the loader mixes them online, as for ts_scene_pool); the audio
index lists those clips at their place in train-v1-20261114, nothing is copied. Present scenes keep the source
record's think row.
usage: build_synth_ts_twins.py OUTPUT_DIR
"""
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATASET = "context_correction_synth_zh"
SOURCE_VERSION = "train-v1-20261114"
VERSION = "ts-twins-v1-20261121"
SPLIT = "train_ts_twins"
SOURCE = Path(f"/workspace/data/datasets/{DATASET}/versions/{SOURCE_VERSION}")
SEED = 20261121
# The alt-twins pools' share of twin records (ts_scene_pool alt-twins-20261024, --twin-share 0.3).
TWIN_SHARE = 0.3
# The clips are 7-26 s paragraphs, longer than any "short" target (0.8-2.5 s), so only the two other modes.
MODES = (("alternate", 0.5), ("burst", 0.5))
# Other voices per scene: the scene bench's owner scenes have one other speaker; two in some scenes.
OTHERS = ((1, 2), (0.7, 0.3))
# build_ts_scene_pool.py's longest alternate scene, gaps included; its gap bound is OnlineMixConfig's default.
ALTERNATE_SECONDS, ALTERNATE_GAP = 30.0, 1.0
# One span of another voice's clip per turn in alternate scenes (a turn of the scene bench's separate-talk scenes).
TURN_SECONDS = (2.0, 8.0)
# In burst scenes each other voice talks over this share of the target's clip.
OVERLAP_SHARE = (0.4, 1.0)


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def span(rng, clip, seconds):
    seconds = min(seconds, clip["duration"])
    start = rng.uniform(0.0, clip["duration"] - seconds)
    return {"cut_id": clip["cut_id"], "start": round(start, 3), "duration": round(seconds, 3)}


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(SEED)
    records = read(SOURCE / "records/train_ts.jsonl.gz")
    index = {row["cut_id"]: row for row in read(SOURCE / "audio-index.jsonl.gz")}
    think = {row["record_id"]: row for row in read(SOURCE / "think.jsonl.gz")}
    voice = lambda r: r["metadata"]["voice_id"]
    clips, enrollments = {}, {}
    for r in records:
        enrollment, mixture = (r["audio_slots"][0]["ref"], r["audio_slots"][1]["ref"])
        clips.setdefault(voice(r), []).append({"cut_id": mixture["cut_id"], "duration": mixture["duration"]})
        enrollments[voice(r)] = {"cut_id": enrollment["cut_id"], "start": 0.0, "duration": enrollment["duration"]}
    voices = sorted(clips)
    ref = lambda cut: {"dataset_id": DATASET, "version": VERSION, "split": SPLIT, **cut}
    out_records, out_think, used, stats = [], [], set(), Counter()
    for pair, r in enumerate(records):
        mode = rng.choices([m for m, _ in MODES], [w for _, w in MODES])[0]
        target = {"cut_id": r["audio_slots"][1]["ref"]["cut_id"], "start": 0.0,
                  "duration": r["audio_slots"][1]["ref"]["duration"]}
        count = rng.choices(*OTHERS)[0]
        room = lambda n: ALTERNATE_SECONDS - target["duration"] - ALTERNATE_GAP * n
        if mode == "alternate" and room(count) / count < TURN_SECONDS[0]:
            count = 1
        assert mode != "alternate" or room(count) / count >= TURN_SECONDS[0], r["id"]
        names = rng.sample([v for v in voices if v != voice(r)], count)
        others = []
        for name in names:
            clip = rng.choice(clips[name])
            if mode == "alternate":
                seconds = rng.uniform(TURN_SECONDS[0], min(TURN_SECONDS[1], room(count) / count))
            else:
                seconds = target["duration"] * rng.uniform(*OVERLAP_SHARE)
            others.append({"speaker": name, "utts": [span(rng, clip, seconds)]})
        scene = {"mode": mode, "target_speaker": voice(r), "targets": [{**target, "text": r["target"]}],
                 "enrollment": enrollments[voice(r)], "interferers": others}
        record_id = f"{VERSION}:{SPLIT}:{mode}:{target['cut_id']}"
        out_records.append({
            "schema_version": "audio-record/1.0", "id": record_id, "task": "ts_asr", "target": r["target"],
            "language": "zh", "hotwords": [],
            "labels": {**r["labels"], "target_present": True, "schedule": mode, "n_interferers": count, "pair": pair},
            "audio_slots": [{"name": "enrollment", "purpose": "enrollment", "ref": ref(enrollments[voice(r)])},
                            {"name": "mixture", "purpose": "mixture", "ref": ref(target)}],
            "metadata": {"scene": scene, "source_group": voice(r), "source_record": r["id"]}})
        row = think.get(r["id"])
        if row is not None:
            out_think.append({**row, "dataset": f"{DATASET}@{VERSION}", "record_id": record_id})
        used.update([target["cut_id"], enrollments[voice(r)]["cut_id"]] + [o["utts"][0]["cut_id"] for o in others])
        stats[f"present {mode}"] += 1
        # TWIN_SHARE of all records are twins: one per base scene with TWIN_SHARE / (1 - TWIN_SHARE).
        if rng.random() < TWIN_SHARE / (1 - TWIN_SHARE):
            enrolled = rng.choice([v for v in voices if v != voice(r) and v not in names])
            twin = {"mode": mode, "target_speaker": enrolled, "targets": [], "enrollment": enrollments[enrolled],
                    "interferers": [{"speaker": voice(r), "utts": [target]}] + others, "twin": {"kind": "random"}}
            out_records.append({
                "schema_version": "audio-record/1.0", "id": f"{VERSION}:{SPLIT}:twin-{mode}:{target['cut_id']}",
                "task": "ts_asr", "target": "", "language": "zh", "hotwords": [],
                "labels": {"target_present": False, "schedule": mode, "n_interferers": count + 1, "pair": pair,
                           "twin_enrollment": "random"},
                "audio_slots": [{"name": "enrollment", "purpose": "enrollment", "ref": ref(enrollments[enrolled])},
                                {"name": "mixture", "purpose": "mixture", "ref": ref(target)}],
                "metadata": {"scene": twin, "source_group": enrolled, "source_record": r["id"]}})
            used.add(enrollments[enrolled]["cut_id"])
            stats[f"twin {mode}"] += 1
    (out / "records").mkdir()
    write(out / f"records/{SPLIT}.jsonl.gz", out_records)
    write(out / "audio-index.jsonl.gz", [index[c] for c in sorted(used)])
    write(out / "think.jsonl.gz", out_think)
    seconds = lambda rec: (sum(u["duration"] for u in rec["metadata"]["scene"]["targets"])
                           + sum(u["duration"] for o in rec["metadata"]["scene"]["interferers"] for u in o["utts"])
                           if rec["metadata"]["scene"]["mode"] == "alternate" else
                           max(u["duration"] for u in rec["metadata"]["scene"]["targets"]
                               + [u for o in rec["metadata"]["scene"]["interferers"] for u in o["utts"]]))
    summary = {"records": len(out_records), "by_kind": dict(sorted(stats.items())),
               "twin_share": round(sum(v for k, v in stats.items() if k.startswith("twin")) / len(out_records), 4),
               "others": dict(Counter(len(r["metadata"]["scene"]["interferers"]) for r in out_records
                                      if r["labels"]["target_present"])),
               "think_rows": len(out_think), "think_on_rows": sum(r["think_on"] for r in out_think),
               "audio_clips": len(used), "voices": len(voices),
               "scene_hours": round(sum(seconds(r) for r in out_records) / 3600, 4),
               "longest_scene_seconds": round(max(seconds(r) for r in out_records), 3)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    state = {"schema_version": "dataset-state/1.0", "dataset_id": DATASET, "version": VERSION, "state": "prepared",
             "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "error": None,
             "metadata": {"owner": "mingdong", "derived_from": f"{DATASET}@{SOURCE_VERSION}",
                          "code": f"open-audio-llm runs/zh-ctx-clue-target-20261117/scripts/build_synth_ts_twins.py "
                                  f"(sha256 {code}), seed {SEED}",
                          "records": {SPLIT: len(out_records)}, "think_rows": len(out_think)}}
    (out / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
