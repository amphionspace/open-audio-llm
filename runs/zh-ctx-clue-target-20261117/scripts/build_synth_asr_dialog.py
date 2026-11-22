"""Plain-ASR dialogues of the synthetic spelling data's TTS voices: every speaker's words, in turn order.

Question 9 (README "判定 · 第 9 问"): J's recipe added TS scenes of these voices (ts-twins-v1-20261121), whose answer
is only the enrolled voice's words, and J then left out whole turns of long dialogues in plain ASR (鼎桥 scene-asr,
GRPO experiment README "漏字归因"). The mainline's plain-ASR data has all-speaker dialogue windows only of real
meetings (allspk-asr-v1-20261103); these are their TTS counterpart: the same voices and clips as the twin scenes,
no enrollment, and the transcript is everybody's.

Each dialogue takes 2 or 3 cloned voices of train-v1-20261114 train_ts and lets them take turns, one whole clip per
turn (a clip's transcript is only known whole), never the same voice twice in a row, until a drawn length is
reached. Voices are set against the first one by the twin scenes' SIR range, turns are apart by OnlineMixConfig's
alternate gap; reverberation, noise and level come from the training recipe's augmentation, as for any plain-ASR
source. The audio is rendered here (FLAC, 16 kHz mono), since the loader composes scenes only for TS records.
usage: build_synth_asr_dialog.py OUTPUT_DIR   (OUTPUT_DIR becomes the version directory)
"""
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

DATASET = "context_correction_synth_zh"
SOURCE_VERSION = "train-v1-20261114"
VERSION = "asr-dialog-v1-20261122"
SPLIT = "train_asr_dialog"
DATA_ROOT = Path("/workspace/data")
SOURCE = DATA_ROOT / f"datasets/{DATASET}/versions/{SOURCE_VERSION}"
SEED = 20261122
RATE = 16000
# About one dialogue per alternate twin scene (2,002 of the 4,184 twin records are alternate).
DIALOGUES = 2000
# Voices per dialogue: the twin scenes' 1 or 2 other voices besides the target.
VOICES = ((2, 3), (0.7, 0.3))
# Window lengths of the real-meeting all-speaker data (allspk-asr-v1-20261103 length_s); the clips are 7-26 s.
LENGTH_SECONDS = (10.0, 90.0)
# OnlineMixConfig.alternate_gap_seconds, the gap between turns of the twin scenes.
GAP_SECONDS = (0.2, 1.0)
# The twin scenes' online_mix sir_db: each later voice is set this many dB under the first one.
SIR_DB = (-6.0, 10.0)
PEAK = 0.99
# A clip transcript that ends without sentence punctuation (2.2% of the synthetic texts) gets a full stop, so turns
# do not run into one another.
END_PUNCTUATION = "。！？；…!?;.,，、"


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def rms(audio):
    voiced = audio[np.abs(audio) > 0]
    return float(np.sqrt(np.mean(voiced ** 2))) if voiced.size else 0.0


def plan(rng, clips, voices):
    """Turns of one dialogue: [(voice, clip)], each clip and each spelling pair at most once."""
    n = rng.choices(*VOICES)[0]
    cast = rng.sample(voices, n)
    goal = rng.uniform(*LENGTH_SECONDS)
    turns, used, pairs, seconds, last = [], set(), set(), 0.0, None
    while True:
        spoken = {v for v, _ in turns}
        if seconds >= goal and spoken == set(cast) and len(turns) >= 2:
            return turns
        options = [v for v in cast if v != last]
        rng.shuffle(options)
        # A voice that has not spoken yet goes first, so every voice is heard.
        options.sort(key=lambda v: v in spoken)
        for voice in options:
            fit = [c for c in clips[voice] if c["cut_id"] not in used and c["pair"] not in pairs
                   and seconds + GAP_SECONDS[1] + c["duration"] <= LENGTH_SECONDS[1]]
            if fit:
                clip = rng.choice(fit)
                turns.append((voice, clip)); used.add(clip["cut_id"]); pairs.add(clip["pair"])
                seconds += clip["duration"] + (GAP_SECONDS[1] if len(turns) > 1 else 0.0)
                last = voice
                break
        else:
            # Nothing fits any more: keep the dialogue if every voice spoke, else draw a new one.
            return turns if spoken == set(cast) and len(turns) >= 2 else None


def render(rng, turns, audio_of):
    gains, pieces, cursor, placed = {}, [], 0, []
    reference = None
    for k, (voice, clip) in enumerate(turns):
        audio = audio_of(clip)
        if voice not in gains:
            if reference is None:
                reference, gains[voice] = rms(audio), 1.0
            else:
                gains[voice] = reference / max(rms(audio), 1e-8) * 10 ** (-rng.uniform(*SIR_DB) / 20)
        if k:
            cursor += int(rng.uniform(*GAP_SECONDS) * RATE)
        placed.append((cursor, audio * gains[voice]))
        cursor += len(audio)
    mixture = np.zeros(cursor, dtype=np.float32)
    for start, audio in placed:
        mixture[start:start + len(audio)] += audio
    peak = float(np.max(np.abs(mixture)))
    if peak > PEAK:
        mixture *= PEAK / peak
    return mixture, [(start / RATE, len(audio) / RATE) for start, audio in placed]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    (out / "audio").mkdir()
    (out / "records").mkdir()
    rng = random.Random(SEED)
    index = {row["cut_id"]: row for row in read(SOURCE / "audio-index.jsonl.gz")}
    clips = {}
    for r in read(SOURCE / "records/train_ts.jsonl.gz"):
        ref = r["audio_slots"][1]["ref"]
        text = r["target"].strip()
        if text[-1] not in END_PUNCTUATION:
            text += "。"
        clips.setdefault(r["metadata"]["voice_id"], []).append(
            {"cut_id": ref["cut_id"], "duration": ref["duration"], "text": text,
             "pair": r["labels"]["context_correction"]["pair_id"]})
    voices = sorted(v for v in clips if len(clips[v]) >= 2)

    def audio_of(clip):
        audio, rate = sf.read(DATA_ROOT / index[clip["cut_id"]]["relative_path"], dtype="float32")
        assert rate == RATE and audio.ndim == 1, clip["cut_id"]
        return audio

    records, audio_index, stats = [], [], Counter()
    while len(records) < DIALOGUES:
        turns = plan(rng, clips, voices)
        if turns is None:
            stats["redrawn"] += 1
            continue
        mixture, times = render(rng, turns, audio_of)
        cut_id = f"dialog-{len(records):05d}"
        path = out / "audio" / f"{cut_id}.flac"
        sf.write(path, mixture, RATE, subtype="PCM_16")
        duration = round(len(mixture) / RATE, 4)
        audio_index.append({
            "channels": 1, "cut_id": cut_id, "duration": duration, "num_frames": len(mixture),
            # Where the version directory is published (registry root aidc_data), wherever it was built.
            "relative_path": f"datasets/{DATASET}/versions/{VERSION}/audio/{cut_id}.flac",
            "root_alias": "aidc_data", "sample_rate": RATE, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        records.append({
            "schema_version": "audio-record/1.0", "id": f"{VERSION}:{SPLIT}:{cut_id}", "task": "asr",
            "target": "".join(clip["text"] for _, clip in turns), "language": "zh", "hotwords": [],
            "labels": {"source": "synthetic"},
            "audio_slots": [{"name": "primary", "purpose": "primary", "ref": {
                "dataset_id": DATASET, "version": VERSION, "split": SPLIT, "cut_id": cut_id, "duration": duration}}],
            "metadata": {"voices": len({v for v, _ in turns}), "turns": [
                {"voice_id": v, "cut_id": c["cut_id"], "start": round(s, 3), "duration": round(d, 3),
                 "text": c["text"]} for (v, c), (s, d) in zip(turns, times)]}})
        stats[f"voices {len({v for v, _ in turns})}"] += 1
        stats["turns"] += len(turns)
        stats[f"{int(duration // 10) * 10}-{int(duration // 10) * 10 + 10}s"] += 1
    write(out / "records" / f"{SPLIT}.jsonl.gz", records)
    write(out / "audio-index.jsonl.gz", audio_index)
    hours = sum(r["duration"] for r in audio_index) / 3600
    summary = {"records": len(records), "hours": round(hours, 3), "clips_used": len({t["cut_id"] for r in records
               for t in r["metadata"]["turns"]}), "voices_used": len({t["voice_id"] for r in records
               for t in r["metadata"]["turns"]}), **dict(sorted(stats.items()))}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    (out / "state.json").write_text(json.dumps({
        "schema_version": "dataset-state/1.0", "dataset_id": DATASET, "version": VERSION, "state": "prepared",
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "error": None,
        "metadata": {"owner": "mingdong", "derived_from": f"{DATASET}@{SOURCE_VERSION}",
                     "code": f"open-audio-llm runs/zh-ctx-clue-target-20261117/scripts/build_synth_asr_dialog.py "
                             f"(sha256 {code}), seed {SEED}", "records": {SPLIT: len(records)}}},
        ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
