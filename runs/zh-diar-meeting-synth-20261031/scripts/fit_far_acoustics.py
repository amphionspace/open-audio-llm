"""Measure real far-field meeting acoustics on training meetings: SNR, per-speaker level spread, speech level.

The synthesizer's acoustic ranges (OnlineMeetingConfig snr_db, level_db_std, speech_dbfs) are read from training
meetings, never from the dev set the fidelity check scores. Audio is the mean of the array channels, as the real
meeting training and evaluation windows.

Noise is the recording's floor: the FLOOR_PERCENTILE-th percentile of its FRAME s frame powers (minimum
statistics), over frames above digital silence. Annotation gaps do not work as noise on these meetings: attempts 001 (mean of gap chunks up to 3 dB
above their median) and 002 (median gap chunk) put AliMeeting train noise 15-25 dB above the ASR-verified room-noise
floors of meeting_room_noise@v2-20261014, often at speech level -- the gaps hold unannotated talk. The gap median is
kept as ``gap_noise_dbfs`` for comparison. A speaker's level is measured on at least 5 s of exclusive speech that
stands 3 dB above the floor.
"""
import argparse
import gzip
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16000
CHUNK, GUARD = 0.5, 0.5
FRAME, FLOOR_PERCENTILE = 0.1, 5
# Frame power below -120 dBFS is digital silence (zero padding, dropouts), under any 16-bit quantization floor (about
# -101 dBFS); attempt 003 crashed when such frames made up 5% of a recording and put its floor at zero.
DIGITAL_SILENCE = 1e-12
INDEXES = {
    "alimeeting": "alimeeting_sdm_meeting_sot/versions/real-meetings-v2-20261010/alimeeting_sdm-train-audio-index.jsonl.gz",
    "aishell4": "aishell4_meeting_sot/versions/real-meetings-v2-20261010/aishell4-train-audio-index.jsonl.gz",
}


def rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mask(turns, n, pad=0.0):
    m = np.zeros(n, dtype=bool)
    for start, end in turns:
        m[max(0, round((start - pad) * RATE)) : min(n, round((end + pad) * RATE))] = True
    return m


def measure(job):
    audio, rate = sf.read(job["path"], dtype="float32", always_2d=True)
    assert rate == RATE
    signal = audio.mean(axis=1)
    del audio
    n = len(signal)
    turns = job["turns"]
    by_speaker = {}
    for speaker, start, end, *_ in turns:
        by_speaker.setdefault(speaker, []).append((start, end))
    every = [(a, b) for spans in by_speaker.values() for a, b in spans]
    active = mask(every, n)
    busy = mask(every, n, GUARD)
    hop = round(CHUNK * RATE)
    chunks = [signal[i : i + hop] for i in range(0, n - hop, hop) if not busy[i : i + hop].any()]
    gap = float(np.median([np.mean(c**2) for c in chunks])) if chunks else float("nan")
    frame = round(FRAME * RATE)
    frames = (signal[: n // frame * frame].reshape(-1, frame) ** 2).mean(axis=1)
    live = frames > DIGITAL_SILENCE
    noise = float(np.percentile(frames[live], FLOOR_PERCENTILE))
    speech = float(np.mean(signal[active] ** 2)) - noise
    counts = np.zeros(n, dtype=np.int8)
    masks = {s: mask(spans, n) for s, spans in by_speaker.items()}
    for m in masks.values():
        counts += m
    levels = {}
    for s, m in masks.items():
        exclusive = m & (counts == 1)
        own = float(np.mean(signal[exclusive] ** 2)) if exclusive.sum() > 5 * RATE else 0.0
        if own > 2 * noise:
            levels[s] = 10 * np.log10(own - noise)
    mean_level = float(np.mean(list(levels.values()))) if levels else float("nan")
    return {"meeting": job["meeting"], "corpus": job["corpus"], "speakers": len(by_speaker),
            "snr_db": 10 * np.log10(max(speech, 1e-12) / noise), "noise_dbfs": 10 * np.log10(noise),
            "speech_dbfs": 10 * np.log10(max(speech, 1e-12)),
            "level_offsets_db": {s: round(v - mean_level, 2) for s, v in levels.items()},
            "gap_noise_dbfs": 10 * np.log10(gap) if chunks else None, "gap_chunks": len(chunks),
            "silent_frame_fraction": round(1 - float(live.mean()), 4)}


def summary(values):
    values = np.asarray(values, dtype=float)
    return {"n": int(len(values)), "mean": round(float(values.mean()), 2), "std": round(float(values.std()), 2),
            **{f"p{q}": round(float(np.percentile(values, q)), 2) for q in (5, 10, 25, 50, 75, 90, 95)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--templates", type=Path, required=True)
    p.add_argument("--datasets", type=Path, default=Path("/workspace/data/datasets"))
    p.add_argument("--roots", type=Path, required=True, help="roots.json resolving the audio index aliases")
    p.add_argument("--per-corpus", type=int, default=60)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    roots = json.loads(args.roots.read_text())
    templates = {(t["corpus"], t["meeting"]): t for t in rows(args.templates)}
    jobs = []
    for corpus, relative in INDEXES.items():
        index = {r["cut_id"]: r for r in rows(args.datasets / relative)}
        meetings = sorted(m for c, m in templates if c == corpus and m in index)
        for meeting in random.Random(args.seed).sample(meetings, min(args.per_corpus, len(meetings))):
            r = index[meeting]
            jobs.append({"corpus": corpus, "meeting": meeting, "turns": templates[(corpus, meeting)]["turns"],
                         "path": str(Path(roots[r["root_alias"]]) / r["relative_path"])})
    with ProcessPoolExecutor(args.workers) as pool:
        measured = list(pool.map(measure, jobs))
    ok = [m for m in measured if "failed" not in m]
    report = {"meetings": measured, "summary": {}}
    for corpus in (*INDEXES, "all"):
        chosen = [m for m in ok if corpus in ("all", m["corpus"])]
        report["summary"][corpus] = {
            "snr_db": summary([m["snr_db"] for m in chosen]),
            "speech_dbfs": summary([m["speech_dbfs"] for m in chosen]),
            "level_offsets_db": summary([v for m in chosen for v in m["level_offsets_db"].values()]),
            "failed": sum(1 for m in measured if "failed" in m and corpus in ("all", m["corpus"])),
        }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
