"""Automatic check of the online meeting synthesis training data before the control training.

1. Leakage: every template and material meeting is a training meeting, and no template or material meeting, nor any
   material speaker, appears in a real-meeting dev/test set (AliMeeting Eval/Test, AISHELL-4 test, AISHELL-5 Dev) or
   the zh245 holdout.
2. Windows through the training path: CatalogSwiftDataset with the training augmentation (arm I's block, this source
   in noise_exclude_datasets) and the synth config's online_meeting block, over several epochs. Per window: audio
   finite, not digitally silent anywhere, peak <= 1, no extra noise applied; target lines parse, 0 <= start < end <=
   duration, labels S1..Sn in first-appearance order, text non-empty; each line's span carries energy above the
   window's floor (speech where it is labelled); characters per second.
3. Unlabelled partner speech: share of placed material seconds whose source utterance overlapped another speaker of
   its own conversation (that speaker leaks into the headset clip and is not in the synthesized target), and how loud
   the leak is: on the source headset tracks of sampled meetings, power where only others talk over power where only
   the wearer talks, each above the power where nobody talks.
"""
import argparse
import glob
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

RATE = 16000
FRAME = round(0.1 * RATE)
DIGITAL_SILENCE = 1e-12  # frame power below -120 dBFS, as in fit_far_acoustics.py
FLOOR_PERCENTILE = 5
# A labelled line whose span is under this many dB above the window floor carries no audible speech.
LINE_MIN_DB = 3.0
LINE = re.compile(r"\[(S\d+)\]\[(\d+\.\d\d)-(\d+\.\d\d)\] (.*)")
CHARS = re.compile(r"[\W_]+")
ALIMEETING = "/workspace/data/datasets/alimeeting/source/openslr-119-local-20260826"
AISHELL4_TEST = "/workspace/data/datasets/aishell4/source/data_aishell4-f68f1bf5/test/wav"
AISHELL5_DEV = "/workspace/data/datasets/aishell5_meeting_sot/versions/far-mic-v1-20261010/aishell5-dev_120s.jsonl.gz"
TRAIN_INDEXES = {
    "alimeeting": "/workspace/data/datasets/alimeeting_sdm_meeting_sot/versions/real-meetings-v2-20261010/alimeeting_sdm-train-audio-index.jsonl.gz",
    "aishell4": "/workspace/data/datasets/aishell4_meeting_sot/versions/real-meetings-v2-20261010/aishell4-train-audio-index.jsonl.gz",
    "aishell5": "/workspace/data/datasets/aishell5_meeting_sot/versions/far-mic-v1-20261010/aishell5-train-audio-index.jsonl.gz",
}
ALIMEETING_NEAR = f"{ALIMEETING}/Train_Ali_near/audio_dir"
AISHELL5_TRAIN = "/workspace/data/datasets/aishell5/source/openslr-159-20261010/extracted/train"
HOLDOUT = "/workspace/data/datasets/alimeeting_sdm_meeting_sot/versions/real-meetings-v2-20261010/zh245-holdout.json"


def rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def aishell4_key(name):
    """AISHELL-4 train meetings carry a date prefix (20200616_M_R001S01C01), test ones not (L_R003S01C02)."""
    match = re.search(r"([SML])_(R\d+S\d+C\d+)$", name)
    return f"{match.group(1)}_{match.group(2)}" if match else name


def leakage(pool):
    templates = rows(pool / "templates.jsonl.gz")
    material = rows(pool / "material.jsonl.gz")
    meetings = defaultdict(set)
    for t in templates:
        meetings[t["corpus"]].add(t["meeting"])
    material_meetings, speakers = defaultdict(set), defaultdict(set)
    for m in material:
        material_meetings[m["corpus"]].add(m["meeting"])
        speakers[m["corpus"]].add(m["speaker"])
    # AISHELL-5 indexes one recording per microphone ("aishell5-train-001_DX01C01"); its meeting is the prefix.
    train = {c: {r["cut_id"].split("_DX")[0] if c == "aishell5" else r["cut_id"] for r in rows(p)}
             for c, p in TRAIN_INDEXES.items()}
    held = set(json.loads(Path(HOLDOUT).read_text())["meetings"])
    eval_meetings = defaultdict(set)
    eval_speakers = defaultdict(set)
    for part in ("Eval_Ali/Eval_Ali_near", "Test_Ali/Test_Ali_near"):
        for path in glob.glob(f"{ALIMEETING}/{part}/audio_dir/*.wav"):
            name = Path(path).stem  # R8002_M8002_N_SPK8005
            eval_meetings["alimeeting"].add(name.split("_N_")[0])
            eval_speakers["alimeeting"].add(name.rsplit("SPK", 1)[1])
    eval_meetings["aishell4"] = {aishell4_key(Path(p).stem) for p in glob.glob(f"{AISHELL4_TEST}/*.flac")}
    for r in rows(AISHELL5_DEV):
        eval_meetings["aishell5"].add(r["metadata"]["meeting_id"])
        eval_speakers["aishell5"].update(t["source_speaker"] for t in r["metadata"]["turns"])
    report = {}
    for corpus in sorted(set(meetings) | set(material_meetings)):
        used = meetings[corpus] | material_meetings[corpus]
        keyed = {aishell4_key(m) for m in used} if corpus == "aishell4" else used
        report[corpus] = {
            "template_meetings": len(meetings[corpus]), "material_meetings": len(material_meetings[corpus]),
            "material_speakers": len(speakers[corpus]),
            "eval_meetings_known": len(eval_meetings[corpus]), "eval_speakers_known": len(eval_speakers[corpus]),
            "not_in_train_index": sorted(used - train.get(corpus, set()))[:20],
            "in_eval": sorted(keyed & eval_meetings[corpus])[:20],
            "in_zh245_holdout": sorted(used & held)[:20],
            "material_speakers_in_eval": sorted(speakers[corpus] & eval_speakers[corpus])[:20],
        }
    report["zh245_holdout_meetings"] = len(held)
    report["pass"] = all(not v[k] for v in report.values() if isinstance(v, dict)
                         for k in ("not_in_train_index", "in_eval", "in_zh245_holdout", "material_speakers_in_eval"))
    return report


def headset_tracks(corpus, meeting, speakers):
    """As build_meeting_pool.py: AliMeeting near-field files, AISHELL-5 DA tracks named by their TextGrid tier."""
    if corpus == "alimeeting":
        found = {s: glob.glob(f"{ALIMEETING_NEAR}/{meeting}_*_SPK{s}.wav") for s in speakers}
        return {s: paths[0] for s, paths in found.items() if len(paths) == 1}
    found = {}
    for grid in sorted(Path(AISHELL5_TRAIN, meeting.rsplit("-", 1)[1]).glob("DA*.TextGrid")):
        raw = grid.read_bytes()
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8")
        names = re.findall(r'name = "([^"]+)"', text)
        if len(names) == 1 and grid.with_suffix(".wav").is_file():
            found[names[0]] = str(grid.with_suffix(".wav"))
    return found


def leak_levels(job):
    corpus, meeting, turns = job
    out = []
    for speaker, path in headset_tracks(corpus, meeting, {t[0] for t in turns}).items():
        x = sf.read(path, dtype="float32", always_2d=True)[0][:, 0].astype(np.float64)
        n = len(x)
        own, others = np.zeros(n, bool), np.zeros(n, np.int16)
        for s, a, b, *_ in turns:
            a, b = max(0, round(a * RATE)), min(n, round(b * RATE))
            if s == speaker:
                own[a:b] = True
            else:
                others[a:b] += 1
        regions = {"own": own & (others == 0), "others": ~own & (others > 0), "none": ~own & (others == 0)}
        if min(m.sum() for m in regions.values()) < 5 * RATE:
            continue
        power = {k: float(np.mean(x[m] ** 2)) for k, m in regions.items()}
        if power["own"] > power["none"] and power["others"] > power["none"]:
            out.append(10 * np.log10((power["others"] - power["none"]) / (power["own"] - power["none"])))
    return out


def window_checks(audio, answer, duration):
    problems = []
    if not np.isfinite(audio).all():
        problems.append("non-finite audio")
        return problems, {}
    peak = float(np.abs(audio).max())
    if peak > 1.0:
        problems.append(f"peak {peak:.2f}")
    frames = (audio[: len(audio) // FRAME * FRAME].reshape(-1, FRAME).astype(np.float64) ** 2).mean(axis=1)
    silent = frames <= DIGITAL_SILENCE
    if silent.any():
        problems.append(f"{int(silent.sum())} digitally silent 100 ms frames")
    floor = float(np.percentile(frames[~silent], FLOOR_PERCENTILE)) if (~silent).any() else 0.0
    if len(audio) / RATE < duration - 0.05 or len(audio) / RATE > duration + 0.05:
        problems.append(f"audio {len(audio) / RATE:.2f} s for a {duration:.2f} s window")
    if "<asr_text>" not in answer:
        problems.append("no <asr_text>")
        return problems, {}
    text = answer.split("<asr_text>", 1)[1]
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        problems.append("empty target")
    labels, line_db, cps, previous = [], [], [], -1.0
    for line in lines:
        match = LINE.fullmatch(line)
        if not match:
            problems.append(f"unparsed line {line[:40]!r}")
            continue
        label, a, b, words = match.group(1), float(match.group(2)), float(match.group(3)), match.group(4)
        if label not in labels:
            labels.append(label)
        if not (0 <= a < b <= duration + 0.01):
            problems.append(f"bad span {a}-{b} in a {duration} s window")
        if a < previous:
            problems.append("lines not in start order")
        previous = a
        chars = len(CHARS.sub("", words))
        if not chars:
            problems.append("empty line text")
        cps.append(chars / max(b - a, 1e-3))
        span = audio[round(a * RATE): round(b * RATE)].astype(np.float64)
        if len(span) and floor > 0:
            line_db.append(10 * np.log10(np.mean(span**2) / floor))
    if labels != [f"S{k}" for k in range(1, len(labels) + 1)]:
        problems.append(f"labels {labels}")
    return problems, {"line_db": line_db, "cps": cps, "speakers": len(labels), "lines": len(lines),
                      "floor_dbfs": 10 * np.log10(floor) if floor > 0 else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--synth-config", type=Path, required=True, help="YAML whose online_meeting block is trained")
    p.add_argument("--augmentation-from", type=Path, required=True, help="training data config whose augmentation, "
                   "noise_sources and rir_sources the control run keeps")
    p.add_argument("--splits", nargs="+", default=["train_30s", "train_120s", "train_300s"])
    p.add_argument("--windows", type=int, default=150, help="per split and epoch")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--leak-meetings", type=int, default=20, help="per material corpus")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    report = {"leakage": leakage(args.pool)}
    print(json.dumps(report["leakage"], ensure_ascii=False), flush=True)
    from concurrent.futures import ProcessPoolExecutor

    templates = [t for t in rows(args.pool / "templates.jsonl.gz") if t["corpus"] in ("alimeeting", "aishell5")]
    pick = np.random.default_rng(20261101)
    jobs = []
    for corpus in ("alimeeting", "aishell5"):
        chosen = [t for t in templates if t["corpus"] == corpus]
        for k in pick.choice(len(chosen), size=min(args.leak_meetings, len(chosen)), replace=False):
            jobs.append((corpus, chosen[k]["meeting"], chosen[k]["turns"]))
    with ProcessPoolExecutor(8) as pool:
        levels = list(pool.map(leak_levels, jobs))
    leak = defaultdict(list)
    for (corpus, *_), values in zip(jobs, levels):
        leak[corpus].extend(values)
    report["headset_leak_db"] = {c: {"tracks": len(v), **{f"p{q}": round(float(np.percentile(v, q)), 1)
                                                         for q in (10, 50, 90)}} for c, v in leak.items() if v}
    print(json.dumps(report["headset_leak_db"]), flush=True)

    import open_audio_llm.data.catalog_dataset as cd

    synth = yaml.safe_load(args.synth_config.read_text())
    base = yaml.safe_load(args.augmentation_from.read_text())
    dataset_id = synth["online_meeting"]["material"]["dataset_id"]
    version = synth["online_meeting"]["material"]["version"]
    augmentation = dict(base["augmentation"])
    augmentation["noise_exclude_datasets"] = sorted({*augmentation.get("noise_exclude_datasets", []), dataset_id})
    config = {"catalog": synth["catalog"], "roots": synth["roots"], "seed": base["seed"],
              "sampling_rate": RATE, "augmentation": augmentation,
              **{k: base[k] for k in ("noise_sources", "rir_sources") if k in base},
              "train": [{"dataset_id": dataset_id, "version": version, "split": s, "max_duration": 650,
                         "online_meeting": synth["online_meeting"]} for s in args.splits]}
    noise_seen = Counter()
    original = cd.augment_waveform

    def watched(audio, sampling_rate, rng, augmentation, *rest, **kw):
        noise_seen["noise_prob>0" if augmentation.noise_prob else "noise_prob=0"] += 1
        return original(audio, sampling_rate, rng, augmentation, *rest, **kw)

    cd.augment_waveform = watched
    dataset = cd.CatalogSwiftDataset(config, message_format="qwen3_asr")
    plans = []
    plan = dataset._plan_meeting

    def keep(record, rng):
        planned = plan(record, rng)
        plans.append(planned.metadata["meeting_plan"])
        return planned

    dataset._plan_meeting = keep
    material = {(m["relative_path"], round(m["offset"], 3)): m for m in rows(args.pool / "material.jsonl.gz")}
    # The splits are 30 / 120 / 300 s windows: stratify the picks by window length.
    durations = np.asarray(getattr(dataset.records, "durations", None)
                           or [r.record.audio_slots[0].ref.duration for r in dataset.records], dtype=float)
    buckets = np.digitize(durations, [35.0, 125.0])
    rng = np.random.default_rng(20261101)
    picks = np.concatenate([rng.choice(np.flatnonzero(buckets == b), size=min(args.windows, int((buckets == b).sum())),
                                       replace=False) for b in np.unique(buckets)])
    print("windows per length bucket", dict(Counter(buckets.tolist())), "picked", len(picks), flush=True)
    problems, stats, failures = Counter(), defaultdict(list), []
    overlap_seconds, placed_seconds = Counter(), Counter()
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        for idx in picks:
            plans.clear()
            sample = dataset[int(idx)]
            audio, rate = sf.read(BytesIO(sample["audios"][0]), dtype="float32")
            assert rate == RATE
            plan_ = plans[-1]
            duration = max(u["start"] + u["duration"] for u in plan_["utterances"]) if plan_["utterances"] else 0
            # The window length is the record's; the audio must match it.
            length = len(audio) / RATE
            found, info = window_checks(audio, sample["messages"][-1]["content"], length)
            for problem in found:
                problems[problem.split(" ")[0] if problem[0].isdigit() else problem.split(" in ")[0]] += 1
            if found:
                failures.append({"epoch": epoch, "index": int(idx), "problems": found[:5]})
            corpus = plan_["corpus"]
            for key in ("line_db", "cps"):
                stats[f"{corpus}/{key}"].extend(info.get(key, []))
            stats[f"{corpus}/speakers"].append(info.get("speakers", 0))
            if info.get("floor_dbfs") is not None:
                stats[f"{corpus}/floor_dbfs"].append(info["floor_dbfs"])
            for u in plan_["utterances"]:
                relative = u["track"].split(f"/{version}/", 1)[-1]
                row = material.get((relative, round(u["offset"], 3)))
                placed_seconds[corpus] += u["duration"]
                if row is not None:
                    overlap_seconds[corpus] += u["duration"] * row["overlap"]
                else:
                    problems["placed utterance not found in material"] += 1
            if duration > length + 0.01:
                problems["placement past the window end"] += 1
        print(f"epoch {epoch} done: {sum(problems.values())} problems so far", flush=True)

    def summary(values):
        v = np.asarray(values, dtype=float)
        return {"n": int(len(v)), **{f"p{q}": round(float(np.percentile(v, q)), 2) for q in (1, 5, 50, 95, 99)}} if len(v) else {}

    report["windows"] = {"checked": int(len(picks) * args.epochs), "problems": dict(problems),
                         "failures": failures[:50], "augment_calls": dict(noise_seen),
                         "noise_exclude_datasets_has_source": dataset_id in augmentation["noise_exclude_datasets"]}
    report["stats"] = {k: summary(v) for k, v in sorted(stats.items())}
    for corpus in placed_seconds:
        lines = np.asarray(stats[f"{corpus}/line_db"], dtype=float)
        report["stats"][f"{corpus}/lines_below_{LINE_MIN_DB:g}dB_share"] = round(float((lines < LINE_MIN_DB).mean()), 4) if len(lines) else None
    report["unlabelled_partner_speech"] = {
        c: {"placed_seconds": round(placed_seconds[c], 1),
            "overlapped_share": round(overlap_seconds[c] / placed_seconds[c], 4)} for c in placed_seconds}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("windows", "unlabelled_partner_speech")}, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report["stats"].items() if "share" in k or k.endswith("line_db")}, indent=1))
    sys.exit(0 if report["leakage"]["pass"] and not problems else 1)


if __name__ == "__main__":
    main()
