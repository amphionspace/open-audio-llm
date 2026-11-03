"""Which placed utterances does the model split off into a hypothesis speaker of its own? (v1 fidelity decodes)

Windows are re-planned exactly as built (checked against the decoded reference). A reference turn of > 3 units counts as
split off when its text is best matched by a hypothesis speaker no reference speaker is mapped to. Split rates are units
split / units per bucket of: duration, gap to the voice's previous utterance, jump in the source order, level relative to
the voice in this window, number of source conversations in the window, the voice's share of the window; and per voice.
In windows of fewer than 4 voices (the model hardly ever outputs a fifth speaker) also by the share of the utterance during
which, in the source meeting, someone without a voice in this window (foreign) or with one (own) was talking, overall and
paired: within each decode that splits, the split rate of utterances with and without foreign leak."""
import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from audio_data_contract import resolve_split

from open_audio_llm.data import meeting_online as mo
from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoints as sc  # noqa: E402

MATCH = 0.34
RATE = 16000


def bucket(value, edges):
    for edge in edges:
        if value < edge:
            return f"<{edge}"
    return f">={edges[-1]}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-records", type=Path, required=True)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--predictions", nargs="+", type=Path, required=True)
    p.add_argument("--corpus", default="alimeeting")
    p.add_argument("--kind", default="near")
    p.add_argument("--seed", default="fidelity-20261031")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = yaml.safe_load(args.data_config.read_text())
    config = replace(mo.OnlineMeetingConfig(**data["online_meeting"]), max_overlap=1.0, corpus_weights={args.corpus: 1.0})
    resolver = LhotseCatalogAudioResolver(data["catalog"], data["roots"])
    m = config.material
    tracks = {}
    for path in resolve_split(resolver.catalog, m["dataset_id"], m["version"], m["split"], "audio_index", resolver.roots):
        for row in mo.read_material_rows([path]):
            tracks[row["cut_id"]] = str(Path(resolver.roots[row["root_alias"]]) / row["relative_path"])
    rows = list(mo.read_material_rows(resolve_split(
        resolver.catalog, m["dataset_id"], m["version"], m["split"], "meeting_material", resolver.roots)))
    track_of = lambda row: tracks[f"{row['corpus']}/{row['meeting']}/{row['speaker']}"]  # noqa: E731
    by_key = {(track_of(r), round(r["offset"], 3)): r for r in rows}
    by_meeting = defaultdict(list)
    for r in rows:
        by_meeting[r["meeting"]].append(r)
    material = mo.load_material(rows, config, track_of)
    records = [json.loads(line) for line in gzip.open(args.panel_records, "rt")]
    predictions = defaultdict(list)
    for path in args.predictions:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("label") == f"syn_fill_{args.kind}_{args.corpus}":
                predictions[row["id"].split("/", 1)[1]].append(row)

    tables = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    paired = []
    per_voice = defaultdict(lambda: [0, 0, 0])
    window_rows = []
    checked = 0
    for r in records:
        length = round(r["audio_slots"][0]["ref"]["duration"] * RATE) / RATE
        turns = sorted(r["metadata"]["turns"], key=lambda t: t["start"])
        template = [[t["source_speaker"], t["start"], t["end"], len(mo.CHARS_PATTERN.sub("", t["text"]))] for t in turns]
        _, placed = mo.plan_meeting(random.Random(f"{args.seed}:fill:{args.corpus}:{r['id']}"), template, length, material, config)
        target, _ = mo.meeting_target(placed)
        decoded = predictions.get(r["id"], [])
        if not decoded:
            continue
        if sc.turns_of(decoded[0]["reference"]) != sc.turns_of(target):
            raise SystemExit(f"re-planned window differs from the decoded reference: {r['id']}")
        checked += 1
        kept = [item for item in placed if round(item[0], 2) < round(item[0] + item[1], 2)]
        ref = sc.turns_of(target)
        if len(kept) != len(ref):
            raise SystemExit(f"{r['id']}: {len(kept)} placed, {len(ref)} turns")
        # Features of each placed utterance.
        levels, previous, seconds = {}, {}, Counter()
        for k, (start, duration, speaker, voice, index) in enumerate(kept):
            a = round(voice.offsets[index] * RATE)
            clip = sf.read(voice.track, start=a, frames=round(duration * RATE), dtype="float32", always_2d=True)[0][:, 0]
            levels[k] = 10 * np.log10(max(float(np.mean(clip.astype(np.float64) ** 2)), 1e-12))
            seconds[speaker] += duration
        total = sum(seconds.values())
        conversations = len({item[3].key[1] for item in kept})
        voice_level = {s: 10 * np.log10(np.mean([10 ** (levels[k] / 10) for k, item in enumerate(kept) if item[2] == s])) for s in seconds}
        features = {}
        window_keys = {item[3].key for item in kept}
        leak_class = {}
        for k, (start, duration, speaker, voice, index) in enumerate(kept):
            before = previous.get(speaker)
            if before is None:
                gap, jump, source = "first", "first", "first"
            else:
                gap = bucket(start - before[0], (0.5, 2, 5, 15, 40))
                step = index - before[1]
                jump = "next" if step == 1 else ("ahead<=8" if 1 < step <= 8 else "far")
                source = bucket(abs(float(voice.starts[index] - voice.starts[before[1]])), (10, 30, 120, 600))
            previous[speaker] = (start + duration, index)
            row = by_key[(voice.track, round(float(voice.offsets[index]), 3))]
            shares = {"foreign": [], "own": []}
            for o in by_meeting[row["meeting"]]:
                if o["speaker"] == row["speaker"]:
                    continue
                a, b = max(o["source_start"], row["source_start"]), min(o["source_end"], row["source_end"])
                if a < b:
                    shares["own" if (o["corpus"], o["meeting"], o["speaker"]) in window_keys else "foreign"].append((a, b))
            leak_share = {}
            for name, spans in shares.items():
                covered, end = 0.0, -1e9
                for a, b in sorted(spans):
                    covered += max(0.0, b - max(a, end))
                    end = max(end, b)
                leak_share[name] = covered / max(row["source_end"] - row["source_start"], 1e-6)
            leak_class[k] = leak_share["foreign"] > 0
            features[k] = {"duration": bucket(duration, (1, 2, 4, 8, 15)), "gap_to_previous": gap, "order_jump": jump,
                           "source_seconds_jump": source,
                           "level_vs_voice_db": bucket(levels[k] - voice_level[speaker], (-12, -6, -3, 0, 3)),
                           "voice_share": bucket(seconds[speaker] / total, (0.05, 0.1, 0.2, 0.4)),
                           "conversations_in_window": str(conversations),
                           "speakers_in_window": str(len(seconds))}
            if len(seconds) < 4:
                band = lambda v: "none" if v == 0 else bucket(v, (0.1, 0.3, 0.7))  # noqa: E731
                features[k]["lt4voices_foreign_leak_share"] = band(leak_share["foreign"])
                features[k]["lt4voices_own_leak_share"] = band(leak_share["own"])
                features[k]["lt4voices_leak_kind"] = ("both" if leak_share["own"] > 0 else "foreign") if leak_share["foreign"] > 0 \
                    else ("own" if leak_share["own"] > 0 else "none")
        for row in decoded:
            hyp = sc.turns_of(row.get("prediction"))
            mapping, _, _ = sc.speaker_mapping(ref, hyp)
            mapped = set(mapping.values())
            nh = len({h[0] for h in hyp})
            window_rows.append({"id": r["id"], "refs": len(seconds), "hyps": nh, "conversations": conversations})
            split_by_leak = {True: [0, 0], False: [0, 0]}
            for k, t in enumerate(ref):
                n = len(t[3])
                if n <= 3:
                    continue
                near = [h for h in hyp if h[1] < t[2] + sc.TEXT_MARGIN and t[1] - sc.TEXT_MARGIN < h[2] and h[3]]
                best = min(near, key=lambda h: sc.semi_global(t[3], h[3]), default=None)
                split = bool(best and sc.semi_global(t[3], best[3]) / n <= MATCH and best[0] not in mapped)
                for name, value in features[k].items():
                    cell = tables[name][value]
                    cell[0] += n
                    cell[1] += n * split
                stats = per_voice["/".join(kept[k][3].key[1:])]
                stats[0] += n
                stats[1] += n * split
                split_by_leak[leak_class[k]][0] += n
                split_by_leak[leak_class[k]][1] += n * split
            if len(seconds) < 4 and split_by_leak[True][1] + split_by_leak[False][1] and split_by_leak[True][0] and split_by_leak[False][0]:
                paired.append(split_by_leak[True][1] / split_by_leak[True][0] - split_by_leak[False][1] / split_by_leak[False][0])
    report = {"windows_checked": checked}
    all_units = sum(v[0] for v in tables["duration"].values())
    all_split = sum(v[1] for v in tables["duration"].values())
    report["split_pct_overall"] = round(100 * all_split / max(all_units, 1), 2)
    for name, table in tables.items():
        report[name] = {k: {"units_share": round(v[0] / all_units, 3), "split_pct": round(100 * v[1] / max(v[0], 1), 2)}
                        for k, v in sorted(table.items())}
    voices = sorted(per_voice.items(), key=lambda kv: -kv[1][1])
    report["paired_split_rate_foreign_minus_clean"] = {"decodes": len(paired), "mean_pct": round(100 * float(np.mean(paired)), 2) if paired else None,
                                                       "positive_share": round(float(np.mean([d > 0 for d in paired])), 3) if paired else None}
    report["voices"] = {"count": len(voices), "with_split": sum(v[1] > 0 for _, v in voices),
                        "top20_share_of_split": round(sum(v[1] for _, v in voices[:20]) / max(all_split, 1), 3),
                        "top": [{"voice": k, "units": v[0], "split_pct": round(100 * v[1] / v[0], 1)} for k, v in voices[:25]]}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
