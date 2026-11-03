"""Do re-voiced windows over-count speakers because of who else was in the material's source conversation?

v1 fidelity decodes, windows re-planned as built (checked against the decoded reference). For each window:
  foreign leak s   seconds of placed speech during which, in the source meeting, a speaker who has no voice in this window
                   was talking (a voice the model has no label for)
  own leak s       the same for speakers who do have a voice in this window
  source speakers  how many speakers the voices' source conversation has
Over-count = hypothesis speakers > reference speakers. For every extra speaker's first turn (the turn that opens the slot),
whether it lies in a foreign-leak span and whether its text matches what was said there or a labelled turn nearby."""
import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from audio_data_contract import resolve_split

from open_audio_llm.data import meeting_online as mo
from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoints as sc  # noqa: E402

RATE, MATCH = 16000, 0.34


def units(text):
    return "".join(sc.UNIT.findall(text)).lower()


def leak_spans(row, others, start):
    """Spans (window seconds) of placed utterance ``row`` at ``start`` during which ``others`` talk in the source."""
    spans = []
    for o in others:
        a, b = max(o["source_start"], row["source_start"]), min(o["source_end"], row["source_end"])
        if a < b:
            spans.append((start + a - row["source_start"], start + b - row["source_start"], units(o["text"])))
    return spans


def merged_seconds(spans):
    total, end = 0.0, -1e9
    for a, b, *_ in sorted(spans):
        total += max(0.0, b - max(a, end))
        end = max(end, b)
    return total


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
        by_meeting[(r["corpus"], r["meeting"])].append(r)
    material = mo.load_material(rows, config, track_of)
    records = [json.loads(line) for line in gzip.open(args.panel_records, "rt")]
    predictions = defaultdict(lambda: defaultdict(list))
    for path in args.predictions:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            label = row.get("label", "")
            if label.endswith(f"_{args.corpus}") and label.startswith("syn_fill_"):
                predictions[label.split("_")[2]][row["id"].split("/", 1)[1]].append(row)

    windows = []
    for r in records:
        length = round(r["audio_slots"][0]["ref"]["duration"] * RATE) / RATE
        turns = sorted(r["metadata"]["turns"], key=lambda t: t["start"])
        template = [[t["source_speaker"], t["start"], t["end"], len(mo.CHARS_PATTERN.sub("", t["text"]))] for t in turns]
        _, placed = mo.plan_meeting(random.Random(f"{args.seed}:fill:{args.corpus}:{r['id']}"), template, length, material, config)
        target, _ = mo.meeting_target(placed)
        keys = {item[3].key for item in placed}
        foreign, own = [], []
        for start, duration, _, voice, index in placed:
            row = by_key[(voice.track, round(float(voice.offsets[index]), 3))]
            others = [o for o in by_meeting[(row["corpus"], row["meeting"])] if o["speaker"] != row["speaker"]]
            foreign += leak_spans(row, [o for o in others if (o["corpus"], o["meeting"], o["speaker"]) not in keys], start)
            own += leak_spans(row, [o for o in others if (o["corpus"], o["meeting"], o["speaker"]) in keys], start)
        source_speakers = max(len({o["speaker"] for o in by_meeting[k[:2]]}) for k in keys)
        windows.append({"id": r["id"], "target": target, "refs": len(keys), "source_speakers": source_speakers,
                        "conversations": len({k[1] for k in keys}), "foreign": foreign, "foreign_s": merged_seconds(foreign),
                        "own_s": merged_seconds(own)})

    report = {}
    for kind, decoded in predictions.items():
        table = {"by_refs": defaultdict(Counter), "by_refs_source": defaultdict(Counter), "by_foreign_seconds": defaultdict(Counter),
                 "by_own_seconds_when_no_foreign": defaultdict(Counter)}
        seeds = Counter()
        examples = []
        for w in windows:
            ref = sc.turns_of(w["target"])
            for row in decoded.get(w["id"], []):
                if sc.turns_of(row["reference"]) != ref:
                    raise SystemExit(f"re-planned window differs from the decoded reference: {w['id']}")
                hyp = sc.turns_of(row.get("prediction"))
                nh = len({h[0] for h in hyp})
                over = nh > w["refs"]
                keys = {"by_refs": str(w["refs"]),
                        "by_refs_source": f"{w['refs']} voices, source {w['source_speakers']} speakers, {w['conversations']} conv"}
                if w["refs"] < 4:
                    keys["by_foreign_seconds"] = f"{w['refs']} voices, foreign {bucket(w['foreign_s'], (1, 5, 10, 20, 40))}"
                    if w["foreign_s"] == 0:
                        keys["by_own_seconds_when_no_foreign"] = f"{w['refs']} voices, own {bucket(w['own_s'], (1, 5, 10, 20, 40))}"
                for name, key in keys.items():
                    table[name][key]["decodes"] += 1
                    table[name][key]["over"] += over
                    table[name][key]["hyp_speakers"] += nh
                mapping, _, _ = sc.speaker_mapping(ref, hyp)
                for speaker in {h[0] for h in hyp} - set(mapping.values()):
                    mine = [h for h in hyp if h[0] == speaker]
                    first = min(mine, key=lambda h: h[1])
                    seeds["extra_speakers"] += 1
                    seeds["extra_units"] += sum(len(h[3]) for h in mine)
                    seeds["extra_speakers_all_turns_le_2_units"] += all(len(h[3]) <= 2 for h in mine)
                    in_foreign = [s for s in w["foreign"] if s[0] - 0.3 < first[2] and first[1] < s[1] + 0.3]
                    seeds["first_in_foreign_span"] += bool(in_foreign)
                    text = first[3]
                    if text:
                        labelled = [t[3] for t in ref if t[1] < first[2] + 2 and first[1] - 2 < t[2]]
                        lab = min((sc.semi_global(text, c) for c in labelled if c), default=len(text)) / len(text)
                        par = min((sc.semi_global(text, s[2]) for s in in_foreign if s[2]), default=len(text)) / len(text)
                        seeds["first_matches_labelled"] += lab <= MATCH
                        seeds["first_matches_foreign_only"] += par <= MATCH and lab > MATCH
                        seeds["first_matches_neither"] += par > MATCH and lab > MATCH
                        seeds["first_le_2_units"] += len(text) <= 2
                    if len(examples) < 16 and len(mine) <= 40:
                        examples.append({"id": w["id"], "turns": [[h[1], h[2], h[3][:24]] for h in sorted(mine, key=lambda h: h[1])[:5]],
                                         "foreign_at_first": [s[2][:24] for s in in_foreign][:3]})
        report[kind] = {name: {k: {"decodes": v["decodes"], "over_pct": round(100 * v["over"] / max(v["decodes"], 1), 1),
                                   "mean_hyp_speakers": round(v["hyp_speakers"] / max(v["decodes"], 1), 2)}
                               for k, v in sorted(t.items())} for name, t in table.items()}
        report[kind]["extra_speaker_first_turns"] = dict(seeds)
        report[kind]["examples"] = examples
    report["windows"] = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in w.items() if k not in ("target", "foreign")} for w in windows]
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    for kind in predictions:
        print("==", kind)
        for name, t in report[kind].items():
            if name == "examples":
                for e in t[:8]:
                    print("  ", json.dumps(e, ensure_ascii=False))
            elif name == "extra_speaker_first_turns":
                print(name, t)
            else:
                print(name)
                for k, v in t.items():
                    print("  ", k, v)


if __name__ == "__main__":
    main()
