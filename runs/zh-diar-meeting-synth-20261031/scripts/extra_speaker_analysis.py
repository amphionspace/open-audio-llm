"""What do the speakers the model adds on re-voiced dev windows say?

For every hypothesis speaker left unmapped by the cpER speaker mapping (v1 fidelity decodes, windows re-planned as built),
each of its turns of at least MIN_UNITS units is matched, within MARGIN s, against
  labelled   the reference turns (the model split a labelled voice),
  partner    what other speakers of the material's source conversation said over the utterances placed there (leak),
  control    the same partner text taken from another window (chance level of the partner match).
A turn counts as a match when semi-global edits / its units <= MATCH."""
import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import yaml
from audio_data_contract import resolve_split

from open_audio_llm.data import meeting_online as mo
from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoints as sc  # noqa: E402

MIN_UNITS, MARGIN, MATCH = 3, 3.0, 0.34


def units(text):
    return "".join(sc.UNIT.findall(text)).lower()


def rate(query, candidates):
    return min((sc.semi_global(query, c) for c in candidates if c), default=len(query)) / len(query)


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
        by_meeting[r["meeting"]].append(r)
    material = mo.load_material(rows, config, track_of)
    records = [json.loads(line) for line in gzip.open(args.panel_records, "rt")]
    predictions = defaultdict(list)
    for path in args.predictions:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            predictions[row["id"]].append(row.get("prediction"))
    windows = []
    for r in records:
        length = round(r["audio_slots"][0]["ref"]["duration"] * 16000) / 16000
        turns = sorted(r["metadata"]["turns"], key=lambda t: t["start"])
        template = [[t["source_speaker"], t["start"], t["end"], len(mo.CHARS_PATTERN.sub("", t["text"]))] for t in turns]
        _, placed = mo.plan_meeting(random.Random(f"{args.seed}:fill:{args.corpus}:{r['id']}"), template, length, material, config)
        target, _ = mo.meeting_target(placed)
        heard = []
        for start, duration, _, voice, index in placed:
            row = by_key[(voice.track, round(float(voice.offsets[index]), 3))]
            partners = [units(o["text"]) for o in by_meeting[row["meeting"]]
                        if o["speaker"] != row["speaker"] and o["source_start"] < row["source_end"] and row["source_start"] < o["source_end"]]
            heard.append((start, start + duration, "".join(partners)))
        windows.append((r["id"], sc.turns_of(target), heard))
    report = {}
    for kind in ("near", "far"):
        counts, examples, decoded, with_extra, extra_units, all_units = Counter(), [], 0, 0, 0, 0
        for w, (rid, ref, heard) in enumerate(windows):
            other = windows[(w + len(windows) // 2) % len(windows)][2]
            for text in predictions.get(f"syn_fill_{kind}_{args.corpus}/{rid}", []):
                hyp = sc.turns_of(text)
                decoded += 1
                mapping, _, _ = sc.speaker_mapping(ref, hyp)
                extra = {h[0] for h in hyp} - set(mapping.values())
                with_extra += bool(extra)
                all_units += sum(len(h[3]) for h in hyp)
                for h in hyp:
                    if h[0] not in extra:
                        continue
                    extra_units += len(h[3])
                    if len(h[3]) < MIN_UNITS:
                        counts["too short"] += len(h[3])
                        continue
                    near = lambda spans: [c for a, b, c in spans if a < h[2] + MARGIN and h[1] - MARGIN < b]  # noqa: E731
                    labelled = [t[3] for t in ref if t[1] < h[2] + MARGIN and h[1] - MARGIN < t[2]]
                    own = rate(h[3], labelled + ["".join(labelled)])
                    partner = rate(h[3], near(heard))
                    control = rate(h[3], near(other))
                    counts["control match"] += len(h[3]) * (control <= MATCH)
                    if own <= MATCH and own <= partner:
                        counts["labelled"] += len(h[3])
                    elif partner <= MATCH:
                        counts["partner"] += len(h[3])
                        if len(examples) < 12:
                            examples.append({"id": rid, "hyp": h[3][:60], "partner_rate": round(partner, 2), "own_rate": round(own, 2)})
                    else:
                        counts["neither"] += len(h[3])
                        if len(examples) < 24 and w % 7 == 0:
                            examples.append({"id": rid, "hyp": h[3][:60], "neither": True})
        report[kind] = {"decodes": decoded, "decodes_with_extra_speaker": with_extra, "extra_units": extra_units,
                        "extra_share_of_hyp_units": round(extra_units / max(all_units, 1), 4),
                        "extra_units_by_source": {k: {"units": int(v), "share": round(v / max(extra_units, 1), 3)} for k, v in counts.most_common()},
                        "examples": examples}
        print(kind, json.dumps({k: v for k, v in report[kind].items() if k != "examples"}, ensure_ascii=False), flush=True)
        for e in examples[:10]:
            print("   ", json.dumps(e, ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
