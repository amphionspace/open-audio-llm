"""How much of each dev template a re-voicing fills at each material overlap limit (planning only, no audio).

Same plans as build_dev_fidelity.py. Per corpus and max_overlap: placed over template speech seconds, by template turn
length; share of window time with two or more speakers, placed against template; and the eligible material's seconds
by utterance length (a long turn rarely has no overlap, so a strict limit removes long utterances first)."""
import argparse
import gzip
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from audio_data_contract import resolve_split

from open_audio_llm.data import meeting_online as mo
from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver

STEP = 0.02
TURN_BINS = (0.0, 2.0, 5.0, 10.0, float("inf"))
UTTERANCE_BINS = (0.0, 2.0, 5.0, 10.0, float("inf"))


def label(bins, k):
    return f"{bins[k]:g}-{bins[k + 1]:g}s"


def overlap_share(spans, length):
    count = np.zeros(int(length / STEP) + 1, dtype=np.int16)
    for a, b in spans:
        count[int(a / STEP): int(b / STEP)] += 1
    return float((count >= 2).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-records", type=Path, required=True)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--limits", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
    p.add_argument("--corpora", nargs="+", default=["alimeeting", "aishell5"])
    p.add_argument("--seed", default="fidelity-20261031")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = yaml.safe_load(args.data_config.read_text())
    config = mo.OnlineMeetingConfig(**data["online_meeting"])
    resolver = LhotseCatalogAudioResolver(data["catalog"], data["roots"])
    m = config.material
    tracks = {}
    for path in resolve_split(resolver.catalog, m["dataset_id"], m["version"], m["split"], "audio_index", resolver.roots):
        for row in mo.read_material_rows([path]):
            tracks[row["cut_id"]] = str(Path(resolver.roots[row["root_alias"]]) / row["relative_path"])
    rows = list(mo.read_material_rows(resolve_split(
        resolver.catalog, m["dataset_id"], m["version"], m["split"], "meeting_material", resolver.roots)))
    track_of = lambda row: tracks[f"{row['corpus']}/{row['meeting']}/{row['speaker']}"]  # noqa: E731
    records = [json.loads(line) for line in gzip.open(args.panel_records, "rt")]
    report = {}
    for corpus in args.corpora:
        report[corpus] = {}
        for limit in args.limits:
            chosen = replace(config, max_overlap=limit, corpus_weights={corpus: 1.0})
            material = mo.load_material(rows, chosen, track_of)
            eligible = [r for r in rows if r["corpus"] == corpus and (r["chars"] <= mo.BACKCHANNEL_CHARS or r["overlap"] <= limit)]
            seconds = np.array([r["duration"] for r in eligible])
            placed_by, template_by = np.zeros(4), np.zeros(4)
            placed_seconds, template_seconds, placed_overlap, template_overlap = 0.0, 0.0, [], []
            for r in records:
                length = r["audio_slots"][0]["ref"]["duration"]
                turns = sorted(r["metadata"]["turns"], key=lambda t: t["start"])
                template = [[t["source_speaker"], t["start"], t["end"], len(mo.CHARS_PATTERN.sub("", t["text"]))] for t in turns]
                _, placed = mo.plan_meeting(random.Random(f"{args.seed}:fill:{corpus}:{r['id']}"), template,
                                            round(length * 16000) / 16000, material, chosen)
                spans = [(s, s + d) for s, d, *_ in placed]
                placed_seconds += sum(d for _, d, *_ in placed)
                template_seconds += sum(t[2] - t[1] for t in template)
                placed_overlap.append(overlap_share(spans, length))
                template_overlap.append(overlap_share([(t[1], t[2]) for t in template], length))
                for t in template:
                    k = int(np.digitize(t[2] - t[1], TURN_BINS)) - 1
                    template_by[k] += t[2] - t[1]
                    inside = sum(max(0.0, min(b, t[2]) - max(a, t[1])) for (a, b), (_, _, spk, *_) in zip(spans, placed)
                                 if spk == t[0])
                    placed_by[k] += min(inside, t[2] - t[1])
            report[corpus][str(limit)] = {
                "placed_over_template_seconds": round(placed_seconds / template_seconds, 3),
                "filled_share_by_turn_length": {label(TURN_BINS, k): round(placed_by[k] / max(template_by[k], 1e-9), 3) for k in range(4)},
                "template_seconds_share_by_turn_length": {label(TURN_BINS, k): round(template_by[k] / template_by.sum(), 3) for k in range(4)},
                "overlap_time_share": {"placed": round(float(np.mean(placed_overlap)), 3), "template": round(float(np.mean(template_overlap)), 3)},
                "eligible_hours": round(float(seconds.sum()) / 3600, 1),
                "eligible_hours_by_utterance_length": {
                    label(UTTERANCE_BINS, k): round(float(seconds[(seconds >= UTTERANCE_BINS[k]) & (seconds < UTTERANCE_BINS[k + 1])].sum()) / 3600, 1)
                    for k in range(4)},
            }
            print(corpus, limit, json.dumps(report[corpus][str(limit)], ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
