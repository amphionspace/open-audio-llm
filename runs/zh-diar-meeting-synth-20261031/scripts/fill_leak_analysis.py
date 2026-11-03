"""Do re-voiced dev windows fail where their material carries unlabelled partner speech?

Re-plans every syn_fill window exactly as build_dev_fidelity.py (same seeds, same config) to recover the material
utterance behind each reference line, then pools the decodes of the fidelity check:
  - per reference line that nobody overlaps in the synthesized window: text error by how much other speakers of the
    material's own conversation overlapped that utterance (their speech leaks into the headset clip, unlabelled);
  - per window: predicted minus reference speakers by the overlapped share of its placed material seconds.
"""
import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from audio_data_contract import resolve_split

from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver
from open_audio_llm.data import meeting_online as mo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlap_breakdown as ob  # noqa: E402
import select_checkpoints as sc  # noqa: E402

RATE = 16000
BUCKETS = ((0.0, 0.0, "0"), (1e-9, 0.5, "(0,0.5)"), (0.5, 1.0, "[0.5,1]"))


def bucket(value):
    return next(name for low, high, name in BUCKETS if low <= value <= high)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-records", type=Path, required=True)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--predictions", nargs="+", type=Path, required=True)
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
    material_rows = list(mo.read_material_rows(resolve_split(
        resolver.catalog, m["dataset_id"], m["version"], m["split"], "meeting_material", resolver.roots)))
    track_of = lambda row: tracks[f"{row['corpus']}/{row['meeting']}/{row['speaker']}"]  # noqa: E731
    by_key = {(track_of(r), round(r["offset"], 3)): r for r in material_rows}
    records = [json.loads(line) for line in gzip.open(args.panel_records, "rt")]
    predictions = defaultdict(list)
    for path in args.predictions:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            predictions[row["id"]].append(row.get("prediction"))
    report = {}
    for corpus in args.corpora:
        material = mo.load_material(material_rows, replace(config, corpus_weights={corpus: 1.0}), track_of)
        lines = {kind: defaultdict(lambda: [0, 0]) for kind in ("near", "far")}
        windows = {kind: defaultdict(list) for kind in ("near", "far")}
        for r in records:
            rid = r["id"]
            length = round(r["audio_slots"][0]["ref"]["duration"] * RATE)
            turns = sorted(r["metadata"]["turns"], key=lambda t: t["start"])
            template = [[t["source_speaker"], t["start"], t["end"], len(mo.CHARS_PATTERN.sub("", t["text"]))] for t in turns]
            _, placed = mo.plan_meeting(random.Random(f"{args.seed}:fill:{corpus}:{rid}"), template, length / RATE,
                                        material, config)
            target, mapping = mo.meeting_target(placed)
            source = {}
            seconds, overlapped = 0.0, 0.0
            for start, duration, speaker, voice, index in placed:
                row = by_key[(voice.track, round(float(voice.offsets[index]), 3))]
                key = (mapping[speaker], round(start, 2), round(start + duration, 2))
                source[key] = row
                seconds += duration
                overlapped += duration * row["overlap"]
            ref = sc.turns_of(target)
            for kind in ("near", "far"):
                for hyp_text in predictions.get(f"syn_fill_{kind}_{corpus}/{rid}", []):
                    hyp = sc.turns_of(hyp_text)
                    errors = sc.text_errors(ref, hyp)
                    for k, t in enumerate(ref):
                        row = source.get((t[0], round(t[1], 2), round(t[2], 2)))
                        if row is None or len(t[3]) <= 3 or ob.category(t, ref) != "none":
                            continue
                        cell = lines[kind][(bucket(row["overlap"]), row.get("origin", "human"))]
                        cell[0] += errors[k]
                        cell[1] += len(t[3])
                    speakers = len({h[0] for h in hyp})
                    windows[kind][bucket(overlapped / max(seconds, 1e-9))].append(speakers - len(mapping))
        report[corpus] = {
            kind: {"none_line_text_error": {f"overlap {b} / {o}": {"units": u, "error_rate": round(e / max(u, 1), 4)}
                                            for (b, o), (e, u) in sorted(lines[kind].items())},
                   "speaker_overcount_by_window_overlapped_share": {
                       b: {"windows": len(v), "mean": round(float(np.mean(v)), 3)} for b, v in sorted(windows[kind].items())}}
            for kind in ("near", "far")}
        print(corpus, json.dumps(report[corpus], ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
