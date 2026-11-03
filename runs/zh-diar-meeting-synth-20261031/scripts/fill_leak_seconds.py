"""Unlabelled partner speech in seconds rather than as a share of the utterance.

Re-plans the windows the v1 fidelity check decoded (max_overlap 1.0, as it was built) and pools its decodes: text error of
reference lines nobody overlaps in the window, by the seconds of source overlap in their material and by line length;
speaker over-count by the window's total source-overlap seconds and by its longest single overlapped stretch. Also, per
candidate limit on overlapped seconds per utterance: eligible hours, fill coverage and the window's overlap time."""
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

from open_audio_llm.data import meeting_online as mo
from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_coverage as fc  # noqa: E402
import overlap_breakdown as ob  # noqa: E402
import select_checkpoints as sc  # noqa: E402

LINE_EDGES = (0.0, 1e-9, 0.5, 1.0, 2.0, 5.0, float("inf"))
WINDOW_EDGES = (0.0, 5.0, 15.0, 30.0, 60.0, float("inf"))
LONGEST_EDGES = (0.0, 1.0, 2.0, 5.0, 10.0, float("inf"))
LENGTHS = ((0.0, 3.0, "<3s"), (3.0, 8.0, "3-8s"), (8.0, float("inf"), ">=8s"))


def edge(edges, value):
    k = int(np.digitize(value, edges)) - 1
    return f"[{edges[k]:g},{edges[k + 1]:g})"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-records", type=Path, required=True)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--predictions", nargs="+", type=Path, required=True)
    p.add_argument("--corpus", default="alimeeting")
    p.add_argument("--limits", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0, 5.0])
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
    records = [json.loads(line) for line in gzip.open(args.panel_records, "rt")]
    predictions = defaultdict(list)
    for path in args.predictions:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            predictions[row["id"]].append(row.get("prediction"))
    material = mo.load_material(rows, config, track_of)
    report = {}
    for kind in ("near", "far"):
        lines = defaultdict(lambda: [0, 0])
        total, longest = defaultdict(list), defaultdict(list)
        for r in records:
            rid = r["id"]
            length = round(r["audio_slots"][0]["ref"]["duration"] * 16000) / 16000
            turns = sorted(r["metadata"]["turns"], key=lambda t: t["start"])
            template = [[t["source_speaker"], t["start"], t["end"], len(mo.CHARS_PATTERN.sub("", t["text"]))] for t in turns]
            _, placed = mo.plan_meeting(random.Random(f"{args.seed}:fill:{args.corpus}:{rid}"), template, length, material, config)
            target, mapping = mo.meeting_target(placed)
            source, leaks = {}, []
            for start, duration, speaker, voice, index in placed:
                row = by_key[(voice.track, round(float(voice.offsets[index]), 3))]
                source[(mapping[speaker], round(start, 2), round(start + duration, 2))] = row
                leaks.append(row["overlap"] * row["duration"])
            ref = sc.turns_of(target)
            for text in predictions.get(f"syn_fill_{kind}_{args.corpus}/{rid}", []):
                hyp = sc.turns_of(text)
                errors = sc.text_errors(ref, hyp)
                for k, t in enumerate(ref):
                    row = source.get((t[0], round(t[1], 2), round(t[2], 2)))
                    if row is None or len(t[3]) <= 3 or ob.category(t, ref) != "none":
                        continue
                    span = t[2] - t[1]
                    name = next(n for a, b, n in LENGTHS if a <= span < b)
                    for key in ((edge(LINE_EDGES, row["overlap"] * row["duration"]), "all"),
                                (edge(LINE_EDGES, row["overlap"] * row["duration"]), name)):
                        lines[key][0] += errors[k]
                        lines[key][1] += len(t[3])
                over = len({h[0] for h in hyp}) - len(mapping)
                total[edge(WINDOW_EDGES, sum(leaks))].append(over)
                longest[edge(LONGEST_EDGES, max(leaks, default=0.0))].append(over)
        report[kind] = {
            "none_line_text_error_by_leak_seconds": {f"{a} / {b}": {"units": u, "error_rate": round(e / max(u, 1), 4)}
                                                     for (a, b), (e, u) in sorted(lines.items())},
            "speaker_overcount_by_window_leak_seconds": {k: {"decodes": len(v), "mean": round(float(np.mean(v)), 3)} for k, v in sorted(total.items())},
            "speaker_overcount_by_longest_leak_seconds": {k: {"decodes": len(v), "mean": round(float(np.mean(v)), 3)} for k, v in sorted(longest.items())},
        }
        print(kind, json.dumps(report[kind], ensure_ascii=False), flush=True)
    candidates = {}
    for limit in args.limits:
        # max_overlap has no seconds form yet: mark the rows so that the existing share test applies the limit.
        marked = [{**r, "overlap": 0.0 if r["overlap"] * r["duration"] <= limit else 1.0} for r in rows if r["corpus"] == args.corpus]
        chosen = replace(config, max_overlap=0.0)
        mat = mo.load_material(marked, chosen, track_of)
        eligible = [r for r in marked if r["chars"] <= mo.BACKCHANNEL_CHARS or r["overlap"] == 0.0]
        placed_s, template_s, placed_o, template_o = 0.0, 0.0, [], []
        placed_by, template_by = np.zeros(4), np.zeros(4)
        for r in records:
            length = round(r["audio_slots"][0]["ref"]["duration"] * 16000) / 16000
            turns = sorted(r["metadata"]["turns"], key=lambda t: t["start"])
            template = [[t["source_speaker"], t["start"], t["end"], len(mo.CHARS_PATTERN.sub("", t["text"]))] for t in turns]
            _, placed = mo.plan_meeting(random.Random(f"{args.seed}:fill:{args.corpus}:{r['id']}"), template, length, mat, chosen)
            spans = [(s, s + d) for s, d, *_ in placed]
            placed_s += sum(d for _, d, *_ in placed)
            template_s += sum(t[2] - t[1] for t in template)
            placed_o.append(fc.overlap_share(spans, length))
            template_o.append(fc.overlap_share([(t[1], t[2]) for t in template], length))
            for t in template:
                k = int(np.digitize(t[2] - t[1], fc.TURN_BINS)) - 1
                template_by[k] += t[2] - t[1]
                inside = sum(max(0.0, min(b, t[2]) - max(a, t[1])) for (a, b), (_, _, spk, *_) in zip(spans, placed) if spk == t[0])
                placed_by[k] += min(inside, t[2] - t[1])
        candidates[str(limit)] = {
            "eligible_hours": round(sum(r["duration"] for r in eligible) / 3600, 1),
            "placed_over_template_seconds": round(placed_s / template_s, 3),
            "filled_share_by_turn_length": {fc.label(fc.TURN_BINS, k): round(placed_by[k] / max(template_by[k], 1e-9), 3) for k in range(4)},
            "overlap_time_share": {"placed": round(float(np.mean(placed_o)), 3), "template": round(float(np.mean(template_o)), 3)},
        }
        print("limit", limit, json.dumps(candidates[str(limit)], ensure_ascii=False), flush=True)
    report["candidates"] = candidates
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
