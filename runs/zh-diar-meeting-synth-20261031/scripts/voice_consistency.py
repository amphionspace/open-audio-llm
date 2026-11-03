"""Are the utterances the model splits off really less like the rest of their voice? (ERes2NetV2, 3D-Speaker)

For windows of 2-3 speakers of the close-talk fills (v1 AliMeeting / AISHELL-5 material, Eval material from 4-person and
2-person meetings, plans rebuilt as built and checked against the decoded references), every placed utterance of >= 1 s is
embedded from its material clip alone. consistency = cosine to the mean of the voice's other utterances in the window.
Reports, per variant: consistency by source size; over-count by the window's lowest voice consistency and by its most
similar pair of voices; and, within a voice, consistency of split-off against kept utterances (paired by decode)."""
import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import glob
import numpy as np
import soundfile as sf
import torch
import yaml
from audio_data_contract import resolve_split
from modelscope.pipelines import pipeline

from open_audio_llm.data import meeting_online as mo
from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoints as sc  # noqa: E402

RATE, MATCH = 16000, 0.34
FOUR_PERSON = ("R8001_M8004", "R8003_M8001", "R8007_M8010", "R8007_M8011")
TWO_PERSON = ("R8009_M8018", "R8009_M8019", "R8009_M8020")
MIN_SECONDS, MAX_SECONDS = 1.0, 10.0


def bucket(value, edges):
    for edge in edges:
        if value < edge:
            return f"<{edge}"
    return f">={edges[-1]}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel-records", type=Path, required=True)
    p.add_argument("--near-dir", type=Path, required=True)
    p.add_argument("--data-config", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--fidelity-predictions", nargs="+", type=Path, required=True)
    p.add_argument("--split-predictions", nargs="+", type=Path, required=True)
    p.add_argument("--seed", default="fidelity-20261031")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = yaml.safe_load(args.data_config.read_text())
    base = mo.OnlineMeetingConfig(**data["online_meeting"])
    resolver = LhotseCatalogAudioResolver(data["catalog"], data["roots"])
    m = base.material
    tracks = {}
    for path in resolve_split(resolver.catalog, m["dataset_id"], m["version"], m["split"], "audio_index", resolver.roots):
        for row in mo.read_material_rows([path]):
            tracks[row["cut_id"]] = str(Path(resolver.roots[row["root_alias"]]) / row["relative_path"])
    rows = list(mo.read_material_rows(resolve_split(resolver.catalog, m["dataset_id"], m["version"], m["split"], "meeting_material", resolver.roots)))
    track_of = lambda row: tracks[f"{row['corpus']}/{row['meeting']}/{row['speaker']}"]  # noqa: E731
    train = {c: mo.load_material(rows, replace(base, corpus_weights={c: 1.0}), track_of) for c in ("alimeeting", "aishell5")}
    sizes = defaultdict(set)
    for r in rows:
        sizes[r["meeting"]].add(r["speaker"])
    records = [json.loads(line) for line in gzip.open(args.panel_records, "rt")]
    by_source = defaultdict(dict)
    for r in records:
        first = r["metadata"]["source_record"]["start"]
        for t in r["metadata"]["turns"]:
            by_source[r["metadata"]["meeting_id"]][(t["source_speaker"], round(first + t["start"], 4), round(first + t["end"], 4))] = t["text"]
    eval_rows = []
    for meeting, turns in by_source.items():
        counts = defaultdict(int)
        for (speaker, a, b), text in sorted(turns.items()):
            eval_rows.append({"corpus": "eval", "meeting": meeting, "speaker": speaker, "index": counts[speaker], "offset": a,
                              "duration": round(b - a, 5), "text": text, "chars": len(mo.CHARS_PATTERN.sub("", text)),
                              "origin": "human", "overlap": 0.0, "source_start": a, "source_end": b})
            counts[speaker] += 1
            sizes[meeting].add(speaker)
    eval_config = replace(base, corpus_weights={"eval": 1.0})
    near = lambda row: glob.glob(f"{args.near_dir}/{row['meeting']}_N_SPK{row['speaker']}.wav")[0]  # noqa: E731
    eval_material = {c[0].key[1]: c for c in mo.load_material(eval_rows, eval_config, near)["eval"]}

    predictions = defaultdict(lambda: defaultdict(list))
    for files in (args.fidelity_predictions, args.split_predictions):
        for path in files:
            for line in path.read_text().splitlines():
                row = json.loads(line)
                predictions[row["label"]][row["id"].split("/", 1)[1]].append(row)

    def plan(variant, r, template, length):
        if variant == "syn_fill_near_alimeeting":
            return mo.plan_meeting(random.Random(f"{args.seed}:fill:alimeeting:{r['id']}"), template, length, train["alimeeting"],
                                   replace(base, corpus_weights={"alimeeting": 1.0}))[1]
        if variant == "syn_fill_near_aishell5":
            return mo.plan_meeting(random.Random(f"{args.seed}:fill:aishell5:{r['id']}"), template, length, train["aishell5"],
                                   replace(base, corpus_weights={"aishell5": 1.0}))[1]
        names = FOUR_PERSON if variant == "unseen_src4" else TWO_PERSON
        pool = {"eval": [eval_material[n] for n in names if n != r["metadata"]["meeting_id"]]}
        return mo.plan_meeting(random.Random(f"{args.seed}:{variant}:{r['id']}"), template, length, pool, eval_config)[1]

    model = pipeline(task="speaker-verification", model=args.model, device="cuda:0" if torch.cuda.is_available() else "cpu")
    cache = {}

    def embed(voice, index):
        key = (voice.track, float(voice.offsets[index]))
        if key not in cache:
            duration = min(float(voice.durations[index]), MAX_SECONDS)
            clip = sf.read(voice.track, start=round(voice.offsets[index] * RATE), frames=round(duration * RATE), dtype="float32", always_2d=True)[0][:, 0]
            e = np.asarray(model([clip], output_emb=True)["embs"])[0]
            cache[key] = e / np.linalg.norm(e)
        return cache[key]

    report = {}
    for variant in ("syn_fill_near_alimeeting", "syn_fill_near_aishell5", "unseen_src4", "unseen_src2"):
        by_source_size = defaultdict(list)
        windows, paired, split_c, kept_c = [], [], [], []
        for r in records:
            decoded = predictions[variant].get(r["id"], [])
            turns = sorted(r["metadata"]["turns"], key=lambda t: t["start"])
            if not decoded or len({t["source_speaker"] for t in turns}) not in (2, 3):
                continue
            length = round(r["audio_slots"][0]["ref"]["duration"] * RATE) / RATE
            template = [[t["source_speaker"], t["start"], t["end"], len(mo.CHARS_PATTERN.sub("", t["text"]))] for t in turns]
            placed = plan(variant, r, template, length)
            target, _ = mo.meeting_target(placed)
            ref = sc.turns_of(target)
            if sc.turns_of(decoded[0]["reference"]) != ref:
                raise SystemExit(f"{variant} {r['id']}: re-planned window differs from the decoded reference")
            kept = [item for item in placed if round(item[0], 2) < round(item[0] + item[1], 2)]
            per_voice, consistency = defaultdict(list), {}
            for k, item in enumerate(kept):
                if item[1] >= MIN_SECONDS:
                    per_voice[item[2]].append(k)
            embeddings = {k: embed(kept[k][3], kept[k][4]) for ks in per_voice.values() for k in ks}
            means = {}
            for speaker, ks in per_voice.items():
                means[speaker] = np.mean([embeddings[k] for k in ks], axis=0)
                if len(ks) >= 2:
                    total = np.sum([embeddings[k] for k in ks], axis=0)
                    for k in ks:
                        rest = (total - embeddings[k]) / (len(ks) - 1)
                        consistency[k] = float(embeddings[k] @ (rest / np.linalg.norm(rest)))
            voice_mean = {s: float(np.mean([consistency[k] for k in ks if k in consistency])) for s, ks in per_voice.items() if any(k in consistency for k in ks)}
            names = sorted(means)
            pair = max((float(means[a] @ means[b] / np.linalg.norm(means[a]) / np.linalg.norm(means[b])) for i, a in enumerate(names) for b in names[i + 1:]), default=float("nan"))
            size = max(len(sizes[item[3].key[1]]) for item in kept)
            for s, v in voice_mean.items():
                by_source_size[str(size)].append(v)
            lowest = min(voice_mean.values()) if voice_mean else float("nan")
            nr = len({t[0] for t in ref})
            for row in decoded:
                hyp = sc.turns_of(row.get("prediction"))
                windows.append((len({h[0] for h in hyp}) > nr, lowest, pair, size))
                mapping, _, _ = sc.speaker_mapping(ref, hyp)
                mapped = set(mapping.values())
                split_rows, kept_rows = defaultdict(list), defaultdict(list)
                for k, t in enumerate(ref):
                    if k not in consistency or len(t[3]) <= 3:
                        continue
                    near_turns = [h for h in hyp if h[1] < t[2] + sc.TEXT_MARGIN and t[1] - sc.TEXT_MARGIN < h[2] and h[3]]
                    best = min(near_turns, key=lambda h: sc.semi_global(t[3], h[3]), default=None)
                    gone = bool(best and sc.semi_global(t[3], best[3]) / len(t[3]) <= MATCH and best[0] not in mapped)
                    (split_rows if gone else kept_rows)[kept[k][2]].append(consistency[k])
                    (split_c if gone else kept_c).append(consistency[k])
                for s in split_rows:
                    if kept_rows[s]:
                        paired.append(float(np.mean(split_rows[s]) - np.mean(kept_rows[s])))
        table = lambda index, edges: {k: {"decodes": len(v), "over_pct": round(100 * float(np.mean(v)), 1)} for k, v in sorted(  # noqa: E731
            {b: [w[0] for w in windows if not np.isnan(w[index]) and bucket(w[index], edges) == b] for b in {bucket(w[index], edges) for w in windows if not np.isnan(w[index])}}.items())}
        report[variant] = {"decodes": len(windows), "over_pct": round(100 * float(np.mean([w[0] for w in windows])), 1),
                           "voice_consistency_by_source_size": {k: {"voices": len(v), "mean": round(float(np.mean(v)), 3), "p10": round(float(np.percentile(v, 10)), 3)} for k, v in sorted(by_source_size.items())},
                           "over_by_lowest_voice_consistency": table(1, (0.4, 0.5, 0.6, 0.7)),
                           "over_by_most_similar_pair": table(2, (0.2, 0.35, 0.5, 0.65)),
                           "utterance_consistency": {"split": {"n": len(split_c), "mean": round(float(np.mean(split_c)), 3) if split_c else None},
                                                     "kept": {"n": len(kept_c), "mean": round(float(np.mean(kept_c)), 3) if kept_c else None},
                                                     "paired_split_minus_kept": {"n": len(paired), "mean": round(float(np.mean(paired)), 3) if paired else None,
                                                                                 "negative_share": round(float(np.mean([d < 0 for d in paired])), 3) if paired else None}}}
        print(variant, json.dumps(report[variant], ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
