"""Context-correction targets for GRPO (``grpo.context_targets``, reward ``qwen3_asr_context``) from an SFT data config.

One JSONL row per record that has a target: ``{source, record_id, kind, targets}``, a target being
``{start, text, alt}`` (``text`` at character ``start`` of the record's transcript, ``alt`` the spelling the context
rules out or None). Records come from the data config's own sources, so the keys match the GRPO run that replays it:
- synthetic spelling data (context_correction_synth_zh): the target segments of ``labels.context_correction``, both
  kinds: ``correct`` (the context points away from the common spelling) and ``confirm`` (it points to the common
  one; changing it is the over-correction the reward's -1 punishes);
- real recordings with a spoken explanation: the think files' think-on rows, joined by unit id with the transcript
  cleaning's decisions (context_correction_clean_zh@v1-20261110/<clean source>/decisions.jsonl). A decision covers a
  unit of one or more segments, one record each; the record's segment is found by id. Its edits give each target
  exactly (``offset`` in the segment's original text, ``from`` -> ``to``); a segment without an edit (the transcript
  was already right) has as targets the explained characters (from the think lines) where they occur outside the
  explanation's own words, with no known alternative.
The segment's corrected text is taken to be the record's transcript (as the cleaned versions were built); the reward
raises on a target that does not match, so a mismatch shows up when the mining plans are scored, before training.
usage: build_context_targets.py DATA_CONFIG OUT_JSONL   (writes OUT_JSONL and OUT_JSONL's .summary.json)
"""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

DATASETS = Path("/workspace/data/datasets")
SYNTH = "context_correction_synth_zh"
DECISIONS = DATASETS / "context_correction_clean_zh/versions/v1-20261110"
THINK_LINE = re.compile(r"说话人说是「(.+?)」，写作「(.+?)」")
# The synthetic splits the ctx plan mines (make_configs.py's ctx lanes). Other synthetic splits of a config are replay
# only, e.g. ts-twins-v1-20261121's train_ts_twins, whose twin records have no context_correction labels.
SYNTH_SPLITS = ("train", "train_ts")


def synthetic(source):
    path = DATASETS / SYNTH / "versions" / source["version"] / "records" / f"{source['split']}.jsonl.gz"
    for line in gzip.open(path, "rt", encoding="utf-8"):
        record = json.loads(line)
        cc = record["labels"]["context_correction"]
        text, targets = "", []
        for segment in cc["segments"]:
            if segment["role"] == "target":
                targets.append({"start": len(text), "text": segment["text"], "alt": segment.get("alt")})
            text += segment["text"]
        assert text == record["target"], record["id"]
        yield record["id"], cc["kind"], targets


def split_of(record_id, splits):
    """The config split a record belongs to, from its id (``<split>:<id>`` or ``<dataset>@<version>:<split>:<id>``)."""
    found = {s for s in splits if record_id.startswith(f"{s}:") or f":{s}:" in record_id}  # a split may repeat
    return found.pop() if len(found) == 1 else None


def explained(text, thinks):
    """Explained characters where they occur outside the explanations' own words."""
    lines = [m.groups() for m in THINK_LINE.finditer(thinks)]
    covered = set()
    for evidence, _ in lines:
        for m in re.finditer(re.escape(evidence), text):
            covered.update(range(m.start(), m.end()))
    targets = []
    for _, chars in lines:
        for m in re.finditer(re.escape(chars), text):
            if not covered & set(range(m.start(), m.end())):
                targets.append({"start": m.start(), "text": chars, "alt": None})
    return {(t["start"], t["text"]): t for t in targets}.values()


def segment_of(decision, record_id):
    """The decision segment that is this record: the only one, or the one whose id ends the record id."""
    segments = decision.get("segments", [])
    if len(segments) == 1:
        return segments[0]
    found = [s for s in segments if record_id.endswith(s["id"].split(":")[-1])]
    return found[0] if len(found) == 1 else None


def corrected(segment, edits):
    """The segment's corrected text and its targets, edits applied left to right."""
    text, targets, shift = segment["text"], [], 0
    for e in sorted(edits, key=lambda e: e["offset"]):
        at = e["offset"] + shift
        assert text[at:at + len(e["from"])] == e["from"], (segment["id"], e)
        text = text[:at] + e["to"] + text[at + len(e["from"]):]
        targets.append({"start": at, "text": e["to"], "alt": e["from"]})
        shift += len(e["to"]) - len(e["from"])
    return text, targets


def real(think_file, sources, stats):
    clean = Path(think_file).parent.name
    decisions = {}
    for line in open(DECISIONS / clean / "decisions.jsonl", encoding="utf-8"):
        row = json.loads(line)
        for key in [row.get("unit_id")] + [s["id"] for s in row.get("segments", [])]:
            if key:
                decisions[key] = row
    for line in gzip.open(think_file, "rt", encoding="utf-8"):
        row = json.loads(line)
        if not row["think_on"] or row["think"] is None:
            continue
        dataset_id, version = row["dataset"].split("@", 1)
        split = split_of(row["record_id"], sources.get((dataset_id, version), []))
        decision = decisions.get(row["unit_id"])
        if split is None:
            stats[f"{clean}: no single config split"] += 1
            continue
        segment = segment_of(decision, row["record_id"]) if decision else None
        if segment is None:
            stats[f"{clean}: no decision segment for the record"] += 1
            continue
        edits = [e for e in decision.get("edits") or [] if e.get("segment", segment["id"]) == segment["id"]]
        text, targets = corrected(segment, edits)
        kind = "corrected"
        if not targets:
            targets, kind = list(explained(text, row["think"])), "explained"
        if not targets:
            stats[f"{clean}: no target outside the explanation"] += 1
            continue
        yield f"{dataset_id}@{version}:{split}", row["record_id"], kind, targets


def main():
    config, out = yaml.safe_load(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
    sources = defaultdict(list)
    for s in config["train"]:
        sources[(s["dataset_id"], s["version"])].append(s["split"])
    rows, stats = [], Counter()
    for s in config["train"]:
        if s["dataset_id"] == SYNTH and s["split"] in SYNTH_SPLITS:
            key = f"{s['dataset_id']}@{s['version']}:{s['split']}"
            rows += [{"source": key, "record_id": rid, "kind": kind, "targets": t} for rid, kind, t in synthetic(s)]
    for think_file in config["think"]["files"]:
        if f"/{SYNTH}/" not in think_file:
            rows += [{"source": key, "record_id": rid, "kind": kind, "targets": t}
                     for key, rid, kind, t in real(think_file, sources, stats)]
    seen = Counter((r["source"], r["record_id"]) for r in rows)
    assert max(seen.values()) == 1, [k for k, v in seen.items() if v > 1][:3]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    summary = {"data_config": sys.argv[1], "records": len(rows),
               "by_source_kind": {f"{s} {k}": n for (s, k), n in sorted(Counter((r["source"], r["kind"]) for r in rows).items())},
               "targets": sum(len(r["targets"]) for r in rows),
               "targets_with_alt": sum(t["alt"] is not None for r in rows for t in r["targets"]),
               "skipped": dict(stats)}
    Path(f"{out}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
