"""Move context targets (build_context_targets.py output, keyed by the SFT config's versions) to the GRPO data
config's training-text final versions, checked against the final transcripts.

The final versions changed some transcripts (letters run together, English written back), so a target's character
offset can move. For each row: map the source key (final_versions.py), read the record's final transcript from the
GRPO data config's own sources (the explained subsets; the last entry of a key, as GRPOPromptPlan resolves it), and
keep each target whose text is at its offset; otherwise move it to the nearest occurrence of its text, or drop it.
Every kept row must score 1.0 with the reward itself (qwen3_asr.context on the transcript), the check the GRPO run
would otherwise fail on mid-training.
usage: finalize_targets.py GRPO_DATA_CONFIG IN_JSONL OUT_JSONL   (also writes OUT_JSONL's .summary.json)
"""
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from final_versions import CATALOG, final_key, final_version

FINAL_CATALOG = CATALOG / "text_final_v1.yaml"
ROOTS = json.loads((CATALOG.parent / "roots.json").read_text())


def catalog_entry(dataset_id, version):
    paths = [FINAL_CATALOG, CATALOG / f"{dataset_id}.yaml"]
    for path in [p for p in paths if p.exists()] + sorted(CATALOG.glob("*.yaml")):
        for d in yaml.safe_load(path.read_text()) or []:
            if (d["dataset_id"], d["version"]) == (dataset_id, version):
                return d
    raise SystemExit(f"{dataset_id}@{version} is not registered")


def artifact_path(artifact):
    return Path(ROOTS[artifact.get("root_alias") or "aidc_data"]) / artifact["relative_path"]


def transcripts(config, wanted):
    """{source key: {record id: transcript}} for the wanted (source key, record id)s, read from the registered
    artifacts the loader reads: the split's Lance record index (column text) or its audio-records JSONL (target).
    Lance rows of records built from Lhotse cuts have the bare cut id; the loader's id is ``<key>:<cut id>``."""
    import lance

    keys = {f"{s['dataset_id']}@{s['version']}:{s['split']}" for s in config["train"]}
    missing = {k for k, _ in wanted} - keys
    if missing:
        raise SystemExit(f"target sources not in the data config: {sorted(missing)}")
    by_key = {}
    for k, record_id in wanted:
        by_key.setdefault(k, set()).add(record_id)
    out = {}
    for k, ids in by_key.items():
        dataset_version, split = k.rsplit(":", 1)
        dataset_id, version = dataset_version.split("@", 1)
        entry = catalog_entry(dataset_id, version)
        artifacts = {a["name"]: a for a in entry["artifacts"]}
        spec = entry["splits"][split]["artifacts"]
        texts = {}
        if spec.get("records_lance"):
            table = lance.dataset(str(artifact_path(artifacts[spec["records_lance"][0]])).replace("index.json", "table.lance"))
            bare = {i[len(k) + 1:] if i.startswith(k + ":") else i: i for i in ids}
            quoted = ",".join("'" + i.replace("'", "''") + "'" for i in bare)
            for row in table.to_table(columns=["id", "text"], filter=f"id IN ({quoted})").to_pylist():
                texts[bare[row["id"]]] = row["text"]
        else:
            for name in spec["records"]:
                with gzip.open(artifact_path(artifacts[name]), "rt", encoding="utf-8") as stream:
                    for line in stream:
                        record = json.loads(line)
                        if record["id"] in ids:
                            texts[record["id"]] = record["target"]
        out[k] = texts
    return out


def place(text, target):
    """The target at its offset, or moved to the nearest occurrence of its text, or None."""
    if text[target["start"]:target["start"] + len(target["text"])] == target["text"]:
        return target, "kept"
    hits, at = [], text.find(target["text"])
    while at >= 0:
        hits.append(at)
        at = text.find(target["text"], at + 1)
    if not hits:
        return None, "dropped"
    return {**target, "start": min(hits, key=lambda h: abs(h - target["start"]))}, "moved"


def main():
    from open_audio_llm.integrations.ms_swift.rewards import qwen3_asr

    config = yaml.safe_load(Path(sys.argv[1]).read_text())
    rows = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8")]
    final = final_version()
    for row in rows:
        old = row["source"].rsplit(":", 1)[0]
        row["source"] = final_key(final, row["source"])
        # Records built from Lhotse cuts carry their version in the id (wenetspeech_clean); the final version's
        # records do too, under the new version.
        if row["record_id"].startswith(old + ":"):
            row["record_id"] = row["source"].rsplit(":", 1)[0] + row["record_id"][len(old):]
    texts = transcripts(config, {(row["source"], row["record_id"]) for row in rows})
    out, stats = [], Counter()
    for row in rows:
        text = texts[row["source"]].get(row["record_id"])
        if text is None:
            stats["row dropped: record not in the source"] += 1
            continue
        targets = []
        for target in row["targets"]:
            placed, how = place(text, target)
            stats[f"target {how}"] += 1
            if placed is not None:
                targets.append(placed)
        targets = list({(t["start"], t["text"]): t for t in targets}.values())
        if not targets:
            stats["row dropped: no target left"] += 1
            continue
        assert qwen3_asr.context(text, text, targets) == 1.0, row
        out.append({**row, "targets": targets})
    Path(sys.argv[3]).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    summary = {"data_config": sys.argv[1], "input": sys.argv[2], "records": len(out),
               "by_source_kind": {f"{s} {k}": n for (s, k), n in sorted(Counter((r["source"], r["kind"]) for r in out).items())},
               "targets": sum(len(r["targets"]) for r in out),
               "targets_with_alt": sum(t["alt"] is not None for r in out for t in r["targets"]),
               "placement": dict(stats)}
    Path(f"{sys.argv[3]}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
