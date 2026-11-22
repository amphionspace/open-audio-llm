"""think-v2-20261113 on the training-text final versions: context_correction_clean_zh@think-v2-20261113-final-v1-20261120.

A think row names its training record as (dataset@version, record_id), and think-v2 names the ctxclean and other
intermediate versions; once a recipe trains on the final versions the loader rejects the file (its datasets are read
by no source). The final versions only change text, never add or drop records (text-final owner amphiondata-64,
2026-11-22), so each row keeps its record and its think:
- the dataset key goes to the final version, one hop through ``provenance.superseded_by``, or the version plus
  ``-final-v1-20261120`` (final_versions.py's rule; every ``text_final_v1*.yaml`` list is read);
- record IDs that carry the version (records built from Lhotse cuts, e.g. wenetspeech_clean) get the new prefix;
  ``split:id`` IDs stay;
- the include lists (think-on record IDs per split, for include_records) are rewritten the same way.
Every row is checked against the final records: the record exists, and for think-on rows every 「quote」 of the think
is still in the final transcript. Records whose transcript changed are counted by the final round's two kinds
(letters joined, e.g. K T V -> KTV; WenetSpeech's English written back, ``custom.zhen_fix.original_text``).
usage: build_think_v2_final.py   (writes into the version directory, which holds the placeholder state.json)
"""
import gzip
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

CATALOG = Path("/workspace/data/registry/catalog")
DATA = Path("/workspace/data")
DATASET = "context_correction_clean_zh"
SOURCE_VERSION = "think-v2-20261113"
SUFFIX = "-final-v1-20261120"
VERSION = SOURCE_VERSION + SUFFIX
SOURCE = DATA / f"datasets/{DATASET}/versions/{SOURCE_VERSION}"
OUT = DATA / f"datasets/{DATASET}/versions/{VERSION}"
QUOTE = re.compile(r"「([^」]+)」")
ID = re.compile(r'"id": "([^"]+)"')


def entries():
    for path in sorted(CATALOG.glob("*.yaml")):
        yield from yaml.safe_load(path.read_text()) or []


def final_map():
    rows = list(entries())
    finals = {(d["dataset_id"], d["version"]) for path in CATALOG.glob("text_final_v1*.yaml")
              for d in yaml.safe_load(path.read_text())}
    superseded = {(d["dataset_id"], d["version"]): (d.get("provenance") or {}).get("superseded_by") for d in rows}
    specs = {(d["dataset_id"], d["version"]): d for d in rows}

    def final(key):
        dataset_id, version = key.split("@", 1)
        if (dataset_id, version + SUFFIX) in finals:
            return f"{dataset_id}@{version}{SUFFIX}"
        target = superseded.get((dataset_id, version))
        if target and tuple(target.split("@", 1)) in finals:
            return target
        raise SystemExit(f"{key}: no training-text final version")

    return final, specs


def artifact_path(spec, split, kind):
    names = set(spec["splits"][split]["artifacts"][kind])
    found = [a for a in spec["artifacts"] if a["name"] in names]
    assert len(found) == 1, (spec["dataset_id"], spec["version"], split, kind, found)
    return DATA / found[0]["relative_path"]


def lookup(spec, split, ids):
    """{record id: (transcript, custom)} for ``ids`` of one split. Record sources list records; Lhotse sources list
    supervisions, whose record ID is ``dataset@version:split:<supervision id>`` (catalog_cache)."""
    if "records" in spec["splits"][split]["artifacts"]:
        path, prefix = artifact_path(spec, split, "records"), ""
    else:
        path, prefix = artifact_path(spec, split, "supervisions"), f"{spec['dataset_id']}@{spec['version']}:{split}:"
    wanted = {i[len(prefix):] for i in ids if i.startswith(prefix)}
    assert len(wanted) == len(ids), (spec["version"], split, "IDs do not carry the source key")
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if any(i in wanted for i in ID.findall(line)):
                row = json.loads(line)
                if row["id"] in wanted:
                    out[prefix + row["id"]] = (row.get("target", row.get("text")), row.get("custom") or {})
    return out


def split_of(record_id, old_key):
    rest = record_id[len(old_key) + 1:] if record_id.startswith(old_key + ":") else record_id
    return rest.split(":", 1)[0]


def renamed(record_id, old_key, new_key):
    return new_key + record_id[len(old_key):] if record_id.startswith(old_key + ":") else record_id


def main():
    state = json.loads((OUT / "state.json").read_text())
    assert state["state"] == "planned", state["state"]
    final, specs = final_map()
    summary = json.loads((SOURCE / "summary.json").read_text())
    checks, sources = {}, {}
    for sub in sorted(p for p in SOURCE.iterdir() if p.is_dir()):
        with gzip.open(sub / "think.jsonl.gz", "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        keys = sorted({r["dataset"] for r in rows})
        assert len(keys) == 1, (sub.name, keys)
        old_key, new_key = keys[0], final(keys[0])
        old_spec, new_spec = specs[tuple(old_key.split("@", 1))], specs[tuple(new_key.split("@", 1))]
        count = Counter(rows=len(rows), think_on=sum(r["think_on"] for r in rows))
        by_split = {}
        for r in rows:
            by_split.setdefault(split_of(r["record_id"], old_key), []).append(r)
        out_rows = []
        for split, group in sorted(by_split.items()):
            old_ids = {r["record_id"] for r in group}
            new_ids = {renamed(i, old_key, new_key) for i in old_ids}
            old = lookup(old_spec, split, old_ids)
            new = lookup(new_spec, split, new_ids)
            for r in group:
                rid = renamed(r["record_id"], old_key, new_key)
                if rid not in new:
                    count["missing in final"] += 1
                    continue
                before, (after, custom) = old.get(r["record_id"], (None, {}))[0], new[rid]
                if before is None:
                    count["missing in source version"] += 1
                elif before != after:
                    fix = custom.get("zhen_fix") or {}
                    count["text changed, English written back" if "original_text" in fix
                          else "text changed, other (letters joined)"] += 1
                if r["think_on"]:
                    lost = [q for q in QUOTE.findall(r["think"]) if q not in after]
                    if lost:
                        count["think-on, quote not in final text"] += 1
                        count.setdefault("examples", [])
                        if len(count["examples"]) < 5:
                            count["examples"].append({"record_id": rid, "lost": lost})
                out_rows.append({**r, "dataset": new_key, "record_id": rid})
        (OUT / sub.name).mkdir(exist_ok=True)
        with gzip.open(OUT / sub.name / "think.jsonl.gz", "wt", encoding="utf-8") as stream:
            for r in out_rows:
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")
        includes = {}
        for name, item in json.loads((sub / "includes.json").read_text()).items():
            key, split = name.split("|")
            assert key == old_key, (sub.name, name)
            ids = Path(item["path"]).read_text().split()
            path = OUT / sub.name / f"include-{new_key.replace('@', '_')}-{split}.txt"
            path.write_text("".join(renamed(i, old_key, new_key) + "\n" for i in ids))
            includes[f"{new_key}|{split}"] = {"path": str(path), "records": len(ids)}
        (OUT / sub.name / "includes.json").write_text(json.dumps(includes, ensure_ascii=False, indent=2) + "\n")
        checks[sub.name] = {"from": old_key, "to": new_key, **dict(count)}
        sources[sub.name] = {"think_on_records": count["think_on"], "listed_records": len(out_rows),
                             "datasets": [new_key]}
        print(sub.name, json.dumps(checks[sub.name], ensure_ascii=False), flush=True)
    (OUT / "summary.json").write_text(json.dumps({"source": summary, "final_checks": checks}, ensure_ascii=False,
                                                 indent=1) + "\n")
    problems = sum(c.get("missing in final", 0) for c in checks.values())
    state.update(state="prepared" if not problems else "failed",
                 updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 error=None if not problems else f"{problems} rows have no record in the final version")
    state["metadata"].update(stage="done" if not problems else "failed", sources=sources,
                             think_on_records=sum(s["think_on_records"] for s in sources.values()),
                             listed_records=sum(s["listed_records"] for s in sources.values()))
    (OUT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
