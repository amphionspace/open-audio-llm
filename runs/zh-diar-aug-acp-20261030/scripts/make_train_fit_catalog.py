"""Local catalog of the fixed trained-window set for the selector's train_fit panel (shared/catalog-train-fit/).

The set is the AliMeeting / AISHELL-4 real-meetings-v2 train_120s windows that round 1's fit check sampled (seed 42,
150 per source; zh-diar-ts-aux-acp-20261028/tasks/05-train-fit/start) minus the windows training excludes (zh245
held-out meetings and empty targets): 143 AliMeeting + 126 AISHELL-4 windows that both arms train on. The registry
entries are copied with dataset_id, version and split name kept so the audio refs resolve; only the train_120s records
artifact is replaced by the subset.
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path

import yaml

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "shared/catalog-train-fit"
REGISTRY = Path("/workspace/data/registry")
SAMPLE = RUN.parent / ("zh-diar-ts-aux-acp-20261028/tasks/05-train-fit/start/attempts/001/artifacts/evaluation/"
                       "meeting-scores.jsonl")
ALIAS = "train_fit_set"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    roots = json.loads((REGISTRY / "roots.json").read_text())
    sampled = {json.loads(line)["id"] for line in open(SAMPLE)}
    entries, kept_ids = [], []
    for entry in yaml.safe_load((REGISTRY / "catalog/zh_meeting_v2.yaml").read_text()):
        if entry["dataset_id"] not in ("aishell4_meeting_sot", "alimeeting_sdm_meeting_sot"):
            continue
        entry = copy.deepcopy(entry)
        records = next(a for a in entry["artifacts"] if a["name"] == "train_120s-records")
        source = Path(roots[records["root_alias"]])
        excluded = {line.strip() for line in open(source / "zh-v2-train-excluded-records.txt") if line.strip()}
        target = OUT / records["relative_path"].replace("train_120s", "train-fit_120s")
        kept = 0
        with gzip.open(source / records["relative_path"], "rt") as f, gzip.open(target, "wt") as g:
            for line in f:
                rid = json.loads(line)["id"]
                if rid in sampled and rid not in excluded:
                    g.write(line)
                    kept += 1
                    kept_ids.append(rid)
        records.update(relative_path=target.name, root_alias=ALIAS,
                       sha256=hashlib.sha256(target.read_bytes()).hexdigest(), metadata={"record_count": kept})
        entry["splits"] = {"train_120s": {**entry["splits"]["train_120s"], "statistics": {"records": kept}}}
        entry["artifacts"] = [a for a in entry["artifacts"] if a["name"] in ("train-audio-index", "train_120s-records")]
        entries.append(entry)
        print(entry["dataset_id"], kept)
    (OUT / "catalog.yaml").write_text(yaml.safe_dump(entries, allow_unicode=True, sort_keys=False))
    # All registry aliases: the audio index points at other roots (e.g. aidc_data).
    (OUT / "roots.json").write_text(json.dumps({**roots, ALIAS: str(OUT)}, indent=2) + "\n")
    (OUT / "ids.txt").write_text("\n".join(sorted(kept_ids)) + "\n")


if __name__ == "__main__":
    main()
