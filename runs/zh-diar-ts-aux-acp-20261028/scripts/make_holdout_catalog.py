"""Local catalog holding only the 120 s windows of the 18 zh245 held-out training meetings (never trained on), for
tasks/05-train-fit: the same build as the trained windows, but unseen.

Copies the registry entries of aishell4 / alimeeting_sdm @ real-meetings-v2-20261010 (shared/catalog-holdout/), keeping
dataset_id, version and the split name train_120s so the records' audio refs still resolve; only the train_120s records
artifact is replaced by the held-out subset. Audio and the audio index stay in the registry roots.
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path

import yaml

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "shared/catalog-holdout"
REGISTRY = Path("/workspace/data/registry")
ALIAS = "train_fit_holdout"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    roots = json.loads((REGISTRY / "roots.json").read_text())
    entries = []
    for entry in yaml.safe_load((REGISTRY / "catalog/zh_meeting_v2.yaml").read_text()):
        if entry["dataset_id"] not in ("aishell4_meeting_sot", "alimeeting_sdm_meeting_sot"):
            continue
        entry = copy.deepcopy(entry)
        records = next(a for a in entry["artifacts"] if a["name"] == "train_120s-records")
        source = Path(roots[records["root_alias"]])
        holdout = {line.strip() for line in open(source / "zh245-holdout-records.txt") if line.strip()}
        target = OUT / records["relative_path"].replace("train_120s", "holdout_120s")
        kept = 0
        with gzip.open(source / records["relative_path"], "rt") as f, gzip.open(target, "wt") as g:
            for line in f:
                if json.loads(line)["id"] in holdout:
                    g.write(line)
                    kept += 1
        records.update(relative_path=target.name, root_alias=ALIAS,
                       sha256=hashlib.sha256(target.read_bytes()).hexdigest(), metadata={"record_count": kept})
        entry["splits"] = {"train_120s": {**entry["splits"]["train_120s"], "statistics": {"records": kept}}}
        entry["artifacts"] = [a for a in entry["artifacts"] if a["name"] in ("train-audio-index", "train_120s-records")]
        entries.append(entry)
        print(entry["dataset_id"], kept)
    (OUT / "catalog.yaml").write_text(yaml.safe_dump(entries, allow_unicode=True, sort_keys=False))
    # All registry aliases: the audio index points at other roots (e.g. aidc_data).
    (OUT / "roots.json").write_text(json.dumps({**roots, ALIAS: str(OUT)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
