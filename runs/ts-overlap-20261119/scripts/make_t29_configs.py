"""tasks/29: the two replay data configs, written from configs/train-data-encreplay.yaml (tasks 25/28, left unchanged).

- train-data-encreplay-t29.yaml: the same replay mix, each version moved along its catalog superseded_by chain to the
  newest final (training-text rule: follow superseded_by to the end). Final v4 (Arabic digits, adc #312) is not
  followed: its owner asked not to switch before the evaluators normalize numbers (amphiondata-58, 2026-11-27).
- train-data-asrreplay-t29.yaml: the second replay stream, only the plain-ASR sources of that mix (task asr, same
  weights and options) plus TALCS train, the code-switch corpus the 11.17 recipe never had and on which tasks 25/28
  drifted most (task 28 attempt 002: +0.155 of the +0.27 zh plain-ASR CER). TALCS has no final text, and its
  punctuated punc-written-v1-20261029 registers supervisions only (no Lhotse recordings mapping), so the source is the
  clean version clean-dual-qwen-moss-v1-20261009 with require_clean_pass (347,533 of 350,000 pass; source transcript,
  mostly unpunctuated: the KL only needs it as the teacher-forced context). Its weight is that of AISHELL-2 and
  KeSpeech, the mix's other zh read/phone corpora of similar size.
usage: python make_t29_configs.py   (prints every version it moved)"""
from pathlib import Path

import yaml

E = Path(__file__).resolve().parent.parent
CATALOG = Path("/workspace/data/registry/catalog")
TALCS = {"dataset_id": "talcs", "split": "train", "version": "clean-dual-qwen-moss-v1-20261009", "task": "asr",
         "weight": 1.9091, "replay_kl": False, "require_clean_pass": True}
HEADER = {"train-data-encreplay-t29.yaml": "# tasks/29: configs/train-data-encreplay.yaml with versions moved to their newest final (not final v4);\n"
                                           "# written by scripts/make_t29_configs.py.\n",
          "train-data-asrreplay-t29.yaml": "# tasks/29 second replay stream: the plain-ASR sources of train-data-encreplay-t29.yaml plus TALCS train\n"
                                           "# (clean-dual-qwen-moss-v1-20261009, require_clean_pass, weight as AISHELL-2 / KeSpeech); written by scripts/make_t29_configs.py.\n"}


def successors():
    nxt = {}
    for path in CATALOG.glob("*.yaml"):
        for d in yaml.safe_load(path.read_text()) or []:
            if not isinstance(d, dict):
                continue
            target = (d.get("provenance") or {}).get("superseded_by")
            if target and target != f"{d['dataset_id']}@{d['version']}" and "-final-v4-" not in target:
                nxt[(d["dataset_id"], d["version"])] = target.split("@", 1)[1]
    return nxt


def newest(nxt, dataset_id, version):
    seen = {version}
    while (dataset_id, version) in nxt:
        version = nxt[(dataset_id, version)]
        if version in seen:
            raise ValueError(f"superseded_by cycle at {dataset_id}@{version}")
        seen.add(version)
    return version


def main():
    cfg = yaml.safe_load((E / "configs/train-data-encreplay.yaml").read_text())
    nxt = successors()
    for s in cfg["train"]:
        v = newest(nxt, s["dataset_id"], s["version"])
        if v != s["version"]:
            print(f"{s['dataset_id']}:{s['split']} {s['version']} -> {v}")
            s["version"] = v
    asr = dict(cfg, train=[dict(s) for s in cfg["train"] if s.get("task") == "asr"] + [dict(TALCS)])
    for name, c in (("train-data-encreplay-t29.yaml", cfg), ("train-data-asrreplay-t29.yaml", asr)):
        (E / "configs" / name).write_text(HEADER[name] + yaml.safe_dump(c, sort_keys=False, allow_unicode=True, width=160))
        print(name, len(c["train"]), "sources")


main()
