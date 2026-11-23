"""Training-text final versions (audio-data-contract catalog/text_final_v1.yaml, 2026-11-20) for config versions.

A version maps to ``<version>-final-v1-20261120`` when that is registered, else to its ``superseded_by`` (the
cleaned intermediate versions, e.g. ``-ctxclean-20261110``, name their final there), else stays. The text-final
owner (amphiondata-64) confirmed both routes on 2026-11-21: some original versions have a final without being
marked superseded.
"""
from pathlib import Path

import yaml

CATALOG = Path("/workspace/data/registry/catalog")
SUFFIX = "-final-v1-20261120"
# Final v2 (2026-11-23, catalog/text_final_v2.yaml): the zh-en spacing rule and three new finals; named <base>-final-v2-20261123.
SUFFIXES = ("-final-v2-20261123", SUFFIX)


def final_version():
    # text_final_v1.yaml (zh/en, 2026-11-20) and later per-language lists such as text_final_v1_es.yaml (2026-11-21).
    finals = {(d["dataset_id"], d["version"]) for path in sorted(CATALOG.glob("text_final_v*.yaml"))
              for d in yaml.safe_load(path.read_text())}
    superseded = {}
    for path in CATALOG.glob("*.yaml"):
        for d in yaml.safe_load(path.read_text()) or []:
            target = (d.get("provenance") or {}).get("superseded_by")
            if target:
                superseded[(d["dataset_id"], d["version"])] = target

    def final(dataset_id, version):
        # A v1 final upgrades to the v2 of the same base when one is registered; v2 is preferred over v1.
        base = version[: -len(SUFFIX)] if version.endswith(SUFFIX) else version
        if version.endswith(SUFFIXES[0]):
            return version
        for suffix in SUFFIXES:
            if (dataset_id, base + suffix) in finals:
                return base + suffix
        target = superseded.get((dataset_id, version))
        if target:
            name, new = target.split("@", 1)
            if name != dataset_id or (name, new) not in finals:
                raise ValueError(f"{dataset_id}@{version} superseded by {target}, which is not a text-final version")
            return final(dataset_id, new)  # a v1 final named by superseded_by upgrades to its v2
        return version

    return final


def final_key(final, key):
    """``dataset@version:split`` with the version mapped."""
    dataset_version, split = key.rsplit(":", 1)
    dataset_id, version = dataset_version.split("@", 1)
    return f"{dataset_id}@{final(dataset_id, version)}:{split}"
