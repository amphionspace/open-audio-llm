#!/usr/bin/env python3
"""Leave out of a training data config the records whose text is a public test sentence.

Several read-speech corpora reuse their prompt sentences across splits: AISHELL-2 train holds
96.5% of the AISHELL-1 test sentences, MagicData train (inside hw_sft) 56% of MagicData test.
A model trained on them reads those test sets from memory (docs/research/
2026-11-23-public-asr-leaderboards.md). This script lists, for every train source of a data
config, the records whose normalized target equals a reference of a public test set, and
writes a new config whose sources exclude them.

Training side: the records as the loader sees them, from each version's Lance record index
(record_index: lance), or through the loader's own resolver when a version has none yet.
Lhotse-built sources are listed with their prefixed ID
(<dataset>@<version>:<split>:<id>), record-built ones with the record ID; both forms are what
the loader matches (catalog_dataset and lance_index._Allowed). An existing exclude_records
list of a source is kept and merged.

Test side: the references of evaluation runs (samples.jsonl of amphion-eval runs), so every
eval set already run counts without resolving its audio. Sources whose name starts with one of
--internal-prefix (default police_) are internal sets and are not used.

Matching: NFKC, lower case, every non-word character removed. A short reference is a stock
phrase both sides share, not a leak: GigaSpeech train repeats "thank you." 10,109 times. A
reference written with CJK characters counts from --min-chars characters (default 8), any other
from --min-words words (default 6).

    python scripts/decontaminate_training_data.py IN.yaml OUT.yaml LISTS_DIR \
        --samples run1/samples.jsonl --samples run2/samples.jsonl ... --report report.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

DATASETS = Path("/workspace/data/datasets")
MIN_CHARS = 8
MIN_WORDS = 6
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")
_NON_WORD = re.compile(r"[\W_]+")


def norm(text):
    return _NON_WORD.sub("", unicodedata.normalize("NFKC", text or "").lower())


def long_enough(reference, min_chars, min_words):
    if _CJK.search(reference):
        return len(norm(reference)) >= min_chars
    return len(reference.split()) >= min_words


def test_references(paths, internal_prefixes, min_chars, min_words):
    """{normalized reference: set of test sources} from evaluation samples."""
    found = collections.defaultdict(set)
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                source = row.get("source") or ""
                name = source.split("@", 1)[0]
                if any(name.startswith(p) for p in internal_prefixes):
                    continue
                reference = row.get("reference") or ""
                if long_enough(reference, min_chars, min_words):
                    found[norm(reference)].add(source)
    return found


def lance_rows(dataset_id, version, split):
    """(record ID as the loader matches it, target text) of one source, from its Lance index."""
    import lance

    index = DATASETS / dataset_id / "versions" / version / "record_index"
    info = json.loads((index / "index.json").read_text())
    lhotse = any(s["kind"].startswith("lhotse") for s in info.get("sources", []))
    table = lance.dataset(str(index / info.get("table_path", "table.lance")))
    prefix = f"{dataset_id}@{version}:{split}:" if lhotse else ""
    for batch in table.to_batches(columns=["id", "splits", "text"]):
        for ident, splits, text in zip(*(batch.column(c).to_pylist() for c in ("id", "splits", "text"))):
            if split in splits:
                yield prefix + ident, text


def catalog_rows(catalog, roots, dataset_id, version, split):
    """The same rows through the loader's JSONL path, for a version without a Lance index.

    Record-built splits: the AudioRecord ID and target. Lhotse-built splits:
    <dataset>@<version>:<split>:<supervision ID> and the supervision text, as
    CatalogSwiftDataset._cut_records names them.
    """
    from audio_data_contract import load_records, resolve_split

    from open_audio_llm.data.catalog_resolver import LhotseCatalogAudioResolver

    resolver = LhotseCatalogAudioResolver(catalog, roots)
    paths = resolve_split(resolver.catalog, dataset_id, version, split, "records", resolver.roots)
    if paths:
        for path in paths:
            for record in load_records(path):
                yield record.id, record.target
        return
    for cut in resolver.iter_cuts(dataset_id, version, split):
        for supervision in cut.supervisions:
            yield f"{dataset_id}@{version}:{split}:{supervision.id}", supervision.text


def scan(job):
    """IDs of a source's records whose target is a test reference, with the matched test sources."""
    (dataset_id, version, split), references, catalog, roots = job
    hits, total = {}, 0
    index = DATASETS / dataset_id / "versions" / version / "record_index" / "index.json"
    rows = (lance_rows(dataset_id, version, split) if index.is_file()
            else catalog_rows(catalog, roots, dataset_id, version, split))
    try:
        for ident, text in rows:
            total += 1
            key = norm(text)
            if key in references:
                hits[ident] = sorted(references[key])
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return (dataset_id, version, split), None, 0, f"{type(exc).__name__}: {exc}"
    return (dataset_id, version, split), hits, total, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("config")
    parser.add_argument("output")
    parser.add_argument("lists", help="directory for the merged exclude lists")
    parser.add_argument("--samples", action="append", required=True, help="samples.jsonl of an evaluation run")
    parser.add_argument("--internal-prefix", action="append", default=None)
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS)
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    internal = tuple(args.internal_prefix or ["police_"])

    config_path = Path(args.config)
    if Path(args.output).resolve().parent != config_path.resolve().parent:
        # Paths in a data config (metadata_cache, exclude lists) are relative to its directory.
        sys.exit("write the output config next to the input config")
    references = test_references(args.samples, internal, args.min_chars, args.min_words)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    keys = sorted({(s["dataset_id"], s["version"], s["split"]) for s in config["train"]})
    catalog, roots = (str((config_path.parent / config[k]).resolve()) for k in ("catalog", "roots"))
    with ProcessPoolExecutor(args.workers) as pool:
        results = {key: (hits, total, error) for key, hits, total, error in
                   pool.map(scan, [(key, references, catalog, roots) for key in keys], chunksize=1)}
    errors = {f"{k[0]}@{k[1]}:{k[2]}": e for k, (_, _, e) in results.items() if e}
    if errors:
        sys.exit("cannot read sources:\n" + "\n".join(f"  {k}: {v}" for k, v in errors.items()))

    lists = Path(args.lists)
    lists.mkdir(parents=True, exist_ok=True)
    report = {"min_chars": args.min_chars, "min_words": args.min_words, "test_references": len(references),
              "samples": args.samples, "internal_prefixes": list(internal), "sources": {}}
    by_test = collections.Counter()
    for source in config["train"]:
        key = (source["dataset_id"], source["version"], source["split"])
        hits, total, _ = results[key]
        name = f"{key[0]}@{key[1]}:{key[2]}"
        if not hits:
            continue
        existing = set()
        if source.get("exclude_records"):
            path = Path(source["exclude_records"])
            path = path if path.is_absolute() else config_path.parent / path
            existing = set(path.read_text().split())
        merged = lists / f"{name.replace('/', '_')}.txt"
        merged.write_text("".join(f"{i}\n" for i in sorted(existing | set(hits))))
        source["exclude_records"] = str(merged.resolve())
        entry = report["sources"].setdefault(name, {"records": total, "excluded": len(hits),
                                                     "kept_from_existing_list": len(existing)})
        tests = collections.Counter(t.split(":", 1)[0] + ":" + t.split(":", 1)[1].split("/", 1)[0]
                                    for matched in hits.values() for t in matched)
        entry["by_test_set"] = dict(tests.most_common())
        by_test.update(tests)
    report["by_test_set"] = dict(by_test.most_common())
    report["excluded_total"] = sum(v["excluded"] for v in report["sources"].values())

    header = (f"# {config_path.name} with records whose text is a public test sentence excluded\n"
              f"# (scripts/decontaminate_training_data.py; report {Path(args.report).name}).\n")
    Path(args.output).write_text(header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False, width=1000),
                                 encoding="utf-8")
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"excluded_total": report["excluded_total"],
                      "sources_with_hits": len(report["sources"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
