"""Join explicit enrollment provenance to SOT records, or freeze an evaluation view.

This command does not infer global identities, modify source audio or launch a
training run. Identity/provenance annotations belong to upstream data preparation.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from audio_data_contract import load_records, write_records

from .target_sot import EnrollmentConfig, eligible_candidates, sample_enrollment


def attach_annotations(record, annotation):
    fields = ("partition", "speaker_identities", "source_spans", "enrollment_candidates")
    updated = replace(record, metadata={**record.metadata, **{k: annotation[k] for k in fields}})
    eligible_candidates(updated, EnrollmentConfig())
    return updated


def freeze_enrollment(record, seed, *, seconds=3.0, mode="all"):
    """Freeze a common 5-second-capable reference pool for all duration panels."""
    if not 1 <= seconds <= 5:
        raise ValueError("Fixed evaluation duration must be in [1, 5]")
    config = EnrollmentConfig(min_seconds=5, max_seconds=5, probability=1, mode=mode)
    view = sample_enrollment(record, config, f"{seed}:{record.id}")
    if "enrollment_view" not in view.metadata:
        raise ValueError(f"No eligible 5-second reference for common duration panel: {record.id}")
    frozen = view.metadata["enrollment_view"]
    frozen = {"mode": mode, "enrollments": [
        {"speaker_id": r["speaker_id"], "ref": {**r["ref"], "duration": seconds}}
        for r in frozen["enrollments"]
    ]}
    return replace(record, metadata={**record.metadata, "fixed_enrollment": frozen})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--annotations", type=Path,
                        help="JSONL keyed by id; explicit partition, identities, source spans and candidates")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-seconds", type=float, choices=[1, 2, 3, 5])
    parser.add_argument("--mode", choices=["all", "targets_only"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.records.resolve():
        parser.error("output must be a new file")
    annotations = {}
    if args.annotations:
        for line in args.annotations.read_text().splitlines():
            item = json.loads(line)
            if item["id"] in annotations:
                raise ValueError(f"Duplicate annotation: {item['id']}")
            annotations[item["id"]] = item

    def records():
        for record in load_records(args.records):
            if args.annotations:
                record = attach_annotations(record, annotations.pop(record.id))
            if args.freeze_seconds is not None:
                record = freeze_enrollment(record, args.seed, seconds=args.freeze_seconds, mode=args.mode)
            yield record
        if annotations:
            raise ValueError("Annotations contain IDs absent from the record manifest")

    # A failed join cannot be mistaken for a completed dataset.
    temporary = args.output.with_name(args.output.stem + ".pending" + args.output.suffix)
    try:
        write_records(records(), temporary)
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
