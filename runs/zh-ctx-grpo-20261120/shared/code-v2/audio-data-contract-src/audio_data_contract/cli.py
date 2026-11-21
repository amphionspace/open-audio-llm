"""Command line validation and migration helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import (
    check_dangling,
    check_registration,
    find_dependents,
    load_catalog,
    resolve_artifact,
    resolve_split,
    verify_artifact_file,
)
from .overview import update_data_overview
from .records import load_records
from .roots import load_roots
from .state import DownloadState, inspect_download, load_state, write_state_atomic
from .views import load_view_catalog, resolve_view


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio-data-contract")
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("extract-audio", "localize-audio"):
        prepare = commands.add_parser(name)
        prepare.add_argument("--manifest", nargs="+", required=True)
        prepare.add_argument("--cache-dir", required=True)
        prepare.add_argument("--output-dir", required=True)
        prepare.add_argument("--workers", type=int, default=4)
        prepare.add_argument("--source-root")

    validate = commands.add_parser("validate-catalog")
    validate.add_argument("catalog")

    records = commands.add_parser("validate-records")
    records.add_argument("records")

    state = commands.add_parser("validate-state")
    state.add_argument("state_file")

    inspect = commands.add_parser("inspect-download")
    inspect.add_argument("path")
    inspect.add_argument("--expected-bytes", type=int)

    transition = commands.add_parser("transition-state")
    transition.add_argument("state_file")
    transition.add_argument("target", choices=[state.value for state in DownloadState])
    transition.add_argument("--error")

    resolve = commands.add_parser("resolve")
    resolve.add_argument("catalog")
    resolve.add_argument("dataset_id")
    resolve.add_argument("version")
    resolve.add_argument("artifact")
    resolve.add_argument("--roots")

    verify = commands.add_parser("verify-artifact")
    verify.add_argument("catalog")
    verify.add_argument("dataset_id")
    verify.add_argument("version")
    verify.add_argument("artifact")
    verify.add_argument("--roots")

    check = commands.add_parser("check-registration")
    check.add_argument("catalog")
    check.add_argument("dataset_id")
    check.add_argument("version")
    check.add_argument("--roots")
    check.add_argument("--sample", type=int, default=20)
    check.add_argument("--seed", type=int, default=0)
    check.add_argument(
        "--require-record-index",
        action="store_true",
        help="fail, instead of warn, when indexable splits have no record index",
    )

    holdout = commands.add_parser(
        "check-holdout",
        help="count records of a split built from recordings evaluation holds out",
    )
    holdout.add_argument("catalog")
    holdout.add_argument("dataset_id")
    holdout.add_argument("version")
    holdout.add_argument("split")
    holdout.add_argument("--roots")

    dangling = commands.add_parser("check-dangling")
    dangling.add_argument("catalog")
    dangling.add_argument("--roots", required=True)
    dangling.add_argument("--sample", type=int, default=20)
    dangling.add_argument("--seed", type=int, default=0)

    dependents = commands.add_parser("find-dependents")
    dependents.add_argument("catalog")
    dependents.add_argument("prefix", help='"root_alias:relative/path" of what is about to be removed')
    dependents.add_argument("--roots", help="also scan every audio-index row (slow)")

    index = commands.add_parser(
        "build-record-index", help="build a Lance record index; print its catalog ref"
    )
    index.add_argument("catalog")
    index.add_argument("dataset_id")
    index.add_argument("version")
    index.add_argument("destination")
    index.add_argument("--name", default="record_index")
    index.add_argument("--roots")
    index.add_argument(
        "--register",
        action="store_true",
        help="write the artifact and records_lance roles into the catalog YAML",
    )

    sync = commands.add_parser(
        "sync-tts-unified",
        help="regenerate tts_unified.yaml from DATA-TTS-UNIFIED/PUBLICATIONS.md",
    )
    sync.add_argument("data_root", help="directory holding DATA-TTS-UNIFIED")
    sync.add_argument("--catalog-dir", default="catalog")
    sync.add_argument("--views")
    sync.add_argument("--roots")
    sync.add_argument("--dry-run", action="store_true")
    sync.add_argument(
        "--notify", action="store_true", help="send Feishu cards on failure and recovery"
    )
    sync.add_argument("--state", help="file remembering the last failure notified")

    validate_views = commands.add_parser("validate-views")
    validate_views.add_argument("views")
    validate_views.add_argument("catalog")

    resolve_dataset_view = commands.add_parser("resolve-view")
    resolve_dataset_view.add_argument("views")
    resolve_dataset_view.add_argument("catalog")
    resolve_dataset_view.add_argument("view_id")
    resolve_dataset_view.add_argument("version")


    overview = commands.add_parser("generate-overview")
    overview.add_argument("--catalog", default="catalog")
    overview.add_argument("--views", default="views")
    overview.add_argument("--output", default="docs/datasets/data-overview.md")
    overview.add_argument("--check", action="store_true")

    duration = commands.add_parser("stats-duration")
    inputs = duration.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--catalog")
    inputs.add_argument("--manifest", nargs="+")
    inputs.add_argument("--audio-dir", nargs="+")
    duration.add_argument("--roots")
    duration.add_argument("--workers", type=int)
    duration.add_argument("--output")
    duration.add_argument("--write", action="store_true")

    audit = commands.add_parser("storage-audit")
    actions = audit.add_subparsers(dest="action", required=True)
    for name in ("scan", "report", "watermark"):
        action = actions.add_parser(name)
        action.add_argument("--zones", required=True)
        if name != "watermark":
            action.add_argument("--output-dir", required=True)
    actions.choices["scan"].add_argument("--workers", type=int, default=32)
    actions.choices["report"].add_argument("--snapshot")
    for name in ("report", "watermark"):
        actions.choices[name].add_argument("--notify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"extract-audio", "localize-audio"}:
        from .audio_prepare import run

        return run(args)
    if args.command == "stats-duration":
        from .duration import run

        return run(args)
    if args.command == "storage-audit":
        from .storage_audit import run

        return run(args)
    if args.command == "sync-tts-unified":
        from .tts_unified_sync import run

        return run(args)
    if args.command == "validate-catalog":
        catalog = load_catalog(args.catalog)
        print(json.dumps({"datasets": len(catalog)}, sort_keys=True))
        return 0
    if args.command == "validate-records":
        count = sum(1 for _ in load_records(args.records))
        print(json.dumps({"records": count}, sort_keys=True))
        return 0
    if args.command == "validate-state":
        state = load_state(args.state_file)
        print(
            json.dumps(
                {
                    "dataset_id": state.dataset_id,
                    "version": state.version,
                    "state": state.state.value,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "inspect-download":
        state = inspect_download(args.path, args.expected_bytes)
        print(json.dumps({"state": state.value}, sort_keys=True))
        return 0
    if args.command == "transition-state":
        state = load_state(args.state_file).transition(args.target, error=args.error)
        write_state_atomic(state, args.state_file)
        print(json.dumps({"state": state.state.value}, sort_keys=True))
        return 0
    if args.command == "resolve":
        catalog = load_catalog(args.catalog)
        roots = load_roots(args.roots)
        print(
            resolve_artifact(
                catalog, args.dataset_id, args.version, args.artifact, roots
            )
        )
        return 0
    if args.command == "verify-artifact":
        catalog = load_catalog(args.catalog)
        roots = load_roots(args.roots)
        spec = catalog.get(args.dataset_id, args.version)
        artifact = spec.artifact(args.artifact)
        path = resolve_artifact(
            catalog, args.dataset_id, args.version, args.artifact, roots
        )
        result = verify_artifact_file(artifact, path)
        result.update(
            {
                "artifact": artifact.name,
                "dataset_id": spec.dataset_id,
                "version": spec.version,
            }
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "check-registration":
        catalog = load_catalog(args.catalog)
        roots = load_roots(args.roots)
        spec = catalog.get(args.dataset_id, args.version)
        from .record_index import index_gaps

        problems = check_registration(spec, roots, sample=args.sample, seed=args.seed)
        warnings = index_gaps(spec)
        if args.require_record_index:
            problems, warnings = problems + warnings, []
        print(
            json.dumps(
                {
                    "dataset": spec.key,
                    "ok": not problems,
                    "problems": problems,
                    "warnings": warnings,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1 if problems else 0
    if args.command == "check-holdout":
        from .source_keys import holdout_overlap, load_holdouts, source_key_specs

        catalog = load_catalog(args.catalog)
        roots = load_roots(args.roots)
        spec = catalog.get(args.dataset_id, args.version)
        specs = source_key_specs(spec, args.split)
        if not specs:
            raise SystemExit(f"{spec.key}:{args.split} declares no provenance.source_keys")
        paths = resolve_split(
            catalog, spec.dataset_id, spec.version, args.split, "records", roots
        )
        if not paths:
            raise SystemExit(f"{spec.key}:{args.split} has no records artifact to read")
        holdouts = load_holdouts(catalog, roots)
        result = holdout_overlap(
            (record for path in paths for record in load_records(path)),
            specs,
            holdouts,
        )
        result.update(
            dataset=spec.key, split=args.split, holdouts=list(holdouts.sources)
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 1 if result["overlapping_records"] else 0
    if args.command == "check-dangling":
        broken = check_dangling(
            load_catalog(args.catalog),
            load_roots(args.roots),
            sample=args.sample,
            seed=args.seed,
        )
        print(json.dumps({"ok": not broken, "broken": broken}, ensure_ascii=False, sort_keys=True))
        return 1 if broken else 0
    if args.command == "find-dependents":
        roots = load_roots(args.roots) if args.roots else None
        found = find_dependents(load_catalog(args.catalog), args.prefix, roots)
        print(json.dumps({"prefix": args.prefix, "dependents": found}, ensure_ascii=False, sort_keys=True))
        return 1 if any(not item["retired"] for item in found) else 0
    if args.command == "build-record-index":
        from .record_index import (
            MANIFEST_NAME,
            build_index,
            catalog_ref,
            register_index,
        )

        catalog = load_catalog(args.catalog)
        roots = load_roots(args.roots)
        manifest = build_index(
            catalog, args.dataset_id, args.version, args.destination, roots
        )
        # Register under the alias the version's own artifacts use.
        alias = catalog.get(args.dataset_id, args.version).artifacts[0].root_alias
        ref = catalog_ref(
            Path(args.destination) / MANIFEST_NAME,
            name=args.name,
            roots=roots,
            root_alias=alias,
        )
        if args.register:
            path = register_index(
                args.catalog, args.dataset_id, args.version, ref, manifest["split_counts"]
            )
            ref = {**ref, "registered_in": str(path)}
        print(json.dumps(ref, sort_keys=True))
        return 0
    if args.command == "validate-views":
        datasets = load_catalog(args.catalog)
        views = load_view_catalog(args.views, datasets)
        print(json.dumps({"views": len(views)}, sort_keys=True))
        return 0
    if args.command == "resolve-view":
        datasets = load_catalog(args.catalog)
        views = load_view_catalog(args.views, datasets)
        view = views.get(args.view_id, args.version)
        result = resolve_view(views, datasets, args.view_id, args.version)
        print(
            json.dumps(
                {
                    "view_id": view.view_id,
                    "view_version": view.version,
                    "dataset_id": result.dataset_id,
                    "dataset_version": result.version,
                    "materialization": view.materialization,
                    "transforms": [step.to_dict() for step in view.transforms],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "generate-overview":
        status = update_data_overview(
            args.output,
            catalog_path=args.catalog,
            views_path=args.views,
            check=args.check,
        )
        print(json.dumps({"output": args.output, "status": status}, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
