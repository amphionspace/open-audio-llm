"""Command line validation and migration helpers."""

from __future__ import annotations

import argparse
import json

from .catalog import load_catalog, resolve_artifact, verify_artifact_file
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"extract-audio", "localize-audio"}:
        from .audio_prepare import run

        return run(args)
    if args.command == "stats-duration":
        from .duration import run

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
