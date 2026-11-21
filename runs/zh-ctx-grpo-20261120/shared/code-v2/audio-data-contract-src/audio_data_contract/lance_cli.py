"""Opt-in Lance commands, separate from training/catalog entry points."""

import argparse
import json

from .lance import (
    LanceArtifact,
    RecordQuery,
    catalog_ref,
    cleanup_artifact,
    compact_artifact,
    import_jsonl,
    inspect_table,
    materialize_layer,
    mirror_artifact,
    protect_artifact,
    rebuild_artifact,
    verify_equivalence,
)
from .lance_export import export_artifact
from .layers import write_layer
from .roots import load_roots
from .types import TransformStep


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        help="roots file; import stores root_alias references instead of absolute "
        "paths (reads also honor AUDIO_DATA_ROOTS_FILE)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("import")
    create.add_argument("source")
    create.add_argument("destination")
    create.add_argument("--source-view", required=True)
    create.add_argument("--batch-size", type=int, default=4096)
    create.add_argument("--workers", type=int, default=1)
    export = commands.add_parser("export")
    export.add_argument("artifact")
    export.add_argument("destination")
    export.add_argument("--id", action="append")
    export.add_argument("--task")
    export.add_argument("--language")
    export.add_argument("--split")
    export.add_argument("--clean-pass", choices=["true", "false"])
    patch = commands.add_parser("patch")
    patch.add_argument("artifact")
    patch.add_argument("patches", help="JSONL rows with id, changes, status")
    patch.add_argument("transform", help="TransformStep JSON file")
    patch.add_argument("destination", help="new canonical Layer directory")
    patch.add_argument("--source-view", required=True)
    patch.add_argument("--tool", required=True)
    patch.add_argument("--model")
    for name in ("rebuild", "verify"):
        command = commands.add_parser(name)
        command.add_argument("artifact")
        command.add_argument("destination" if name == "rebuild" else "source")
    compact = commands.add_parser("compact", help="publish a compacted sidecar")
    compact.add_argument("artifact")
    compact.add_argument("destination", help="new sidecar JSON path")
    cleanup = commands.add_parser("cleanup", help="delete old untagged versions")
    cleanup.add_argument("artifact")
    cleanup.add_argument("--older-than-days", type=int, required=True)
    mirror = commands.add_parser("mirror", help="copy a table to another root")
    mirror.add_argument("artifact")
    mirror.add_argument("target_root", help="e.g. s3://bucket/prefix (AWS_* env)")
    for name, text in (
        ("protect", "tag this sidecar's snapshot"),
        ("doctor", "report published/unpublished versions"),
    ):
        commands.add_parser(name, help=text).add_argument("artifact")
    register = commands.add_parser("catalog-ref", help="print a catalog ArtifactRef")
    register.add_argument("artifact")
    register.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    roots = load_roots(args.roots, allow_urls=True) if args.roots else None
    if args.command == "import":
        result = import_jsonl(
            args.source,
            args.destination,
            source_view=args.source_view,
            batch_size=args.batch_size,
            roots=roots,
            workers=args.workers,
        ).to_dict()
    else:
        artifact = LanceArtifact.read(args.artifact, roots)
        if args.command == "compact":
            result = compact_artifact(artifact, args.destination).to_dict()
        elif args.command == "cleanup":
            result = cleanup_artifact(artifact, older_than_days=args.older_than_days)
        elif args.command == "mirror":
            result = mirror_artifact(artifact, args.target_root)
        elif args.command == "protect":
            result = {"published": protect_artifact(artifact)}
        elif args.command == "doctor":
            result = inspect_table(artifact)
        elif args.command == "catalog-ref":
            result = catalog_ref(
                artifact, args.artifact, name=args.name, roots=artifact.location_roots()
            ).to_dict()
        elif args.command == "export":
            query = RecordQuery(
                ids=tuple(args.id) if args.id is not None else None,
                task=args.task,
                language=args.language,
                split=args.split,
                clean_pass=None
                if args.clean_pass is None
                else args.clean_pass == "true",
            )
            export_artifact(artifact, args.destination, query=query)
            result = {"output": args.destination}
        elif args.command == "patch":
            with open(args.transform) as stream:
                step = TransformStep.from_dict(json.load(stream))
            with open(args.patches) as stream:
                layer = write_layer(
                    args.destination,
                    (json.loads(line) for line in stream if line.strip()),
                    parent=artifact,
                    step=step,
                    tool=args.tool,
                    model=args.model,
                )
            result = materialize_layer(
                layer, source_view=args.source_view, roots=roots
            ).to_dict()
        elif args.command == "rebuild":
            result = rebuild_artifact(artifact, args.destination).to_dict()
        else:
            result = verify_equivalence(artifact, args.source)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
