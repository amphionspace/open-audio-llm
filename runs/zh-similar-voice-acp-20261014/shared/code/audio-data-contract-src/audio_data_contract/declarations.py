"""Read YAML declarations and legacy JSONL catalogs; preserve comments on edits."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from ruamel.yaml.nodes import SequenceNode

from .errors import ContractError


def declaration_files(path: str | Path) -> list[Path]:
    selected = Path(path)
    sources = (
        sorted(
            p for p in selected.iterdir()
            if p.is_file() and p.suffix in {".yaml", ".yml", ".jsonl"}
        )
        if selected.is_dir() else [selected]
    )
    if not sources:
        raise ContractError(f"no YAML or JSONL declarations found: {selected}")
    return sources


def read_declarations(path: str | Path) -> Iterator[tuple[int, dict]]:
    """Yield each declaration with its source line for errors and overview links."""
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        if source.suffix == ".jsonl":
            for number, line in enumerate(stream, 1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ContractError(f"{source}:{number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ContractError(f"{source}:{number}: declaration must be an object")
                yield number, row
            return
        if source.suffix not in {".yaml", ".yml"}:
            raise ContractError(f"unsupported declaration format: {source}")
        parser = YAML(typ="safe")
        try:
            node = parser.compose(stream)
            if not isinstance(node, SequenceNode):
                raise ContractError(f"{source}: expected a YAML list of declarations")
            rows = parser.constructor.construct_document(node)
        except YAMLError as exc:
            raise ContractError(f"{source}: {exc}") from exc
        for child, row in zip(node.value, rows):
            number = child.start_mark.line + 1
            if not isinstance(row, dict):
                raise ContractError(f"{source}:{number}: declaration must be an object")
            yield number, row


def editable_declarations(path: str | Path) -> list:
    """Load YAML for an update without discarding comments or quoted versions."""
    source = Path(path)
    if source.suffix == ".jsonl":
        return [row for _, row in read_declarations(source)]
    parser = YAML()
    parser.preserve_quotes = True
    try:
        rows = parser.load(source)
    except YAMLError as exc:
        raise ContractError(f"{source}: {exc}") from exc
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ContractError(f"{source}: expected a YAML list of declarations")
    return rows


def write_declarations(path: str | Path, rows: list) -> None:
    destination = Path(path)
    if destination.suffix not in {".yaml", ".yml", ".jsonl"}:
        raise ContractError(f"unsupported declaration format: {destination}")
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        if destination.suffix == ".jsonl":
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            parser = YAML()
            parser.allow_unicode = True
            parser.width = 512
            parser.indent(mapping=2, sequence=4, offset=2)
            parser.dump(rows, stream)
    temporary.replace(destination)
