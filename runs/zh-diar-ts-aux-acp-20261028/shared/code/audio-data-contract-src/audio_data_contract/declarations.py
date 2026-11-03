"""Read YAML declarations and legacy JSONL catalogs; preserve comments on edits."""

from __future__ import annotations

import json
import shutil
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


def _faithful_yaml(text: str) -> YAML | None:
    """A round-trip parser whose dump reproduces `text` exactly, or None."""
    import io

    first = next(
        (line for line in text.split("\n") if line.strip() and not line.lstrip().startswith("#")),
        "",
    )
    for width in (4096, 80, 120, 512):
        parser = YAML()
        parser.preserve_quotes = True
        parser.allow_unicode = True
        parser.width = width
        if first.startswith("- "):
            parser.indent(mapping=2, sequence=2, offset=0)
        else:
            parser.indent(mapping=2, sequence=4, offset=2)
        output = io.StringIO()
        parser.dump(parser.load(text), output)
        if output.getvalue() == text:
            return parser
    return None


def edit_declaration(path: str | Path, dataset_id: str, version: str, edit) -> bool:
    """Apply `edit(row)` to one declaration in a YAML file, changing nothing else.

    Returns False when the file does not declare dataset_id@version. Refuses files
    whose formatting a rewrite would not reproduce, instead of reformatting them.
    """
    import io

    source = Path(path)
    if source.suffix not in {".yaml", ".yml"}:
        return False
    text = source.read_text(encoding="utf-8")
    try:
        parser = YAML()
        rows = parser.load(text)
    except YAMLError as exc:
        raise ContractError(f"{source}: {exc}") from exc
    if not isinstance(rows, list):
        return False
    matches = [
        index
        for index, row in enumerate(rows)
        if isinstance(row, dict)
        and row.get("dataset_id") == dataset_id
        and str(row.get("version")) == version
    ]
    if not matches:
        return False
    parser = _faithful_yaml(text)
    if parser is None:
        raise ContractError(f"{source}: rewriting would reformat this file; edit it by hand")
    rows = parser.load(text)
    edit(rows[matches[0]])
    output = io.StringIO()
    parser.dump(rows, output)
    temporary = source.with_name(source.name + ".tmp")
    temporary.write_text(output.getvalue(), encoding="utf-8")
    shutil.copymode(source, temporary)
    temporary.replace(source)
    return True
