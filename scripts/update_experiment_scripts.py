"""Mechanical relocation of pilot path expressions; leave frozen originals untouched."""

import argparse
import ast
import json
from pathlib import Path


def transform(source):
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    changes = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        replacement = None
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.left, ast.Name)
            and node.left.id == "D"
        ):
            replacement = "artifact(" + ast.get_source_segment(source, node.right) + ")"
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "D"
        ):
            if node.attr == "parent":
                replacement = "RUNS"
            elif node.attr == "name":
                replacement = "ROOT.name"
        if replacement is not None:
            start = offsets[node.lineno - 1] + node.col_offset
            end = offsets[node.end_lineno - 1] + node.end_col_offset
            changes.append((start, end, replacement))
    for start, end, replacement in sorted(changes, reverse=True):
        source = source[:start] + replacement + source[end:]
    source = source.replace("D = Path(__file__).resolve().parent", "D = ROOT")
    source = source.replace("D=Path(__file__).resolve().parent", "D = ROOT")
    source = source.replace(
        "Path(__file__).resolve().parent / 'evaluation'", "artifact('evaluation')"
    )
    if "artifact(" in source or "D = ROOT" in source:
        source = (
            "from settings import ROOT, RUNS, OUTPUT, SETTINGS, artifact, child\n"
            + source
        )
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    options = parser.parse_args()
    patches = []
    # Controllers and tracking are replaced by the shared launcher separately.
    for name in (
        "prepare",
        "extract",
        "screen",
        "generate",
        "supplement",
        "prepare_training",
        "infer",
        "score",
        "verify_data",
    ):
        path = options.root / "scripts" / (name + ".py")
        original = path.read_text()
        updated = transform(original)
        patches.append(
            f"*** Update File: {path}\n@@\n"
            + "\n".join("-" + line for line in original.splitlines())
            + "\n"
            + "\n".join("+" + line for line in updated.splitlines())
            + "\n"
        )
    print(json.dumps("*** Begin Patch\n" + "".join(patches) + "*** End Patch"))


if __name__ == "__main__":
    main()
