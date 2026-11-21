"""Prepare the existing vLLM architecture override without copying model weights."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    config = json.loads((args.source / "config.json").read_text())
    config["architectures"] = [args.architecture]
    for item in args.source.iterdir():
        if item.name not in {
            "config.json",
            "configuration_audio_llm.py",
            "modeling_audio_llm.py",
        }:
            (args.output / item.name).symlink_to(item.absolute())
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ("configuration_audio_llm.py", "modeling_audio_llm.py"):
        shutil.copy2(args.code / name, args.output / name)


if __name__ == "__main__":
    main()
