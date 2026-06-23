"""Build `sample_index.jsonl` from Lhotse recordings/supervisions manifests."""

from __future__ import annotations

import argparse

from open_audio_llm.data.lhotse_index_builder import build_index_file


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", required=True)
    parser.add_argument("--supervisions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--task", default="asr")
    parser.add_argument("--language", default="N/A")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    build_index_file(
        args.recordings,
        args.supervisions,
        args.output,
        dataset_id=args.dataset_id,
        task=args.task,
        language=args.language,
    )


if __name__ == "__main__":
    main()
