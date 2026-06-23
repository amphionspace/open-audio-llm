#!/usr/bin/env python3
"""Convert Amphion-4B sft-format JSONL to Qwen3-ASR (amphion_asr_1.7b) format.

Amphion-4B format (user-only):
    messages = [
        {"role": "user",      "content": "Given the speaker's voice:<audio>\\nTranscribe...\\nHotwords: xxx\\n<audio>"},
        {"role": "assistant", "content": "转写文本"},
    ]

Qwen3-ASR format (system + user):
    messages = [
        {"role": "system",    "content": "Given the speaker's voice in the first audio.\\nHotwords: xxx"},
        {"role": "user",      "content": "<audio><audio>"},   ← ONLY audio placeholders
        {"role": "assistant", "content": "转写文本"},
    ]

Usage::

    python convert_to_qwen3asr_fmt.py input.jsonl output.jsonl
    python convert_to_qwen3asr_fmt.py input.jsonl output.jsonl --workers 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── audio placeholder ──────────────────────────────────────────────────────────
_AUDIO_PH = "<audio>"


def _parse_old_user(user_content: str, audios: list) -> tuple[bool, str | None]:
    """Return (has_enrollment, hotwords_str) from the old Amphion-4B user string."""
    has_enrollment = len(audios) == 2
    # The original build_unified_instruction appends <audio> directly to the
    # last line with no newline, so "Hotwords: xxx<audio>" is one segment.
    m = re.search(r"Hotwords:\s*(.+?)(?=<audio>|\n|$)", user_content)
    hotwords_str = m.group(1).strip() if m else None
    return has_enrollment, hotwords_str


def convert_sample(line: str) -> str:
    """Convert one JSONL line from Amphion-4B format to Qwen3-ASR format."""
    d = json.loads(line)
    msgs = d.get("messages", [])
    audios = d.get("audios", [])

    # ── already in new format (has a system message) ──────────────────────────
    if any(m["role"] == "system" for m in msgs):
        return line  # pass-through

    usr_content = next((m["content"] for m in msgs if m["role"] == "user"), "")
    ast_content = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

    has_enrollment, hotwords_str = _parse_old_user(usr_content, audios)

    # ── build system content (same logic as build_qwen3_asr_system) ───────────
    sys_lines: list[str] = []
    if has_enrollment:
        sys_lines.append("Given the speaker's voice in the first audio.")
    if hotwords_str and hotwords_str != "N/A":
        sys_lines.append(f"Hotwords: {hotwords_str}")
    sys_content = "\n".join(sys_lines)

    # ── build user content (audio placeholders only) ──────────────────────────
    usr_new = _AUDIO_PH + _AUDIO_PH if has_enrollment else _AUDIO_PH

    # ── reassemble ────────────────────────────────────────────────────────────
    d["messages"] = [
        {"role": "system",    "content": sys_content},
        {"role": "user",      "content": usr_new},
        {"role": "assistant", "content": ast_content},
    ]
    return json.dumps(d, ensure_ascii=False) + "\n"


def convert_file(src: Path, dst: Path, workers: int = 1) -> None:
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    total = len(lines)
    print(f"[convert] {src.name}: {total:,} lines → {dst}")

    if workers <= 1:
        converted = [convert_sample(l) for l in lines]
    else:
        converted = [""] * total
        chunk = max(1, total // (workers * 4))
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as exe:
            for i in range(0, total, chunk):
                batch = lines[i: i + chunk]
                fut = exe.submit(_convert_batch, batch)
                futures[fut] = i
            for fut in as_completed(futures):
                start = futures[fut]
                for j, out in enumerate(fut.result()):
                    converted[start + j] = out

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(converted), encoding="utf-8")
    print(f"[convert] done → {dst}")


def _convert_batch(lines: list[str]) -> list[str]:
    return [convert_sample(l) for l in lines]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input",  type=Path, help="Input JSONL (Amphion-4B format)")
    ap.add_argument("output", type=Path, help="Output JSONL (Qwen3-ASR format)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel workers (default: 4)")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: input not found: {args.input}")
    if args.output.exists():
        print(f"[convert] WARNING: output already exists, overwriting: {args.output}")

    convert_file(args.input, args.output, workers=args.workers)


if __name__ == "__main__":
    main()
