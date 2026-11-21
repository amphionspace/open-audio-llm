#!/usr/bin/env python3
"""OpenAI-compatible vLLM server for Qwen3-ASR (vLLM 0.18 vendored backend).

`vllm serve` inspects Qwen3ASR in a subprocess and SIGSEGVs. Register the
class in-process first (same as src/tsasr/eval_hf.py), then start the CLI.

Usage:
  python src/tsasr/serve_qwen3.py /path/to/ckpt --port 8001 --served-model-name NAME ...
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vllm018_compat import bootstrap_vllm_runtime, register_vendored_backend  # noqa: E402
from global_audio_attn import describe_mode, vllm_hf_overrides  # noqa: E402

bootstrap_vllm_runtime()
register_vendored_backend()
print(describe_mode(), flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: serve_qwen3.py <model> [vllm serve args...]", file=sys.stderr)
        sys.exit(2)
    extra = sys.argv[2:]
    overrides = vllm_hf_overrides()
    if overrides and "--hf-overrides" not in extra:
        import json

        extra = ["--hf-overrides", json.dumps(overrides), *extra]
    sys.argv = ["vllm", "serve", sys.argv[1], *extra]
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
