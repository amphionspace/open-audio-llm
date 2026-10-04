#!/usr/bin/env bash
# Parameters live in YAML; accepts --config, --dry-run and --resume only.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec open-audio-llm serve --config "$REPO_ROOT/examples/configs/serve/vllm.yaml" "$@"
