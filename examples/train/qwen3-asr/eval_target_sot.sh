#!/usr/bin/env bash
set -eo pipefail
# Credentials remain in the shell environment; never copy them to a recipe.
source "$HOME/.bashrc" >/dev/null 2>&1
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export AUDIO_DATA_CONTRACT_ROOT="${AUDIO_DATA_CONTRACT_ROOT:-$(dirname "$REPO_ROOT")/audio-data-contract}"
export PYTHONPATH="$REPO_ROOT/src:$AUDIO_DATA_CONTRACT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m open_audio_llm.eval.target_sot_vllm "$@"
