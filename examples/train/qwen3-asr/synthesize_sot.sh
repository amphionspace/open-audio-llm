#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMPHION_DATA_ROOT="${AMPHION_DATA_ROOT:-$SCRIPT_DIR/../../../../AmphionData}"
AUDIO_DATA_CONTRACT_ROOT="${AUDIO_DATA_CONTRACT_ROOT:-$SCRIPT_DIR/../../../../audio-data-contract}"
AMPHION_DATA_PYTHON="${AMPHION_DATA_PYTHON:-$AMPHION_DATA_ROOT/.venv/bin/python}"
export PYTHONPATH="$AMPHION_DATA_ROOT/src:$AUDIO_DATA_CONTRACT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$AMPHION_DATA_PYTHON" -m amphiondata multispeaker "$@"
