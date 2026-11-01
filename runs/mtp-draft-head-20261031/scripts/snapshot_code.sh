#!/usr/bin/env bash
# Freeze the committed code for the ACP job (the container mounts /workspace only, so the job
# must not read a worktree that may move on). Usage: snapshot_code.sh [COMMIT]
set -euo pipefail
EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(cd "$EXP/../.." && pwd)
COMMIT=$(git -C "$REPO" rev-parse "${1:-HEAD}")
DEST=$EXP/shared/code
if [[ -e $DEST ]]; then
    echo "$DEST exists; snapshots are not overwritten" >&2
    exit 1
fi
mkdir -p "$DEST"
git -C "$REPO" archive "$COMMIT" src | tar -x -C "$DEST"
echo "$COMMIT" > "$DEST/COMMIT"
echo "snapshot $COMMIT -> $DEST"
