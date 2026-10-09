#!/usr/bin/env bash
set -euo pipefail

# Usage: bash download_runs_resumable.sh [remote Runs path] [local Runs path]
# Run from the local Make-Directories directory. Re-run the same command
# after an interruption; rsync transfers only missing or changed files.

REMOTE_RUNS="${1:-/groups/aniket/Grafted/Batch-Jobs-ewall_0.0/Make-Directories/Runs}"
LOCAL_RUNS="${2:-$PWD/Runs}"
REMOTE_USER_HOST="abhattac@stokes.ist.ucf.edu"

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required; install it or use a Mac that has it available." >&2
    exit 1
fi

mkdir -p "$LOCAL_RUNS"
echo "Downloading ${REMOTE_USER_HOST}:${REMOTE_RUNS}/ into ${LOCAL_RUNS}/"
rsync -av --partial --progress -e ssh \
    "${REMOTE_USER_HOST}:${REMOTE_RUNS%/}/" "${LOCAL_RUNS%/}/"
echo "Transfer complete. Re-running this command will check for any remaining files."