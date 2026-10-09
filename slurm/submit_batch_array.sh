#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Sizes and submits one SLURM array job covering every row of a run's
# job_manifest.csv (produced by `src/generate_jobs.py`). This is the
# cluster-side counterpart to src/run_local.py: where run_local.py loops the
# manifest in-process one task at a time, this submits all tasks as a single
# array job that SLURM schedules in parallel across available nodes.
#
# Usage: slurm/submit_batch_array.sh <run_dir> [throttle]
#   <run_dir>   runs/<name>_<MMDD> directory already containing job_manifest.csv
#   [throttle]  optional cap on array tasks running at once, e.g. 50 — keeps a
#               1000-row manifest from claiming the whole queue at once. Not
#               required by Stokes as far as we know, just considerate; add one
#               if your jobs stall in queue because you asked for more than
#               what's available at once.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <run_dir> [throttle]" >&2
    exit 1
fi
RUN_DIR="$1"
THROTTLE="${2:-}"
MANIFEST="$RUN_DIR/job_manifest.csv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$MANIFEST" ]; then
    echo "Manifest not found: $MANIFEST (run src/generate_jobs.py first)" >&2
    exit 1
fi

N_TASKS=$(($(wc -l < "$MANIFEST") - 1))  # minus header row
if [ "$N_TASKS" -lt 1 ]; then
    echo "Manifest has no job rows: $MANIFEST" >&2
    exit 1
fi

ARRAY_SPEC="1-${N_TASKS}"
if [ -n "$THROTTLE" ]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${THROTTLE}"
fi

echo "Submitting $N_TASKS array tasks for $RUN_DIR (array=$ARRAY_SPEC)"
sbatch --array="$ARRAY_SPEC" "$SCRIPT_DIR/submit_array.sh" "$RUN_DIR"
