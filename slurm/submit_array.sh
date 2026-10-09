#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Per-task SLURM array script for this pipeline, CPU-only (Stokes has no GPUs).
# Do not submit this directly — use slurm/submit_batch_array.sh, which sizes
# the --array range from the manifest and calls `sbatch` on this file for you.
#
# One array task == one row of job_manifest.csv == one simulation.py run.
# $SLURM_ARRAY_TASK_ID is passed straight through as --task_id, so it must
# land in the same 1..N range as the task_id column generate_jobs.py wrote.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=idp-md
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # EDIT after calibration — see "Calibrate before
                                    # committing a walltime" below. HOOMD's CPU device
                                    # threads via TBB; for IDP-sized chains (tens to a
                                    # few hundred beads) returns past ~4 cores are
                                    # usually small, so more isn't automatically better
                                    # and just burns core-hours against the 80k/month cap.
#SBATCH --time=24:00:00            # EDIT per batch. No max is documented for Stokes, but
                                    # there is no checkpoint/restart in simulation.py — a
                                    # run killed by the scheduler for hitting --time loses
                                    # all progress and the task must restart from scratch.
                                    # Size this to comfortably cover your longest sequence
                                    # in the batch (see calibration note), not the average.

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: sbatch --array=1-N slurm/submit_array.sh <run_dir>" >&2
    exit 1
fi
RUN_DIR="$1"
MANIFEST="$RUN_DIR/job_manifest.csv"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Environment ────────────────────────────────────────────────────────────
# EDIT: match your cluster's actual module name (this mirrors the PI's
# submit_create_directories_Stokes_fixed.sh, which used anaconda-2023.09).
module purge
module load anaconda/anaconda-2023.09
# EDIT: this must be a CPU-variant HOOMD build, not the gpu_py3* build this repo
# uses on the author's workstation GPU — see the one-time cluster setup note.
source activate md-env

cd "$REPO_ROOT"

python src/simulation.py \
    --manifest "$MANIFEST" \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --device cpu

module purge
