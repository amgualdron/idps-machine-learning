# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline for running coarse-grained (HPS/Mpipi-style, one-bead-per-residue) molecular
dynamics simulations of intrinsically disordered proteins (IDPs) in HOOMD-blue, extracting
structural observables (Rg, Ree, asphericity) from the resulting trajectories, and pooling
many runs into a dataset for downstream ML work.

## Environment

Python code runs in the conda env `md-env` (`/home/andres/miniforge3/envs/md-env`), which has
`hoomd`, `MDAnalysis`, `jax`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `tqdm`,
`scikit-learn`, `pyyaml`. There is no `environment.yml`/`requirements.txt` checked in yet — use
`conda activate md-env` before running anything, or invoke the env's Python directly.

`src/simulation.py` requires a GPU (`hoomd.device.GPU()` is hardcoded) — it cannot run on a
CPU-only machine.

There is no test suite, linter, or formatter configured in this repo. `src/test.py` is a
throwaway scratch/debug script (prints `project_root`), not a real test — don't treat it as one.

## Pipeline / architecture

Everything flows through **run directories** under `runs/`, each one a self-contained record of
one experiment batch:

```
config/config.yaml          → control panel: which sequences, how many replicates, which
                               param_sets, runner (local|slurm)
config/sequences.yaml       → sequence database: name -> {sequence, length, description}
config/physics.yaml         → named physics parameter sets (e.g. "baseline", "cold"); entries
                               can `extends:` another set to inherit+override fields

        │  src/generate_jobs.py --config config/config.yaml
        ▼
runs/<run_name>_<MMDD>/
  job_manifest.csv           one row per task_id: sequence, sequence_name, seed, and every
                              physics param from physics.yaml flattened as param_<key>
  logs/<seq>_task_<id>.json  per-task status ("pending"→"running"→"completed"), written by
                              generate_jobs.py and updated in place by simulation.py — this is
                              how downstream steps know a job succeeded and where its data is
  data/timeseries/           pooled per-sequence CSVs written by analysis.py
  trajectories/              raw *.gsd trajectory files written by simulation.py

        │  src/run_local.py --run_dir runs/<run_name>_<MMDD>   (loops the manifest, shells out
        │                                                        to simulation.py per task_id)
        │  (or slurm array job indexing the same manifest by $SLURM_ARRAY_TASK_ID — not yet
        │   present in this repo, only referenced in README)
        ▼
src/simulation.py --manifest <manifest> --task_id <N>
  Reads exactly one row of job_manifest.csv, builds a linear-chain HOOMD snapshot for that
  sequence, assigns per-residue amino-acid parameters (mass/charge/sigma/hydropathy from the
  hardcoded AA_PARAMS table — HPS1 = Dignon et al. 2018, HPS2 = Tesei et al. 2021 scales),
  derives electrostatics (Yukawa via a temperature/ionic-strength-dependent Debye length) and
  Rouse-time-based equilibration/production step counts when not explicitly overridden, runs
  equilibration then production, and streams frames to trajectories/<seq>_traj_<id>.gsd.
  Updates the matching logs/*.json with status/timing/frame count as it goes.

        │  src/analysis.py --run_dir runs/<run_name>_<MMDD>
        ▼
  Groups job_manifest.csv by sequence_name, and for every task whose log says "completed" and
  whose GSD file exists, uses MDAnalysis to compute per-frame Rg, Rg², Ree², asphericity
  (center-of-mass unwrapping handles PBC without needing bond topology). Concatenates all
  replicate/seed runs for a sequence into one
  data/timeseries/<seq>_pooled_timeseries.csv.

        │  src/quick_plot.py <path/to/*_pooled_timeseries.csv>
        ▼
  Fits an exponentially-modified-Gaussian to the Rg distribution and saves a two-panel
  histogram+KDE plot as <seq_name>_plot.png in the current directory.
```

`notebook/ml-data.ipynb` is a separate, offline aggregation step: it scans a large external
directory of many past run outputs (not this repo's `runs/`), parses per-folder physics
parameters and trajectory statistics in parallel, and builds `idp_ml_dataset.parquet` — the
actual ML training dataset. `notebook/notebook.ipynb` is exploratory/scratch (JAX reimplementation
of the AA parameter table, sequence loading experiments).

## Key conventions to preserve when editing this pipeline

- **The manifest is the single source of truth for a run.** Every stage (simulation, analysis)
  re-derives everything it needs from `job_manifest.csv` + the matching `logs/*.json` rather than
  taking parameters as flags. If you add a new physics parameter, add it in `physics.yaml` and it
  will automatically flow into the manifest as `param_<key>` — `simulation.py` reads it off the
  manifest row by that same name.
- **Derived quantities (KT, Yukawa epsilon/kappa, equilibration/production step counts) are
  computed in `simulation.py`, never stored in the YAML configs** — the physics configs only hold
  the inputs to those formulas (temperature, ionic concentration, multipliers).
- **`equilibration_steps`/`production_steps` being null in `physics.yaml` means "derive from the
  Rouse time"** (`rouse_multiplier`/`production_multiplier` × N^2.2 / dt); a non-null value is a
  hard override. Same null-means-derive pattern applies to `frames`.
- **Logs are the completion contract between stages.** `analysis.py` will silently skip any task
  whose log is missing, not `"status": "completed"`, or whose referenced trajectory file doesn't
  exist — it's designed to tolerate partial/failed runs in a batch rather than crash.
- Amino acid parameters (`AA_PARAMS`) are currently duplicated between `src/simulation.py` and
  `notebook/notebook.ipynb`; keep them in sync if either changes.
