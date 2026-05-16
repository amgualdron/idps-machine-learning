```text 
project/
│
├── configs/                          # CONTROL PANEL
│   ├── sequences.yaml                # Full sequence database: ID -> AA string + metadata
│   ├── physics.yaml                  # Named parameter sets (default, cold)
│   └── config.yaml                   # control panel, choose sequences to run, steps, simulation parameters     
│                                
├── R15_R17_full_run_0414/
│   ├── data/                         # processed data (Rg, anything else needed for ml) 
│   ├── job_manifest.csv              # stores sequence,task id, parameters, trajectory file location
│   ├── logs/                         # per job info, avoids race conditions, metadata. 
│   │   ├── R15_task_1.json
│   │   ├── R15_task_2.json
│   │   ├── R17_task_3.json
│   │   └── R17_task_4.json
│   └── trajectories/                # raw output files
│       ├── R15_traj_1.gsd
│       ├── R15_traj_2.gsd
│       ├── R17_traj_3.gsd
│       └── R17_traj_4.gsd
│
├── src/                              # Core 
│   ├── simulation.py                 # HOOMD script, reads job_manifest(only one line), writes logs and trajectories, called by run_local or slurm
│   ├── analysis.py                   # Rg, contacts, asphericity, distributions, etc.
│   ├── run_local.py                  # reads jobs_manifest, calls simulation per task
│   └── gen_jobs.py                   # reads config.yaml, creates run directory and sub directories, creates job_manifest.csv
│
├── slurm/
│   └── submit_array.sh               # Indexes into manifest.csv via $SLURM_ARRAY_TASK_ID
│
├── environment.yml
└── README.md
project/
│
├── configs/                          # CONTROL PANEL
│   ├── sequences.yaml                # Full sequence database: ID -> AA string + metadata
│   ├── physics.yaml                  # Named parameter sets (default, cold)
│   └── config.yaml                   # control panel, choose sequences to run, steps, simulation parameters     
│                                
├── R15_R17_full_run_0414/
│   ├── data/                         # processed data (Rg, anything else needed for ml) 
│   ├── job_manifest.csv              # stores sequence,task id, parameters, trajectory file location
│   ├── logs/                         # per job info, avoids race conditions, metadata. 
│   │   ├── R15_task_1.json
│   │   ├── R15_task_2.json
│   │   ├── R17_task_3.json
│   │   └── R17_task_4.json
│   └── trajectories/                # raw output files
│       ├── R15_traj_1.gsd
│       ├── R15_traj_2.gsd
│       ├── R17_traj_3.gsd
│       └── R17_traj_4.gsd
│
├── src/                              # Core 
│   ├── simulation.py                 # HOOMD script, reads job_manifest(only one line), writes logs and trajectories, called by run_local or slurm
│   ├── analysis.py                   # Rg, contacts, asphericity, distributions, etc.
│   ├── run_local.py                  # reads jobs_manifest, calls simulation per task
│   └── gen_jobs.py                   # reads config.yaml, creates run directory and sub directories, creates job_manifest.csv
│
├── slurm/
│   └── submit_array.sh               # Indexes into manifest.csv via $SLURM_ARRAY_TASK_ID
│
├── environment.yml
└── README.md
``` `
