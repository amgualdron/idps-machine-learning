import argparse
import itertools
import json
import pandas as pd
import random
from datetime import datetime
from pathlib import Path

import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_yaml(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def create_unique_dir(base_name):
    path = Path(base_name)
    counter = 1
    
    # Keep incrementing the counter if the path already exists
    # e.g., 'data', then 'data (1)', then 'data (2)'
    new_path = path
    while new_path.exists():
        new_path = Path(f"{base_name}_({counter})")
        counter += 1
        
    new_path.mkdir()
    return new_path

# ── Main Generator Logic ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate job directories and manifests."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the master run_config.yaml"
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    run_config = load_yaml(config_path)

    params_path= project_root / "config" / "physics.yaml"
    params = load_yaml(params_path)

    timestamp = datetime.now().strftime("%m%d")
    exp_name = run_config.get("run_name", "unnamed_run")
    exp_id = f"{exp_name}_{timestamp}"

    # 1. Load Sequences
    seq_source_path = config_path.parent / run_config["sequences"]["source"]
    all_sequences = load_yaml(seq_source_path)  

    # 2. Sequence Selection Logic
    selection = run_config["sequences"].get("select", "all")
    selected_seqs = {}

    if isinstance(selection, list):
        # Explicit list provided
        for name in selection:
            if name in all_sequences:
                selected_seqs[name] = all_sequences[name]
            else:
                print(f"Warning: Sequence '{name}' not found in {seq_source_path.name}")

    elif selection == "all":
        # Full database
        selected_seqs = all_sequences

    # 3. Setup Experiment Directory
    exp_dir = create_unique_dir(project_root / "runs" / exp_id)
    # Create logs directory for SLURM output
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "data").mkdir(exist_ok=True)
    (exp_dir / "trajectories").mkdir(exist_ok=True)

    manifest_path = exp_dir / "job_manifest.csv"
    manifest_rows = []

    # 4. The Cartesian Product Loop (Sequences x num_replicates)
    task_id = 1
    for seq_name, replicate, param_set in itertools.product(
        selected_seqs.keys(), range(run_config["num_replicates"]), run_config["param_sets"]
    ):
        seq_data = selected_seqs[seq_name]
        seq_string = seq_data["sequence"]

        # Write Stage 1 run_metadata.json
        meta = {
            "task_id": task_id,
            "status": "pending",  # Ready to be updated by simulation.py
        }

        with open(exp_dir / "logs" / f"{seq_name}_task_{task_id}.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Append to manifest tracking
        manifest_row = {
                "task_id": task_id,
                "sequence_name": seq_name,
                "sequence": seq_string,
                "seed": random.randint(1000, 9999),  # Using random seed for replicates
            }
        for key,value in params[param_set].items():
            manifest_row[f'param_{key}'] = value

        manifest_rows.append(manifest_row)
        task_id += 1

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False)

    # 6. Generate Execution Script
    total_jobs = len(manifest_rows)
    runner = run_config.get("runner", "local")
    print(f"✅ Generated {total_jobs} jobs in {exp_dir}")
    print(f"📄 Manifest written to: {manifest_path}")

if __name__ == "__main__":
    main()