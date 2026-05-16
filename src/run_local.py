import argparse
import pandas as pd
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run MD jobs locally from a manifest.")
    parser.add_argument("--run_dir", required=True, help="Path to the experiment directory (e.g., runs/my_run_0412_1)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "job_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    total_jobs = len(manifest_df)

    print(f"Found {total_jobs} jobs. Starting local execution...")

    # Iterate through the manifest
    for index, row in manifest_df.iterrows():
        task_id = int(row['task_id'])
        seq_name = row['sequence_name']
        
        print(f"\n[{index + 1}/{total_jobs}] Starting Task {task_id} ({seq_name})...")

        # Call simulation.py using subprocess. 
        # We pass the manifest path and the specific task_id it needs to run.
        command = [
            "python", "src/simulation.py",
            "--manifest", str(manifest_path),
            "--task_id", str(task_id)
        ]

        # subprocess.run will wait for simulation.py to finish before moving to the next loop iteration
        result = subprocess.run(command)

        # Basic error handling
        if result.returncode == 0:
            print(f"Task {task_id} completed successfully.")
        else:
            print(f"Task {task_id} FAILED with return code {result.returncode}.")
            # Optional: break here if you want the pipeline to stop on the first error
            # break

    print("\nAll local jobs finished!")

if __name__ == "__main__":
    main()