import argparse
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
import hoomd
import hoomd.md
import hoomd.write
import numpy as np
from tqdm import tqdm

# ── Config loading ─────────────────────────────────────────────────────────────
project_root = Path(__file__).parent.parent


def update_metadata(log_path: Path, updates: dict):
    """Merge updates into log_path.json in place."""

    with open(log_path) as f:
        meta = json.load(f)
    meta.update(updates)
    with open(log_path, "w") as f:
        json.dump(meta, f, indent=2)

def run_with_progress(sim, total_steps, description="Running"):
    """Runs a HOOMD simulation in chunks to update a tqdm progress bar."""
    
    # Calculate a chunk size. 
    # We update the bar 100 times, or every 1000 steps (whichever is larger)
    # This prevents the Python loop from slowing down the fast GPU calculations.
    chunk_size = max(1000, total_steps // 100)
    
    with tqdm(total=total_steps, desc=description, dynamic_ncols=True) as pbar:
        steps_run = 0
        while steps_run < total_steps:
            steps_to_run = min(chunk_size, total_steps - steps_run)
            sim.run(steps_to_run)
            pbar.update(steps_to_run)
            steps_run += steps_to_run

def main():
    # ── Parse args and load ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to jobs_manifest.csv")
    parser.add_argument("--task_id", type=int, required=True, help="Task ID to execute")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    RUN_DIR = manifest_path.parent
    
    # 1. Load the manifest and find the specific row for this task
    df = pd.read_csv(manifest_path)
    row = df[df['task_id'] == args.task_id].iloc[0]


    # ── Unpack ─────────────────────────────────────────────────────────────────────
    name = row["sequence_name"]
    sequence = row["sequence"]
    T_Kelvin = row["param_temperature_K"]
    DT = row["param_dt"]
    EPSILON = row["param_epsilon_lj"]
    ionic_concentration = row["param_ionic_concentration_M"]
    BOX_SIZE = row["param_box_size"]
    hps_scale = row["param_hydropathy_scale"]
    ph = row.get("ph", None)
    N = len(sequence)

    LOG_FILE = RUN_DIR / "logs" / f"{name}_task_{args.task_id}.json"

    # ── Derived quantities (never stored in yaml) ──────────────────────────────────
    KT = T_Kelvin * 0.001987204259
    epsw = (
        5321 / T_Kelvin
        + 233.76
        - 0.9297 * T_Kelvin
        + 0.1417e-2 * T_Kelvin**2
        - 0.8292e-6 * T_Kelvin**3
    )
    lB = (1.6021766**2 / (4 * np.pi * 8.854188 * epsw)) * (6.022 * 1000 / KT) / 4.184
    yukawa_eps = lB * KT
    yukawa_kappa = np.sqrt(8 * np.pi * lB * ionic_concentration * 6.022 / 10)

    EQUIL_STEPS = row.get("param_equilibration_steps")
    if pd.isna(EQUIL_STEPS):
        EQUIL_STEPS = int((row["param_rouse_multiplier"] * N**2.2) / DT)
    else:
        EQUIL_STEPS = int(EQUIL_STEPS) # Cast it just in case it's a numpy float

    N_STEPS = row.get("param_production_steps")
    if pd.isna(N_STEPS):
        N_STEPS = int((row["param_production_multiplier"] * N**2.2) / DT)
    else:
        N_STEPS = int(N_STEPS)

    SAVE_EVERY = int(N_STEPS/row["param_frames"])
    # ── Amino Acid Force Field Parameters ─────────────────────────────────────────
    # masses in g/mol, sigmas in nm, charges in elementary units
    # HPS1: Dignon et al. 2018, HPS2: Tesei et al. 2021

    AA_PARAMS = {
        "A": dict(mass=71.08, charge=0.0, sigma=0.504, HPS1=0.730, HPS2=0.003),
        "R": dict(mass=156.20, charge=1.0, sigma=0.656, HPS1=0.000, HPS2=0.723),
        "N": dict(mass=114.10, charge=0.0, sigma=0.568, HPS1=0.432, HPS2=0.160),
        "D": dict(mass=115.10, charge=-1.0, sigma=0.558, HPS1=0.378, HPS2=0.002),
        "C": dict(mass=103.10, charge=0.0, sigma=0.548, HPS1=0.595, HPS2=0.400),
        "Q": dict(mass=128.10, charge=0.0, sigma=0.602, HPS1=0.514, HPS2=0.468),
        "E": dict(mass=129.10, charge=-1.0, sigma=0.592, HPS1=0.459, HPS2=0.022),
        "G": dict(mass=57.05, charge=0.0, sigma=0.450, HPS1=0.649, HPS2=0.784),
        "H": dict(mass=137.10, charge=0.5, sigma=0.608, HPS1=0.514, HPS2=0.487),
        "I": dict(mass=113.20, charge=0.0, sigma=0.618, HPS1=0.973, HPS2=0.687),
        "L": dict(mass=113.20, charge=0.0, sigma=0.618, HPS1=0.973, HPS2=0.335),
        "K": dict(mass=128.20, charge=1.0, sigma=0.636, HPS1=0.514, HPS2=0.095),
        "M": dict(mass=131.20, charge=0.0, sigma=0.618, HPS1=0.838, HPS2=0.993),
        "F": dict(mass=147.20, charge=0.0, sigma=0.636, HPS1=1.000, HPS2=0.871),
        "P": dict(mass=97.12, charge=0.0, sigma=0.556, HPS1=1.000, HPS2=0.471),
        "S": dict(mass=87.08, charge=0.0, sigma=0.518, HPS1=0.595, HPS2=0.487),
        "T": dict(mass=101.10, charge=0.0, sigma=0.562, HPS1=0.676, HPS2=0.274),
        "W": dict(mass=186.20, charge=0.0, sigma=0.678, HPS1=0.946, HPS2=0.753),
        "Y": dict(mass=163.20, charge=0.0, sigma=0.646, HPS1=0.865, HPS2=0.984),
        "V": dict(mass=99.07, charge=0.0, sigma=0.586, HPS1=0.892, HPS2=0.428),
    }

    # ── Derived arrays ───────────────────────────────────
    AA_ORDER = list(AA_PARAMS.keys())
    ids = {aa: i for i, aa in enumerate(AA_ORDER)}

    # vectorized arrays for construction of sequence values
    lambdas = np.array([AA_PARAMS[aa][hps_scale] for aa in AA_ORDER], dtype=np.float32)
    masses = np.array([AA_PARAMS[aa]["mass"] for aa in AA_ORDER], dtype=np.float32)
    charges = np.array([AA_PARAMS[aa]["charge"] for aa in AA_ORDER], dtype=np.float32)
    sigmas = np.array([AA_PARAMS[aa]["sigma"] for aa in AA_ORDER], dtype=np.float32)

    seq_id = np.array([ids[aa] for aa in sequence], dtype=np.int32)
    seq_mass = masses[seq_id]
    seq_charge = charges[seq_id]
    seq_sigma = sigmas[seq_id]
    seq_hp = lambdas[seq_id]

    # ── Output path ───────────────────────────────────────────────────────────────
    GSD_FILE = str(RUN_DIR / "trajectories" / f"{name}_traj_{args.task_id}.gsd")

    # ── Build snapshot ────────────────────────────────────────────────────────────
    device = hoomd.device.GPU()
    seed = int(row["seed"])
    sim = hoomd.Simulation(device=device, seed=seed)

    snapshot = hoomd.Snapshot()
    snapshot.configuration.box = [BOX_SIZE, BOX_SIZE, BOX_SIZE, 0, 0, 0]
    snapshot.particles.types = AA_ORDER
    snapshot.particles.N = N
    snapshot.bonds.N = N - 1  # must pre-allocate before populate_snapshot
    snapshot.bonds.types = ["AA-bond"]
    # Particles
    snapshot.particles.typeid[:] = seq_id
    snapshot.particles.mass[:] = seq_mass
    snapshot.particles.charge[:] = seq_charge
    snapshot.particles.diameter[:] = seq_sigma

    # Place beads in a straight line with ~0.38 nm spacing (realistic Cα-Cα)
    snapshot.particles.position[:] = [[i * 0.38 - (N * 0.38) / 2, 0, 0] for i in range(N)]

    # Bonds: chain connectivity
    snapshot.bonds.N = N - 1
    snapshot.bonds.types = ["AA-bond"]
    snapshot.bonds.typeid[:] = [0] * (N - 1)
    snapshot.bonds.group[:] = [[i, i + 1] for i in range(N - 1)]

    sim.create_state_from_snapshot(snapshot)

    # ── Forces ────────────────────────────────────────────────────────────────────
    nl = hoomd.md.nlist.Tree(buffer=0.2)
    nl.exclusions = ["bond"]
    lj1 = hoomd.md.pair.LJ(nlist=nl, mode="shift")
    lj2 = hoomd.md.pair.LJ(nlist=nl)


    yukawa = hoomd.md.pair.Yukawa(nlist=nl, default_r_cut=3.5, mode="shift")

    for i, aa1 in enumerate(AA_ORDER):
        for j, aa2 in enumerate(AA_ORDER):
            if j < i:
                continue
            sig_ij = (sigmas[i] + sigmas[j]) / 2
            lam_ij = (lambdas[i] + lambdas[j]) / 2
            lj2.params[(aa1, aa2)] = dict(epsilon=EPSILON * lam_ij, sigma=sig_ij)
            lj1.params[(aa1, aa2)] = dict(epsilon=EPSILON * (1 - lam_ij), sigma=sig_ij)

            lj1.r_cut[(aa1, aa2)] = 2 ** (1 / 6) * sig_ij  # purely repulsive (WCA)

            lj2.r_cut[(aa1, aa2)] = 3.5  # attractive
            yukawa.params[(aa1, aa2)] = dict(
                epsilon=yukawa_eps * charges[i] * charges[j], kappa=yukawa_kappa
            )
            yukawa.r_cut[(aa1, aa2)] = 3.5

    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params["AA-bond"] = dict(k=row["param_harmonic_k"], r0=row["param_harmonic_r0"])
    # ── Integrator ────────────────────────────────────────────────────────────────
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=KT)
    sim.operations.integrator = hoomd.md.Integrator(
        dt=DT, methods=[langevin], forces=[lj1, lj2, yukawa, harmonic]
    )

    # ── Writer ────────────────────────────────────────────────────────────────────
    gsd_writer = hoomd.write.GSD(
        filename=GSD_FILE, trigger=hoomd.trigger.Periodic(SAVE_EVERY), mode="wb"
    )
    sim.operations.writers.append(gsd_writer)

    update_metadata(
        LOG_FILE,
        {
            "status": "running",
            "seed": int(seed),
            "KT": float(round(KT, 6)),
            "equilibration_steps": int(EQUIL_STEPS),
            "production_steps": int(N_STEPS),
            "total_steps": int(EQUIL_STEPS + N_STEPS),
            "trajectory_file": GSD_FILE,
        },
    )

    print(f"Sequence:     {sequence} ({N} residues)")
    print(f"Temperature:  {T_Kelvin} K")
    print(f"Scale:        {hps_scale}")
    print(f"Equilibration:{EQUIL_STEPS} steps")
    print(
        f"Production:   {N_STEPS} steps -> {N_STEPS // SAVE_EVERY} frames to {GSD_FILE}\n"
    )

    # ── Equilibration (no writing) ────────────────────────────────────────────────
    t_start = time.time()

    run_with_progress(sim, EQUIL_STEPS, description=f"{name} Task {args.task_id} (Equilibrating)")

    # ── Production ────────────────────────────────────────────────────────────────
    run_with_progress(sim, N_STEPS, description=f"{name} Task {args.task_id} (Production)")
    gsd_writer.flush()

    # ── Finalize ──────────────────────────────────────────────────────────────────
    wall_time = round(time.time() - t_start)
    n_frames = N_STEPS // SAVE_EVERY

    update_metadata(
        LOG_FILE,
        {
            "status": "completed",
            "wall_time_s": int(wall_time),
            "n_frames": int(n_frames),  # n_frames is int64 because SAVE_EVERY came from pandas
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    print(f"Wrote {n_frames} frames to {GSD_FILE}")
    print(f"Wall time: {wall_time}s")
    print("Simulation complete!")

if __name__ == "__main__":
    main()