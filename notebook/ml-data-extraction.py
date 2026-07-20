"""
Build the IDP-ML dataset for sequence -> P(Rg) training.

Design (see reasoning in comments):
  * Moments (mean/std/skew/kurt) are computed on the FULL trajectory --- they are
    low-variance shape targets and must not be re-estimated from a thinned set.
  * The KDE distogram is DROPPED. Training now uses continuous NLL against raw
    frames, so no bandwidth / grid / bin resolution is baked into the target.
  * Each trajectory is randomly subsampled to a fixed budget of K frames. MD frames
    are heavily autocorrelated, so a uniform random subsample is an unbiased draw
    from the same empirical P(Rg) and also decorrelates for free. K frames per
    protein keeps the whole frame store to ~1-2 GB instead of ~4 billion floats.
  * Frames are stored as float32 (Rg is single-digit nm; float32 gives ~7 sig figs,
    well past MD precision --- float16 would quantize the tails where skew/kurt live).
  * Output is split by access pattern:
        - metadata (sequence, length, moments, env params)  -> parquet
        - frames, shape (P, K), row-aligned to the parquet  -> memmappable .npy
    Row i of the .npy corresponds to row i of the parquet (both built from the same
    folder_id-sorted list; an explicit `frame_row` column documents the contract).
  * Subsampling is reproducible: the RNG is seeded per folder by its integer id.
  * Original running_stat.dat files are the source of truth --- never deleted. This
    .npy is a derived artifact; regenerate it (different K, proper autocorr, audit)
    from source whenever needed.
"""

import os
import sys
import re
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR     = Path("/run/media/andres/2TB-Black/IDPAI/HPS1/Runs")
OUTPUT_FILE  = "idp_ml_dataset.parquet"   # metadata + moments (one row per protein)
FRAMES_FILE  = "idp_rg_frames.npy"        # (P, K) float32, row-aligned to parquet
K_FRAMES     = 30_000                     # frames kept per protein (generous; ~1.2 GB @ 10k proteins)
NUM_CORES    = os.cpu_count()

print("BASE_DIR exists:", BASE_DIR.exists())


# ══════════════════════════════════════════════════════════════════════════════
# Environmental-parameter parsing  (module level --- was nested inside the sanity
# checks in the original, which would have made process_single_folder throw NameError)
# ══════════════════════════════════════════════════════════════════════════════
def parse_parameters(folder_path):
    """Parse System_Parameters.dat if present; fall back to sane defaults."""
    param_file = folder_path / "System_Parameters.dat"
    params = {}
    if not param_file.exists():
        return params
    try:
        with open(param_file, "r") as f:
            content = f.read()
        ionic   = re.search(r"Ionic concentration.*:\s*([\d\.]+)", content)
        kappa   = re.search(r"Yukawa kappa.*:\s*([\d\.]+)",        content)
        epsilon = re.search(r"Yukawa epsilon.*:\s*([\d\.]+)",      content)
        bjerrum = re.search(r"Bjerrum length.*:\s*([\d\.]+)",      content)
        temp    = re.search(r"Temperature.*:\s*([\d\.]+)",         content)
        params["ionic_concentration_M"]   = float(ionic.group(1))   if ionic   else 0.15
        params["yukawa_kappa_nm_inv"]     = float(kappa.group(1))   if kappa   else 1.273787
        params["yukawa_epsilon_kcal_mol"] = float(epsilon.group(1)) if epsilon else 0.423234
        params["bjerrum_length_nm"]       = float(bjerrum.group(1)) if bjerrum else 0.714697
        params["temperature_K"]           = float(temp.group(1))    if temp    else 298.0
    except Exception:
        pass  # defaults filled downstream if some keys are missing
    return params


# ══════════════════════════════════════════════════════════════════════════════
# Per-folder worker
# ══════════════════════════════════════════════════════════════════════════════
def _load_rg_frames(running_file):
    """Fast read of column 2 (0-indexed) from running_stat.dat.

    pandas read_csv is dramatically faster than np.loadtxt on million-row files.
    """
    df_run = pd.read_csv(
        running_file,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[2],
        engine="c",
    )
    return df_run.iloc[:, 0].to_numpy()


def _subsample(rg_values, k, seed):
    """Uniform random subsample to exactly k frames.

    total >= k : draw without replacement (plain thinning of the empirical dist).
    total <  k : draw WITH replacement --- statistically clean, it's just resampling
                 the empirical distribution again, and keeps the output rectangular
                 so the frame store stays a simple memmappable (P, K) array.
    """
    rng = np.random.default_rng(seed)
    n = len(rg_values)
    replace = n < k
    idx = rng.choice(n, size=k, replace=replace)
    return rg_values[idx].astype(np.float32), replace


def process_single_folder(folder_path):
    """Parse one simulation folder; return (metadata_dict, frames_float32) or None."""
    try:
        # 1. Sequence
        seq_file = folder_path / "OneLetterCode.dat"
        if not seq_file.exists():
            return None
        with open(seq_file, "r") as f:
            sequence = "".join(line.strip() for line in f if line.strip())

        # 2. config_stat.dat scalar metrics
        config_file = folder_path / "config_stat.dat"
        config_data = np.loadtxt(config_file, comments="#")
        if config_data.ndim > 1:
            config_data = config_data[0]
        particle_n         = int(config_data[0])
        persistence_length = config_data[2]   # <Lp>
        r_end_sq           = config_data[3]   # <Rend2>
        avg_rg_config      = config_data[4]   # <Rg>

        # 3. Full Rg trajectory
        running_file = folder_path / "running_stat.dat"
        rg_values = _load_rg_frames(running_file)
        rg_values = np.atleast_1d(rg_values)
        total_frames = len(rg_values)
        if total_frames == 0:
            return None

        # 4. Moments on the FULL trajectory (low-variance targets; not from subsample)
        rg_mean = float(np.mean(rg_values))
        rg_std  = float(np.std(rg_values))
        rg_skew = float(skew(rg_values))
        rg_kurt = float(kurtosis(rg_values))   # excess kurtosis (Fisher); Gaussian -> 0

        # 5. Subsample frames to fixed budget (reproducible per folder)
        seed = int(folder_path.name)  # folder names are digit strings
        frames, was_padded = _subsample(rg_values, K_FRAMES, seed)

        # 6. Environment
        env_params = parse_parameters(folder_path)

        # 7. Assemble metadata
        result = {
            "folder_id":            folder_path.name,
            "sequence":             sequence,
            "seq_length":           particle_n,
            "persistence_length_lp": persistence_length,
            "mean_r_end_sq":        r_end_sq,
            "config_avg_rg":        avg_rg_config,
            "total_frames":         total_frames,      # TRUE count, pre-subsample
            "kept_frames":          K_FRAMES,
            "was_padded":           was_padded,        # True if run shorter than K
            "calculated_rg_mean":   rg_mean,
            "calculated_rg_std":    rg_std,
            "calculated_rg_skew":   rg_skew,
            "calculated_rg_kurtosis": rg_kurt,
        }
        result.update(env_params)
        return result, frames

    except Exception as e:
        print(f"Error in folder {folder_path.name}: {e}", file=sys.stderr)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Sanity checks (unchanged in intent; distogram was never used here)
# ══════════════════════════════════════════════════════════════════════════════
def run_dataset_sanity_checks(df):
    mismatches = df[df["sequence"].str.len() != df["seq_length"]]
    print(f"[Data Integrity] {len(mismatches)} sequence-length discrepancies.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["seq_length"], bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    ax.set_title("Distribution of IDP Sequence Lengths")
    ax.set_xlabel("Sequence Length ($N$)"); ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
    plt.savefig("sanity_check_length_hist.png", dpi=300)
    print("Saved: sanity_check_length_hist.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["seq_length"], df["calculated_rg_mean"], color="darkgreen",
               alpha=0.5, label="Simulated IDPs")
    log_N, log_Rg = np.log(df["seq_length"]), np.log(df["calculated_rg_mean"])
    nu, log_a = np.polyfit(log_N, log_Rg, 1)
    N_grid = np.linspace(df["seq_length"].min(), df["seq_length"].max(), 100)
    ax.plot(N_grid, np.exp(log_a) * N_grid ** nu, "r--", lw=2,
            label=f"Fit: $R_g \\propto N^{{{nu:.3f}}}$")
    ax.set_title(r"Flory Scaling Sanity Check ($R_g$ vs. $N$)")
    ax.set_xlabel("Sequence Length ($N$)"); ax.set_ylabel("Mean $R_g$ [nm]")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
    plt.savefig("sanity_check_flory_scaling.png", dpi=300)
    print("Saved: sanity_check_flory_scaling.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["total_frames"], bins=15, color="gold", edgecolor="black", alpha=0.7)
    ax.set_title("MD Sampling Depth (Total Frames, pre-subsample)")
    ax.set_xlabel("Total Frames"); ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
    plt.savefig("sanity_check_total_frames.png", dpi=300)
    print("Saved: sanity_check_total_frames.png")

    rg_sq_mean = df["calculated_rg_mean"] ** 2 + df["calculated_rg_std"] ** 2
    ratio = df["mean_r_end_sq"] / rg_sq_mean
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["seq_length"], ratio, color="purple", alpha=0.5, label="Simulated Chains")
    ax.axhline(6.0, color="black", lw=1.5, label="Ideal Linear Chain (6.0)")
    ax.axhline(6.4, color="orange", ls="--", lw=1.5, label="SAW Limit (~6.4)")
    ax.set_title(r"Polymer Shape Factor ($\langle R_e^2 \rangle / \langle R_g^2 \rangle$)")
    ax.set_xlabel("Sequence Length ($N$)")
    ax.set_ylabel(r"$\langle R_e^2 \rangle / \langle R_g^2 \rangle$")
    ax.set_ylim(2, 10); ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
    plt.savefig("sanity_check_polymer_ratio.png", dpi=300)
    print("Saved: sanity_check_polymer_ratio.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Scanning directories...")
    all_folders = [f for f in BASE_DIR.iterdir() if f.is_dir() and f.name.isdigit()]
    print(f"Found {len(all_folders)} folders to parse.")

    print(f"Spinning up {NUM_CORES} CPU processes...")
    collected = []
    with Pool(processes=NUM_CORES) as pool:
        for res in pool.imap_unordered(process_single_folder, all_folders, chunksize=15):
            if res is not None:
                collected.append(res)
            if len(collected) % 500 == 0 and collected:
                print(f"Progress: {len(collected)} / {len(all_folders)} parsed...")

    if not collected:
        print("No folders parsed successfully. Aborting.", file=sys.stderr)
        return

    # Sort by integer folder_id so parquet rows and frame rows share one ordering.
    collected.sort(key=lambda rf: int(rf[0]["folder_id"]))
    meta_rows = [rf[0] for rf in collected]
    frame_list = [rf[1] for rf in collected]

    print("Building DataFrame...")
    df = pd.DataFrame(meta_rows)
    df["folder_id"] = df["folder_id"].astype(int)
    df["frame_row"] = np.arange(len(df), dtype=np.int64)  # explicit row <-> .npy contract

    print("Stacking frames...")
    # (P, K) float32. ~1.2 GB at K=30k, P=10k. If K is pushed much higher and RAM is
    # tight, preallocate a np.lib.format.open_memmap of shape (P, K) and fill row by
    # row instead of np.stack --- but at these sizes the simple stack is fine.
    frames = np.stack(frame_list).astype(np.float32)
    assert frames.shape[0] == len(df), "frame/metadata row-count mismatch"

    n_padded = int(df["was_padded"].sum())
    print(f"[frames] shape={frames.shape} dtype={frames.dtype} "
          f"({frames.nbytes / 1e9:.2f} GB) | {n_padded} proteins padded (run < K)")

    print(f"Writing metadata -> {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, engine="pyarrow", index=False)

    print(f"Writing frames   -> {FRAMES_FILE}")
    np.save(FRAMES_FILE, frames)

    print("Extraction successful.")
    print("Load at train time with: frames = np.load('idp_rg_frames.npy', mmap_mode='r')")
    print("Row p of frames  <->  parquet row where frame_row == p.")

    # Optional: uncomment to regenerate the sanity-check figures.
    # run_dataset_sanity_checks(df)


if __name__ == "__main__":
    main()