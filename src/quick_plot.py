import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import exponnorm

def main():
    parser = argparse.ArgumentParser(description="Plot Rg distributions from timeseries CSV.")
    parser.add_argument("csv_file", help="Path to the *_pooled_timeseries.csv file")
    args = parser.parse_args()

    # 1. Load the data
    print(f"Loading {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    rg_data = df['rg'].dropna().values
    seq_name = df['sequence_name'].iloc[0] if 'sequence_name' in df.columns else "Unknown Sequence"

    # 2. Fit the Exponnorm distribution
    print("Fitting Exponentially Modified Gaussian...")
    K, loc, scale = exponnorm.fit(rg_data)
    print(f"Fit Results -> K: {K:.3f}, loc: {loc:.3f}, scale: {scale:.3f}")

    # 3. Setup the plotting style
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Histogram + Theoretical Fit ---
    ax1 = axes[0]
    # Plot empirical histogram (stat="density" normalizes it so the curve fits over it)
    sns.histplot(rg_data, bins=50, stat="density", alpha=0.5, color="royalblue", 
                 edgecolor="black", label="Simulation Data", ax=ax1)

    # Calculate and plot the theoretical fit curve
    x = np.linspace(rg_data.min() - 2, rg_data.max() + 2, 500)
    pdf = exponnorm.pdf(x, K, loc, scale)
    ax1.plot(x, pdf, 'k-', lw=3, label=f'Exponnorm Fit\nK={K:.2f}, $\\mu$={loc:.2f}, $\\sigma$={scale:.2f}')

    ax1.set_title(f"Radius of Gyration ($R_g$) Histogram", pad=15)
    ax1.set_xlabel("$R_g$ (Å)")
    ax1.set_ylabel("Probability Density")
    ax1.legend()

    # --- Panel 2: Distogram (Kernel Density Estimate) ---
    ax2 = axes[1]
    # KDE is a smoothed, continuous distogram of the raw data
    sns.kdeplot(rg_data, fill=True, color="darkorange", alpha=0.5, lw=2, ax=ax2)
    
    # Add a line for the mean to anchor the visualization
    mean_rg = np.mean(rg_data)
    ax2.axvline(mean_rg, color='red', linestyle='--', lw=2, label=f'Mean $R_g$: {mean_rg:.5f} Å')
    
    ax2.set_title("Continuous Distogram (KDE)", pad=15)
    ax2.set_xlabel("$R_g$ (Å)")
    ax2.set_ylabel("Density")
    ax2.legend()

    # Final layout adjustments
    plt.suptitle(f"Structural Distribution Progress: {seq_name}", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure locally so you don't lose it
    out_file = f"{seq_name}_plot.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_file}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()