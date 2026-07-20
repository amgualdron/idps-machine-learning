"""
benchmark_ensemble.py
======================
Automates batch inference over a YAML sequence list and generates a validation
scatter plot comparing predicted Rg means against literature simulation data (in nm).
"""

from __future__ import annotations
import argparse
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# Reuse your structural model definition from your files
from predictv2 import RgMixtureNet, featurize, mixture_moments

# ══════════════════════════════════════════════════════════════════════════════
# Ground Truth Reference Data (Directly in Nanometers matching your table)
# ══════════════════════════════════════════════════════════════════════════════


def load_inference_setup(weights_path, stats_path):
    s_file = np.load(stats_path, allow_pickle=True)
    stats = dict(s_file)
    kernels = tuple(int(k) for k in stats["kernel_sizes"])
    n_comp = int(stats["n_comp"])

    skeleton = RgMixtureNet(
        key=jax.random.PRNGKey(0), 
        flory_nu=float(stats["nu"]), 
        flory_log_a=float(stats["log_a"]),
        kernel_sizes=kernels, 
        n_comp=n_comp, 
        n_globals=2
    )
    model = eqx.tree_deserialise_leaves(weights_path, skeleton)
    return model, stats, kernels

def generate_benchmark_plot(names, predicted, experimental, output_path):
    pred_arr = np.array(predicted)
    exp_arr = np.array(experimental)
    
    # Calculate performance metrics
    r, _ = pearsonr(exp_arr, pred_arr)
    r2 = r2_score(exp_arr, pred_arr)
    mae = np.mean(np.abs(exp_arr - pred_arr))

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    
    # Draw identity line
    min_val = min(exp_arr.min(), pred_arr.min()) - 0.3
    max_val = max(exp_arr.max(), pred_arr.max()) + 0.3
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="#7f7f7f", alpha=0.7, label="Identity ($y=x$)")
    
    # Scatter plot
    ax.scatter(exp_arr, pred_arr, color="#1f77b4", s=45, alpha=0.8, edgecolors="none")
    
    # Label landmarks dynamically to preserve clarity
    for i, name in enumerate(names):
        if name in ["alpha_synuclein", "sNase", "OPN", "ProTa_N","CspTm", "hCyp", "Protein_L", "R15", "R17",'Histatin-5','Nucleoporin153','ERMTADn']:
            ax.annotate(name, (exp_arr[i], pred_arr[i]), textcoords="offset points", 
                        xytext=(0,6), ha='center', fontsize=8, color="#333")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', 'box')
    
    ax.set_xlabel("Simulation $R_g$ (nm)")
    ax.set_ylabel("Predicted Mixture Mean $R_g$ (nm)")
    ax.set_title("Ensemble Model Validation Performance")
    
    metrics_text = f"Pearson $r$ = {r:.3f}\n$R^2$ Score   = {r2:.3f}\nMAE        = {mae:.3f} nm"
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            family="monospace", bbox=dict(boxstyle="round", facecolor="white", edgecolor="#ddd", alpha=0.9))
    
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"[plot] Benchmark visualization saved to -> {output_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="config/sequences.yaml")
    ap.add_argument("--weights", default="rg_mixture_model.eqx")
    ap.add_argument("--stats", default="rg_mixture_stats.npz") 
    ap.add_argument("--out", default="model_vs_simulation.png")
    a = ap.parse_args()

    with open(a.yaml, "r") as f:
        yaml_content = yaml.safe_load(f)
    
    if not isinstance(yaml_content, dict):
        raise ValueError("Your sequences.yaml must be structured as key-value pairs (e.g., 'synuclein: MDV...')")

    model, stats, kernels = load_inference_setup(a.weights, a.stats)
    
    benchmark_names = []
    predicted_means = []
    simulation_means = []

    print("\n--- Running Ensemble Batch Inference (nm) ---")
    for idp_name in yaml_content.keys():
        clean_seq = yaml_content[idp_name]['sequence']
            
        feats = featurize(clean_seq, stats, kernels)
        logit_pi, mu, sigma = jax.vmap(model)(*feats)
        
        # Pull analytical moment vector directly as standard Python scalar float
        m1_jax, _, _, _ = mixture_moments(logit_pi[0], mu[0], sigma[0])
        pred_rg = float(m1_jax)
        sim_rg = yaml_content[idp_name]['Rg']
        
        benchmark_names.append(idp_name)
        predicted_means.append(pred_rg)
        simulation_means.append(sim_rg)
        
        print(f"  {idp_name:<15s} | L = {len(clean_seq):3d} | Predicted: {pred_rg:.3f} nm | Target: {sim_rg:.3f} nm")

    if not predicted_means:
        print("[error] No benchmark matches found between the YAML file and the table registry.")
        return
        
    generate_benchmark_plot(benchmark_names, predicted_means, simulation_means, a.out)

if __name__ == "__main__":
    main()