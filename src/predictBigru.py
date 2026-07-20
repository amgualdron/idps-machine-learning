"""
predict_bigru.py — forward passes, density plots, and benchmark scatter for the
BiGRU log-normal mixture model.

Two modes:

    # one sequence -> moments + continuous density plot
    python predict_bigru.py single --seq MDVFMKGLSK... --out rg_density.png

    # a YAML of sequences -> table + CSV + benchmark scatter vs reference Rg
    python predict_bigru.py benchmark --yaml config/sequences.yaml --out bench.png

────────────────────────────────────────────────────────────────────────────────
DESIGN NOTE — why this file does NOT redefine the network.

The old predictv2.py / compare.py pair COPIED the entire architecture (AA table,
physics encoders, FiLM, ScaleEncoder, RgMixtureNet, moments) out of the training
script. That is the single most reliable way to get a silently wrong model: the
moment you touch a normalisation, a feature, or an init in training, inference
drifts and nothing raises. (The copy in predictv2.py had already rotted -- its
shd() carried a stray `if 'dmat' in locals()` branch that does not exist in the
trainer, and `List` was used without being imported.)

So: this file IMPORTS from bigru_mixture. One definition of the model, one
definition of tokenize(), one definition of the globals. If training changes,
inference changes with it or fails loudly.
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import argparse
import csv
import sys
from typing import List, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── single source of truth: everything comes from the trainer ────────────────
from bigru_mixture import (
    AA_TO_ID, param_matrix, N_RAW, N_GLOBALS, GLOBAL_NAMES,
    RgBiGRUNet, tokenize, sequence_globals,
    mixture_moments, mixture_log_prob,
)

VALID_AA = set("ARNDCQEGHILKMFPSTWYV")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Checkpoint loading
# ══════════════════════════════════════════════════════════════════════════════
def load_model(weights_path, stats_path):
    """Rebuild the skeleton from the SAVED hyper-params, then load the weights.

    Every architectural knob (hidden, n_layers, embed_dim, use_raw, n_comp) is
    read from the stats file -- never hard-coded here. A skeleton that disagrees
    with the checkpoint would deserialise into garbage, or crash, depending on
    which leaf mismatches first.
    """
    s = dict(np.load(stats_path, allow_pickle=True))

    required = ("r_mu", "r_sd", "g_mu", "g_sd", "nu", "log_a", "n_comp")
    missing = [k for k in required if k not in s]
    if missing:
        raise KeyError(
            f"{stats_path} is missing {missing}. If this is a stats file from the "
            f"OLD multi-scale CNN (it will contain 'p_mu'/'kernel_sizes'), it is not "
            f"compatible -- retrain with bigru_mixture.py."
        )
    if "kernel_sizes" in s and "hidden" not in s:
        raise ValueError(
            f"{stats_path} looks like a CNN-era stats file (has 'kernel_sizes', no "
            f"'hidden'). Point --stats at the BiGRU run's *_stats.npz."
        )

    n_comp    = int(s["n_comp"])
    hidden    = int(s.get("hidden", 64))
    n_layers  = int(s.get("n_layers", 1))
    embed_dim = int(s.get("embed_dim", 16))
    use_raw   = bool(s.get("use_raw", True))

    # g_mu tells us how many globals the model was trained with -- trust it over
    # the current N_GLOBALS, so an old checkpoint fails loudly instead of silently
    # being fed a differently-shaped conditioning vector.
    n_globals = int(np.asarray(s["g_mu"]).shape[0])
    if n_globals != N_GLOBALS:
        print(f"[warn] checkpoint was trained with {n_globals} globals but "
              f"bigru_mixture.py now defines {N_GLOBALS} {GLOBAL_NAMES}. "
              f"Using {n_globals} to match the weights -- but the feature ORDER "
              f"must still line up. Verify before trusting these numbers.",
              file=sys.stderr)

    skeleton = RgBiGRUNet(
        key=jax.random.PRNGKey(0),
        flory_nu=float(s["nu"]), flory_log_a=float(s["log_a"]),
        embed_dim=embed_dim, n_raw=N_RAW if use_raw else 0,
        hidden=hidden, n_layers=n_layers,
        n_globals=n_globals, n_comp=n_comp,
    )
    model = eqx.tree_deserialise_leaves(weights_path, skeleton)

    print(f"[model] loaded {weights_path} | BiGRU hidden={hidden} layers={n_layers} "
          f"embed={embed_dim} raw={use_raw} K={n_comp} | "
          f"Flory: log Rg = {float(s['log_a']):.4f} + {float(s['nu']):.4f} log N")
    return model, s


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Featurisation  (padded batch; identical preprocessing to training)
# ══════════════════════════════════════════════════════════════════════════════
def clean_sequence(seq, name="<seq>"):
    """Uppercase, strip whitespace, and REFUSE non-standard residues.

    Why this is strict: AA_TO_ID.get(aa, 0) maps any unknown letter to token 0 --
    which is the PAD symbol '-'. The mask, however, is built from sequence LENGTH
    and stays 1.0 there. So an 'X' becomes a ghost residue: zero mass, zero charge,
    zero hydropathy, but still counted in L and still fed to the GRU. It is
    completely silent. Sequences from UniProt routinely carry X (unknown), U
    (selenocysteine), and B/Z (ambiguous), so this WILL bite on real data.
    """
    s = "".join(seq.split()).upper()
    bad = sorted(set(s) - VALID_AA)
    if bad:
        raise ValueError(
            f"{name}: non-standard residue(s) {bad}. These would silently become "
            f"the pad token (id 0) while still counting toward length. Substitute "
            f"or drop them explicitly -- e.g. X->G, U->C -- so the choice is yours "
            f"and not an accident."
        )
    if not s:
        raise ValueError(f"{name}: empty sequence")
    return s


def featurize(seqs: Sequence[str], stats, names=None):
    """seqs -> (ids, raw, glob, mask), right-padded to the longest sequence.

    Batching sequences of DIFFERENT lengths is safe here only because BiGRU freezes
    its hidden state wherever mask==0, in both directions. That invariant is
    asserted in bigru_mixture.smoke() and re-checked by --self-test below.
    """
    names = names or [f"seq{i}" for i in range(len(seqs))]
    seqs = [clean_sequence(s, n) for s, n in zip(seqs, names)]

    B = len(seqs)
    L = max(len(s) for s in seqs)
    ids  = np.zeros((B, L), np.int32)
    mask = np.zeros((B, L), np.float32)
    glob = np.zeros((B, N_GLOBALS), np.float32)

    for i, s in enumerate(seqs):
        ids[i], mask[i] = tokenize(s, L)            # SAME tokenize as training
        glob[i] = sequence_globals(ids[i, :len(s)])  # SAME globals as training

    raw = param_matrix[ids]                          # (B, L, 5)

    # ── standardise with the TRAIN-set statistics stored in the checkpoint ────
    raw  = ((raw - stats["r_mu"]) / stats["r_sd"]) * mask[..., None]
    glob = (glob - stats["g_mu"]) / stats["g_sd"]

    return (jnp.array(ids), jnp.array(raw, jnp.float32),
            jnp.array(glob, jnp.float32), jnp.array(mask))


@eqx.filter_jit
def _forward(model, ids, raw, glob, mask):
    return jax.vmap(model)(ids, raw, glob, mask)


def predict(model, stats, seqs, names=None):
    """-> dict of arrays: logit_pi, mu, sigma, mean, std, skew, exkurt  (all (B,...))"""
    ids, raw, glob, mask = featurize(seqs, stats, names)
    logit_pi, mu, sigma = _forward(model, ids, raw, glob, mask)
    m1, sd, sk, ku = jax.vmap(mixture_moments)(logit_pi, mu, sigma)
    return dict(
        logit_pi=np.asarray(logit_pi), mu=np.asarray(mu), sigma=np.asarray(sigma),
        mean=np.asarray(m1), std=np.asarray(sd),
        skew=np.asarray(sk), exkurt=np.asarray(ku),
        length=np.array([len(s) for s in seqs]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Density evaluation
# ══════════════════════════════════════════════════════════════════════════════
def density_grid(mu, sigma, n=1000, pad=4.5):
    """Adaptive grid in Rg, spanning the mixture's own support.

    The old script hard-coded linspace(0, 120) and then plotted xlim(0, max/10) --
    a fudge that silently depends on the training units. Deriving the window from
    (mu, sigma) makes the plot correct in Å or nm without anyone editing a constant.
    """
    lo = float(np.exp(mu.min() - pad * sigma.max()))
    hi = float(np.exp(mu.max() + pad * sigma.max()))
    lo = max(lo * 0.75, 1e-3)
    return jnp.linspace(lo, hi * 1.05, n)


def eval_density(logit_pi, mu, sigma, grid):
    return np.asarray(jnp.exp(mixture_log_prob(
        jnp.asarray(logit_pi), jnp.asarray(mu), jnp.asarray(sigma), grid)))


def component_densities(logit_pi, mu, sigma, grid):
    """Each pi_k-weighted log-normal component separately, for the plot."""
    pi = np.asarray(jax.nn.softmax(jnp.asarray(logit_pi)))
    g = np.asarray(grid)
    out = []
    for k in range(len(mu)):
        y = np.log(np.clip(g, 1e-8, None))
        comp = (np.exp(-0.5 * ((y - mu[k]) / sigma[k]) ** 2)
                / (sigma[k] * np.sqrt(2 * np.pi) * np.clip(g, 1e-8, None)))
        out.append(pi[k] * comp)
    return out, pi


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Plots
# ══════════════════════════════════════════════════════════════════════════════
def plot_single(p, i, unit, out_path, show_components=True, ref_rg=None, title=None):
    lp, mu, sg = p["logit_pi"][i], p["mu"][i], p["sigma"][i]
    m1, sd = float(p["mean"][i]), float(p["std"][i])
    sk, ku = float(p["skew"][i]), float(p["exkurt"][i])
    K = len(mu)

    grid = density_grid(mu, sg)
    g = np.asarray(grid)
    dens = eval_density(lp, mu, sg, grid)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.fill_between(g, dens, alpha=0.22, color="#2ca02c")
    ax.plot(g, dens, color="#2ca02c", lw=2.3, label="Predicted mixture", zorder=3)

    if show_components:
        comps, pi = component_densities(lp, mu, sg, grid)
        order = np.argsort(-pi)
        for rank, k in enumerate(order):
            if pi[k] < 1e-3:                    # dead component -> don't clutter
                continue
            ax.plot(g, comps[k], lw=1.0, alpha=0.75, ls="-",
                    color=plt.cm.viridis(rank / max(K - 1, 1)),
                    label=f"comp {k} ($\\pi$={pi[k]:.2f})" if rank < 4 else None,
                    zorder=2)

    # matched-moment Gaussian: shows what the skew/kurtosis is actually buying you
    gauss = np.exp(-0.5 * ((g - m1) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    ax.plot(g, gauss, "--", color="#7f7f7f", lw=1.3,
            label="Gaussian (same $\\mu,\\sigma$)", zorder=1)

    ax.axvline(m1, color="#d62728", ls=":", lw=1.5, zorder=4)
    if ref_rg is not None:
        ax.axvline(ref_rg, color="#1f77b4", ls="-.", lw=1.5, zorder=4,
                   label=f"Reference ({ref_rg:.2f} {unit})")

    ax.set_xlim(g.min(), g.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"Radius of gyration $R_g$ [{unit}]")
    ax.set_ylabel("Probability density")
    ax.set_title(title or f"Predicted $P(R_g)$ — log-normal mixture, K={K}")

    txt = (f"L        = {int(p['length'][i])}\n"
           f"mean     = {m1:.2f} {unit}\n"
           f"std      = {sd:.2f} {unit}\n"
           f"skew     = {sk:+.3f}\n"
           f"ex-kurt  = {ku:+.3f}")
    ax.text(0.975, 0.94, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#ddd", alpha=0.92))
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] density -> {out_path}")


def plot_benchmark(names, pred, ref, unit, out_path, annotate=None, pred_err=None):
    from scipy.stats import pearsonr, spearmanr

    pred = np.asarray(pred, float)
    ref  = np.asarray(ref, float)

    r, _   = pearsonr(ref, pred)
    rho, _ = spearmanr(ref, pred)
    ss_res = np.sum((ref - pred) ** 2)
    ss_tot = np.sum((ref - ref.mean()) ** 2)
    r2   = 1.0 - ss_res / ss_tot          # R^2 vs the IDENTITY line, not a refit
    mae  = np.mean(np.abs(ref - pred))
    rmse = np.sqrt(np.mean((ref - pred) ** 2))
    bias = np.mean(pred - ref)            # signed: is the model systematically big/small?

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    lo = min(ref.min(), pred.min())
    hi = max(ref.max(), pred.max())
    span = hi - lo
    lo, hi = lo - 0.08 * span, hi + 0.08 * span

    ax.plot([lo, hi], [lo, hi], "--", color="#7f7f7f", alpha=0.75, lw=1.3,
            label="Identity ($y=x$)", zorder=1)
    ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1],
                    color="#7f7f7f", alpha=0.08, zorder=0, label="±10%")

    if pred_err is not None:
        ax.errorbar(ref, pred, yerr=np.asarray(pred_err), fmt="none",
                    ecolor="#1f77b4", alpha=0.35, elinewidth=1.1, capsize=2, zorder=2)
    ax.scatter(ref, pred, color="#1f77b4", s=48, alpha=0.85,
               edgecolors="white", linewidths=0.6, zorder=3)

    ann = set(annotate or names)
    for i, nm in enumerate(names):
        if nm in ann:
            ax.annotate(nm, (ref[i], pred[i]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.5, color="#333")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(f"Reference $R_g$ [{unit}]")
    ax.set_ylabel(f"Predicted mixture mean $R_g$ [{unit}]")
    ax.set_title("BiGRU mixture — validation against reference")

    txt = (f"n     = {len(ref)}\n"
           f"r     = {r:.3f}\n"
           f"rho   = {rho:.3f}\n"
           f"R^2   = {r2:.3f}\n"
           f"MAE   = {mae:.3f} {unit}\n"
           f"RMSE  = {rmse:.3f} {unit}\n"
           f"bias  = {bias:+.3f} {unit}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#ddd", alpha=0.92))
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] benchmark -> {out_path}")
    return dict(n=len(ref), pearson=r, spearman=rho, r2=r2,
                mae=mae, rmse=rmse, bias=bias)


def plot_ridgeline(names, p, unit, out_path, max_show=24):
    """All predicted densities stacked -- the thing a distributional model is FOR.

    A scatter of means throws away everything the mixture head is doing. This is
    the plot that shows whether the widths and shapes vary sensibly across IDPs.
    """
    n = min(len(names), max_show)
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * n + 1.6))
    lo = min(float(np.exp(p["mu"][i].min() - 4 * p["sigma"][i].max())) for i in range(n))
    hi = max(float(np.exp(p["mu"][i].max() + 4 * p["sigma"][i].max())) for i in range(n))
    grid = jnp.linspace(max(lo * 0.8, 1e-3), hi * 1.05, 800)
    g = np.asarray(grid)

    order = np.argsort(p["mean"][:n])
    for row, i in enumerate(order):
        d = eval_density(p["logit_pi"][i], p["mu"][i], p["sigma"][i], grid)
        d = d / d.max() * 0.85
        y = row
        ax.fill_between(g, y, y + d, color=plt.cm.viridis(row / max(n - 1, 1)),
                        alpha=0.55, lw=0)
        ax.plot(g, y + d, color="#333", lw=0.7)
        ax.plot([p["mean"][i]], [y], marker="|", color="#d62728", ms=7)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([names[i] for i in order], fontsize=7.5)
    ax.set_xlabel(f"$R_g$ [{unit}]")
    ax.set_title("Predicted $P(R_g)$ per sequence (red tick = mean)")
    ax.set_xlim(g.min(), g.max())
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] ridgeline -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  YAML handling  (same schema as the old compare.py: name -> {sequence, Rg})
# ══════════════════════════════════════════════════════════════════════════════
def load_yaml(path):
    import yaml
    with open(path) as f:
        y = yaml.safe_load(f)
    if not isinstance(y, dict):
        raise ValueError("sequences.yaml must be a mapping: name -> {sequence: ..., Rg: ...}")

    names, seqs, refs = [], [], []
    for name, v in y.items():
        if isinstance(v, str):                       # allow  name: MDVFMK...
            seq, rg = v, None
        elif isinstance(v, dict):
            if "sequence" not in v:
                raise KeyError(f"{name}: no 'sequence' key")
            seq = v["sequence"]
            rg = v.get("Rg", v.get("rg", None))
        else:
            raise TypeError(f"{name}: expected str or mapping, got {type(v).__name__}")
        names.append(name); seqs.append(seq)
        refs.append(float(rg) if rg is not None else np.nan)
    return names, seqs, np.array(refs, float)


def check_units(pred, ref, unit):
    """Guard against the Å-vs-nm trap.

    The old compare.py printed 'Predicted: {x:.3f} nm' while the model emits Rg in
    whatever units the TRAINING FRAMES used. If those were Ångström, every number
    on that plot was off by 10x and the scatter would sit on a line of slope ~10
    while still reporting a beautiful Pearson r -- because r is scale-invariant.
    That is exactly the failure mode a correlation coefficient cannot see.
    """
    ok = np.isfinite(ref) & np.isfinite(pred)
    if ok.sum() < 2:
        return
    ratio = float(np.median(pred[ok] / ref[ok]))
    if 5.0 < ratio < 20.0:
        print(f"\n[UNITS?] predicted / reference median ratio = {ratio:.2f}. "
              f"That is suspiciously close to 10 -- the model is probably predicting "
              f"Ångström while your reference table is in nm.\n"
              f"         Fix with:  --unit-scale 0.1 --unit nm\n", file=sys.stderr)
    elif 0.05 < ratio < 0.2:
        print(f"\n[UNITS?] predicted / reference median ratio = {ratio:.2f} (~1/10). "
              f"Model likely in nm, reference in Å. Fix with: --unit-scale 10 --unit A\n",
              file=sys.stderr)


def write_csv(path, names, p, ref, unit):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "length", f"pred_mean_{unit}", f"pred_std_{unit}",
                    "pred_skew", "pred_exkurt", f"ref_rg_{unit}", "abs_err", "rel_err"])
        for i, nm in enumerate(names):
            r = ref[i]
            ae = abs(p["mean"][i] - r) if np.isfinite(r) else np.nan
            re_ = ae / r if (np.isfinite(r) and r != 0) else np.nan
            w.writerow([nm, int(p["length"][i]),
                        f"{p['mean'][i]:.4f}", f"{p['std'][i]:.4f}",
                        f"{p['skew'][i]:.4f}", f"{p['exkurt'][i]:.4f}",
                        f"{r:.4f}" if np.isfinite(r) else "",
                        f"{ae:.4f}" if np.isfinite(ae) else "",
                        f"{re_:.4f}" if np.isfinite(re_) else ""])
    print(f"[csv]  -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Self-test:  batched-with-padding == one-at-a-time
# ══════════════════════════════════════════════════════════════════════════════
def self_test(model, stats):
    """The one invariant this script depends on.

    Every benchmark number is produced by vmapping over a RIGHT-PADDED batch of
    mixed-length sequences. If padding leaked into the recurrence, short sequences
    in a batch with a long one would get different answers than they do alone --
    and the error would be quiet, length-dependent, and would look like 'the model
    is just bad at short IDPs'. So: assert it.
    """
    seqs = ["MDVFMKGLSKAKEGVVAAAEK",
            "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQ",
            "GSHMASMTGGQQMGRDLYDDDDKDRWGSELEKAMVALIDVFHQYSGREGDKHKLKKSELKEL"]
    batched = predict(model, stats, seqs)
    worst = 0.0
    for i, s in enumerate(seqs):
        one = predict(model, stats, [s])
        for k in ("mean", "std", "skew", "exkurt"):
            worst = max(worst, abs(float(one[k][0]) - float(batched[k][i])))
    print(f"[self-test] padded-batch vs single-sequence, max |Δmoment| = {worst:.3e}")
    if worst > 1e-4:
        raise AssertionError(
            f"PADDING LEAKS: batching changed the predictions by {worst:.3e}. "
            f"The BiGRU mask-freeze is broken -- every batched number is suspect.")
    print("[self-test] OK — padding is inert, batched inference is trustworthy.")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["single", "benchmark", "selftest"])
    ap.add_argument("--weights", default="rg_bigru_model.eqx")
    ap.add_argument("--stats",   default="rg_bigru_stats.npz")
    ap.add_argument("--unit", default="A", help="label for the Rg axis (A, nm, ...)")
    ap.add_argument("--unit-scale", type=float, default=1.0,
                    help="multiply model outpu  t by this. Model predicts in the units "
                         "of the TRAINING frames; use 0.1 if those were A and your "
                         "reference table is nm.")
    # single
    ap.add_argument("--seq", default=None, help="one sequence (single mode)")
    ap.add_argument("--ref", type=float, default=None, help="reference Rg to overlay")
    # benchmark
    ap.add_argument("--yaml", default="config/sequences.yaml")
    ap.add_argument("--csv", default=None, help="write per-sequence predictions here")
    ap.add_argument("--ridgeline", default=None, help="also write a density ridgeline")
    ap.add_argument("--annotate", nargs="*", default=None,
                    help="names to label on the scatter (default: all)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    model, stats = load_model(a.weights, a.stats)
    sc = a.unit_scale

    if a.mode == "selftest":
        self_test(model, stats)
        return

    if a.mode == "single":
        if not a.seq:
            ap.error("single mode needs --seq")
        p = predict(model, stats, [a.seq])
        for k in ("mean", "std"):
            p[k] = p[k] * sc
        m1, sd = float(p["mean"][0]), float(p["std"][0])
        sk, ku = float(p["skew"][0]), float(p["exkurt"][0])
        # mu lives in LOG space -> a unit rescale is an additive shift there
        p["mu"] = p["mu"] + np.log(sc)
        print(f"\n[prediction]  L = {int(p['length'][0])}")
        print(f"  mean Rg   {m1:8.3f} {a.unit}")
        print(f"  std Rg    {sd:8.3f} {a.unit}   (CV = {sd/m1:.3f})")
        print(f"  skew      {sk:+8.3f}")
        print(f"  ex-kurt   {ku:+8.3f}")
        pi = np.asarray(jax.nn.softmax(jnp.asarray(p["logit_pi"][0])))
        print(f"  live components (pi>0.01): {int((pi > 0.01).sum())}/{len(pi)}  "
              f"pi = {np.array2string(pi, precision=3, suppress_small=True)}")
        plot_single(p, 0, a.unit, a.out or "rg_density.png", ref_rg=a.ref)
        return

    # ── benchmark ────────────────────────────────────────────────────────────
    names, seqs, ref = load_yaml(a.yaml)
    print(f"[yaml] {len(names)} sequences from {a.yaml}")
    p = predict(model, stats, seqs, names)
    p["mean"] = p["mean"] * sc
    p["std"]  = p["std"] * sc
    p["mu"]   = p["mu"] + np.log(sc)

    print(f"\n{'name':<20s} {'L':>4s} {'pred':>9s} {'std':>7s} {'skew':>7s} "
          f"{'exkurt':>7s} {'ref':>9s} {'err':>8s}")
    print("─" * 82)
    for i, nm in enumerate(names):
        r = ref[i]
        err = f"{p['mean'][i] - r:+8.3f}" if np.isfinite(r) else "       —"
        rs = f"{r:9.3f}" if np.isfinite(r) else "        —"
        print(f"{nm:<20s} {int(p['length'][i]):4d} {p['mean'][i]:9.3f} "
              f"{p['std'][i]:7.3f} {p['skew'][i]:+7.3f} {p['exkurt'][i]:+7.3f} "
              f"{rs} {err}")

    check_units(p["mean"], ref, a.unit)

    if a.csv:
        write_csv(a.csv, names, p, ref, a.unit)
    if a.ridgeline:
        plot_ridgeline(names, p, a.unit, a.ridgeline)

    ok = np.isfinite(ref)
    if ok.sum() < 2:
        print("\n[skip] fewer than 2 reference Rg values in the YAML -- no scatter. "
              "Add an 'Rg:' field per entry to benchmark.")
        return

    m = plot_benchmark([n for n, k in zip(names, ok) if k],
                       p["mean"][ok], ref[ok], a.unit,
                       a.out or "model_vs_reference.png",
                       annotate=a.annotate,
                       pred_err=p["std"][ok])
    print(f"\n[metrics] n={m['n']}  r={m['pearson']:.3f}  rho={m['spearman']:.3f}  "
          f"R2={m['r2']:.3f}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
          f"bias={m['bias']:+.3f} {a.unit}")


if __name__ == "__main__":
    main()