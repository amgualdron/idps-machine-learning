"""
predict_residual.py — forward passes, density plots, benchmarks and order-sensitivity
diagnostics for the *residual* model:  MLP(globals) + BiGRU(sequence) correction.

Four modes:

    # one sequence -> moments + density, with the frozen-MLP baseline overlaid
    python predict_residual.py single --seq MDVFMKGLSK... --out rg_density.png

    # a YAML of sequences -> table + CSV + benchmark scatter vs reference Rg
    python predict_residual.py benchmark --yaml config/sequences.yaml --out bench.png

    # does the sequence branch actually buy anything in Rg-space?
    # baseline vs residual, side by side, same references
    python predict_residual.py compare --yaml config/sequences.yaml --out compare.png

    # is the BiGRU using ORDER, or just re-encoding composition?
    # shuffles residues (composition-preserving) and measures how the correction moves
    python predict_residual.py shuffle --yaml config/sequences.yaml --n-shuffle 20

────────────────────────────────────────────────────────────────────────────────
DESIGN NOTE — Single source of truth.
Featurisation, tokenisation, the head decode and the mixture algebra are imported
from residual.py / mlp_mixture.py / bigru_mixture.py. Plot machinery is imported
from predict_mlp.py. Nothing about the model is re-implemented here, so if the
physics or the head parameterisation changes, this script follows or fails loudly.

DESIGN NOTE — Padding invariance.
The BiGRU freezes its hidden state on PAD steps and both pooling ops are masked,
so the prediction does not depend on how much padding a sequence is batched with.
`selftest` asserts this: a sequence predicted alone (no padding) must match the
same sequence predicted inside a batch padded to the longest member.
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import argparse
import csv
import sys
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bigru_mixture import AA_TO_ID, sequence_globals
from mlp_mixture import RgMLPNet, mixture_moments, mixture_log_prob
from residual import ResidualRgNet, SeqResidual, PAD_ID

# Reuse the baseline script's plotting / IO verbatim — same axes, same metrics.
from predict_mlp import (
    VALID_AA, clean_sequence, density_grid, eval_density, component_densities,
    plot_benchmark, plot_ridgeline, load_yaml, check_units,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Checkpoint Loading
# ══════════════════════════════════════════════════════════════════════════════
def load_model(weights_path, stats_path):
    """Rebuild base MLP + BiGRU residual from the saved hyper-parameters, then load.

    Every architectural knob (width, depth, K, embed_dim, hidden, gru_layers,
    head_width, res_scale, channels, cond_globals, vocab) is read from the stats
    file, so a checkpoint can never be silently deserialised into the wrong skeleton.
    """
    s = dict(np.load(stats_path, allow_pickle=True))

    required = ("g_mu", "g_sd", "nu", "log_a", "n_comp", "width", "depth",
                "embed_dim", "hidden", "gru_layers", "head_width", "vocab")
    missing = [k for k in required if k not in s]
    if missing:
        raise KeyError(
            f"{stats_path} is missing {missing}. Point this at a residual stats file "
            f"(*_stats.npz written by residual.py), not an mlp_mixture one."
        )

    n_comp = int(s["n_comp"])
    n_globals = int(np.asarray(s["g_mu"]).shape[0])
    channels = tuple(str(c) for c in np.atleast_1d(s["channels"]))
    cond_globals = bool(s["cond_globals"])

    kb, ks = jax.random.split(jax.random.PRNGKey(0))
    base = RgMLPNet(
        key=kb, flory_nu=float(s["nu"]), flory_log_a=float(s["log_a"]),
        width=int(s["width"]), depth=int(s["depth"]),
        n_globals=n_globals, n_comp=n_comp,
    )
    seq = SeqResidual(
        key=ks, vocab=int(s["vocab"]), n_comp=n_comp,
        embed_dim=int(s["embed_dim"]), hidden=int(s["hidden"]),
        layers=int(s["gru_layers"]), head_width=int(s["head_width"]),
        dropout=float(s.get("dropout", 0.1)),            # irrelevant: inference=True
        res_scale=float(s["res_scale"]), channels=channels, cond_globals=cond_globals,
    )
    model = eqx.tree_deserialise_leaves(weights_path, ResidualRgNet(base=base, seq=seq))

    print(f"[model] loaded {weights_path}")
    print(f"        base MLP width={int(s['width'])} depth={int(s['depth'])} K={n_comp} | "
          f"Flory: log Rg = {float(s['log_a']):.4f} + {float(s['nu']):.4f} log N")
    print(f"        BiGRU embed={int(s['embed_dim'])} hidden={int(s['hidden'])} "
          f"layers={int(s['gru_layers'])} | res_scale={float(s['res_scale']):.2f} | "
          f"corrects [{','.join(channels)}] | cond_globals={cond_globals} | "
          f"trained pad-L={int(s['max_len'])}")
    tau = float(s.get("res_thresh", 0.0))
    if tau > 0:
        print(f"        group soft-threshold tau={tau:.4f} ACTIVE — proteins whose "
              f"correction is worth less than the toll get an exact-zero residual")
    return model, s


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Featurisation (globals + logL, exactly as in training, plus tokens)
# ══════════════════════════════════════════════════════════════════════════════
def featurize(seqs: Sequence[str], stats, names=None):
    """seqs -> (glob, logL, tok, mask). Padding is to the batch max, which is
    safe because the recurrence and both pooling ops are masked."""
    names = names or [f"seq{i}" for i in range(len(seqs))]
    seqs = [clean_sequence(s, n) for s, n in zip(seqs, names)]

    B = len(seqs)
    n_globals = int(np.asarray(stats["g_mu"]).shape[0])
    train_L = int(stats["max_len"])

    glob = np.zeros((B, n_globals), np.float32)
    logL = np.zeros(B, np.float32)
    L = max(len(s) for s in seqs)

    over = [(n, len(s)) for n, s in zip(names, seqs) if len(s) > train_L]
    if over:
        print(f"[warn] {len(over)} sequence(s) longer than the training pad length "
              f"({train_L}): {over[:3]}{'...' if len(over) > 3 else ''}. The BiGRU will "
              f"still run, but these are length-extrapolation and the Flory anchor is "
              f"being used outside its fitted range.", file=sys.stderr)

    tok = np.zeros((B, L), np.int32)
    for i, s in enumerate(seqs):
        ids = np.array([AA_TO_ID.get(a, 0) for a in s], np.int32)
        glob[i] = sequence_globals(ids)
        logL[i] = np.log(len(ids) + 1e-6)
        tok[i, :len(ids)] = ids + 1                      # +1: PAD_ID == 0 stays free

    # ── Standardize global physical features using the training fold stats ──
    glob = (glob - stats["g_mu"]) / stats["g_sd"]
    mask = tok != PAD_ID

    return (jnp.array(glob, jnp.float32), jnp.array(logL, jnp.float32),
            jnp.array(tok), jnp.array(mask))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Forward passes
# ══════════════════════════════════════════════════════════════════════════════
@eqx.filter_jit
def _forward_residual(model, glob, logL, tok, mask, thresh=0.0):
    key = jax.random.PRNGKey(0)                          # unused when inference=True
    keys = jax.random.split(key, glob.shape[0])
    f = lambda g, l, t, m, k: model(g, l, t, m, key=k, inference=True, thresh=thresh)
    return jax.vmap(f)(glob, logL, tok, mask, keys)


@eqx.filter_jit
def _forward_base(model, glob, logL):
    """The frozen MLP alone — i.e. the model with the sequence branch switched off."""
    return jax.vmap(model.base)(glob, logL)


def predict(model, stats, seqs, names=None):
    """-> dict of (B, ...) arrays for BOTH heads.

    Residual model:  logit_pi, mu, sigma, mean, std, skew, exkurt
    Frozen baseline: base_*  (same fields)
    Correction:      res (B, 3K) raw head-space delta, res_rms, d_mean/d_std/...
    """
    glob, logL, tok, mask = featurize(seqs, stats, names)
    clean = [clean_sequence(s) for s in seqs]

    # The training-time group soft-threshold MUST be reapplied at inference, or a
    # protein the model deliberately left alone silently gets its correction back.
    thresh = float(stats.get("res_thresh", 0.0))
    lp, mu, sg, res = _forward_residual(model, glob, logL, tok, mask, thresh)
    m1, sd, sk, ku = jax.vmap(mixture_moments)(lp, mu, sg)

    blp, bmu, bsg = _forward_base(model, glob, logL)
    bm1, bsd, bsk, bku = jax.vmap(mixture_moments)(blp, bmu, bsg)

    p = dict(
        logit_pi=np.asarray(lp), mu=np.asarray(mu), sigma=np.asarray(sg),
        mean=np.asarray(m1), std=np.asarray(sd), skew=np.asarray(sk), exkurt=np.asarray(ku),
        base_logit_pi=np.asarray(blp), base_mu=np.asarray(bmu), base_sigma=np.asarray(bsg),
        base_mean=np.asarray(bm1), base_std=np.asarray(bsd),
        base_skew=np.asarray(bsk), base_exkurt=np.asarray(bku),
        res=np.asarray(res),
        res_rms=np.sqrt(np.mean(np.asarray(res) ** 2, axis=1)),
        length=np.array([len(s) for s in clean]),
    )
    for k in ("mean", "std", "skew", "exkurt"):
        p["d_" + k] = p[k] - p["base_" + k]
    return p


def apply_units(p, sc):
    """Rescale Rg-valued fields. mu lives in log-space, so it shifts by log(sc)."""
    if sc == 1.0:
        return p
    for k in ("mean", "std", "base_mean", "base_std", "d_mean", "d_std"):
        p[k] = p[k] * sc
    for k in ("mu", "base_mu"):
        p[k] = p[k] + np.log(sc)
    return p


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Plots
# ══════════════════════════════════════════════════════════════════════════════
def plot_single(p, i, unit, out_path, show_components=False, ref_rg=None, title=None):
    """Residual density in green, frozen-MLP baseline dashed underneath.
    The gap between the two curves IS the sequence contribution."""
    lp, mu, sg = p["logit_pi"][i], p["mu"][i], p["sigma"][i]
    blp, bmu, bsg = p["base_logit_pi"][i], p["base_mu"][i], p["base_sigma"][i]
    m1, sd = float(p["mean"][i]), float(p["std"][i])
    sk, ku = float(p["skew"][i]), float(p["exkurt"][i])
    K = len(mu)

    # grid must cover both mixtures
    grid = density_grid(np.concatenate([mu, bmu]), np.concatenate([sg, bsg]))
    g = np.asarray(grid)
    dens = eval_density(lp, mu, sg, grid)
    bdens = eval_density(blp, bmu, bsg, grid)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.fill_between(g, dens, alpha=0.22, color="#2ca02c")
    ax.plot(g, dens, color="#2ca02c", lw=2.3, label="MLP + BiGRU residual", zorder=4)
    # ax.plot(g, bdens, color="#ff7f0e", lw=1.8, ls="--",
    #         label="Frozen MLP baseline", zorder=3)

    if show_components:
        comps, pi = component_densities(lp, mu, sg, grid)
        order = np.argsort(-pi)
        for rank, k in enumerate(order):
            if pi[k] < 1e-3:
                continue
            ax.plot(g, comps[k], lw=1.0, alpha=0.75,
                    color=plt.cm.viridis(rank / max(K - 1, 1)),
                    label=f"comp {k} ($\\pi$={pi[k]:.2f})" if rank < 4 else None, zorder=2)

    gauss = np.exp(-0.5 * ((g - m1) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    ax.plot(g, gauss, ":", color="#7f7f7f", lw=1.3,
            label="Gaussian (same $\\mu,\\sigma$)", zorder=1)

    ax.axvline(m1, color="#d62728", ls=":", lw=1.5, zorder=5)
    if ref_rg is not None:
        ax.axvline(ref_rg, color="#1f77b4", ls="-.", lw=1.5, zorder=5,
                   label=f"Reference ({ref_rg:.2f} {unit})")

    ax.set_xlim(g.min(), 20)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"Radius of gyration $R_g$ [{unit}]")
    ax.set_ylabel("Probability density")
    ax.set_title(title or f"Predicted $P(R_g)$ — MLP + BiGRU residual, K={K}")

    txt = (f"L        = {int(p['length'][i])}\n"
           f"mean     = {m1:.2f} {unit}   ({p['d_mean'][i]:+.2f} vs base)\n"
           f"std      = {sd:.2f} {unit}   ({p['d_std'][i]:+.2f} vs base)\n"
           f"skew     = {sk:+.3f}  ({p['d_skew'][i]:+.3f})\n"
           f"ex-kurt  = {ku:+.3f}  ({p['d_exkurt'][i]:+.3f})\n"
           f"res rms  = {p['res_rms'][i]:.4f}")
    ax.text(0.975, 0.94, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#ddd", alpha=0.92))
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] density -> {out_path}")


def plot_compare(names, p, ref, unit, out_path, annotate=None):
    """Baseline vs residual on the same references: does the +0.02 nats show up in Rg?"""
    from scipy.stats import pearsonr

    ok = np.isfinite(ref)
    r_, b_ = ref[ok], p["base_mean"][ok]
    m_ = p["mean"][ok]

    def stats_of(pred):
        return dict(r=pearsonr(r_, pred)[0],
                    r2=1.0 - np.sum((r_ - pred) ** 2) / np.sum((r_ - r_.mean()) ** 2),
                    mae=float(np.mean(np.abs(r_ - pred))),
                    rmse=float(np.sqrt(np.mean((r_ - pred) ** 2))),
                    bias=float(np.mean(pred - r_)))

    sb, sm = stats_of(b_), stats_of(m_)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2))
    lo = min(r_.min(), b_.min(), m_.min())
    hi = max(r_.max(), b_.max(), m_.max())
    span = hi - lo
    lo, hi = lo - 0.08 * span, hi + 0.08 * span

    for ax, pred, st, ttl, col in (
            (axes[0], b_, sb, "Frozen MLP baseline", "#ff7f0e"),
            (axes[1], m_, sm, "MLP + BiGRU residual", "#2ca02c")):
        ax.plot([lo, hi], [lo, hi], "--", color="#7f7f7f", alpha=0.75, lw=1.3, zorder=1)
        ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1],
                        color="#7f7f7f", alpha=0.08, zorder=0, label="±10%")
        ax.scatter(r_, pred, color=col, s=46, alpha=0.85,
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", "box")
        ax.set_xlabel(f"Reference $R_g$ [{unit}]")
        ax.set_ylabel(f"Predicted mean $R_g$ [{unit}]")
        ax.set_title(ttl)
        ax.text(0.04, 0.96,
                f"r     = {st['r']:.3f}\nR^2   = {st['r2']:.3f}\n"
                f"MAE   = {st['mae']:.3f} {unit}\nRMSE  = {st['rmse']:.3f} {unit}\n"
                f"bias  = {st['bias']:+.3f} {unit}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#ddd", alpha=0.92))
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)

    # arrows on the right panel: where the residual moved each point
    axes[1].quiver(r_, b_, np.zeros_like(r_), m_ - b_, angles="xy", scale_units="xy",
                   scale=1.0, width=0.004, color="#555", alpha=0.45, zorder=2)

    d_mae = sb["mae"] - sm["mae"]
    fig.suptitle(f"Sequence correction: MAE {sb['mae']:.3f} -> {sm['mae']:.3f} {unit} "
                 f"({d_mae:+.3f}, {'better' if d_mae > 0 else 'worse'})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] compare -> {out_path}")
    return sb, sm


def plot_shuffle(names, real, shuf, unit, out_path):
    """Per-sequence: true-order prediction vs the spread over composition-preserving shuffles."""
    n = len(names)
    order = np.argsort(real["mean"])
    fig, ax = plt.subplots(figsize=(7.8, 0.34 * n + 2.0))

    for row, i in enumerate(order):
        s = shuf["mean"][i]
        ax.plot([s.min(), s.max()], [row, row], color="#bbb", lw=2.0, solid_capstyle="round",
                zorder=1)
        ax.scatter(s, np.full_like(s, row), s=8, color="#777", alpha=0.55, zorder=2)
        ax.scatter([real["mean"][i]], [row], s=42, color="#2ca02c", zorder=3,
                   edgecolors="white", linewidths=0.6)
        ax.scatter([real["base_mean"][i]], [row], s=42, marker="s", color="#ff7f0e", zorder=3,
                   edgecolors="white", linewidths=0.6)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([names[i] for i in order], fontsize=7.5)
    ax.set_xlabel(f"Predicted mean $R_g$ [{unit}]")
    ax.set_title("Order sensitivity: true sequence (green) vs shuffled residues (grey)\n"
                 "orange square = frozen MLP baseline (order-blind by construction)",
                 fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] shuffle -> {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CSV
# ══════════════════════════════════════════════════════════════════════════════
def write_csv(path, names, p, ref, unit):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "length",
                    f"pred_mean_{unit}", f"pred_std_{unit}", "pred_skew", "pred_exkurt",
                    f"base_mean_{unit}", f"base_std_{unit}", "base_skew", "base_exkurt",
                    f"d_mean_{unit}", "d_skew", "d_exkurt", "res_rms",
                    f"ref_rg_{unit}", "abs_err", "base_abs_err", "rel_err"])
        for i, nm in enumerate(names):
            r = ref[i]
            ae = abs(p["mean"][i] - r) if np.isfinite(r) else np.nan
            bae = abs(p["base_mean"][i] - r) if np.isfinite(r) else np.nan
            re_ = ae / r if (np.isfinite(r) and r != 0) else np.nan
            fmt = lambda x, d=4: (f"{x:.{d}f}" if np.isfinite(x) else "")
            w.writerow([nm, int(p["length"][i]),
                        fmt(p["mean"][i]), fmt(p["std"][i]), fmt(p["skew"][i]), fmt(p["exkurt"][i]),
                        fmt(p["base_mean"][i]), fmt(p["base_std"][i]),
                        fmt(p["base_skew"][i]), fmt(p["base_exkurt"][i]),
                        fmt(p["d_mean"][i]), fmt(p["d_skew"][i]), fmt(p["d_exkurt"][i]),
                        fmt(p["res_rms"][i]),
                        fmt(r), fmt(ae), fmt(bae), fmt(re_)])
    print(f"[csv]  -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Self-test
# ══════════════════════════════════════════════════════════════════════════════
TEST_SEQS = ["MDVFMKGLSKAKEGVVAAAEK",
             "MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQ",
             "GSHMASMTGGQQMGRDLYDDDDKDRWGSELEKAMVALIDVFHQYSGREGDKHKLKKSELKEL"]


def self_test(model, stats):
    """1) batched == single (this also tests PAD invariance: the short sequence is
          padded to 63 in the batch and to its own length when run alone)
       2) the base head inside the residual model still reproduces plain RgMLPNet
       3) masked-out channels really receive zero correction"""
    batched = predict(model, stats, TEST_SEQS)
    worst = 0.0
    for i, s in enumerate(TEST_SEQS):
        one = predict(model, stats, [s])
        for k in ("mean", "std", "skew", "exkurt", "base_mean", "res_rms"):
            worst = max(worst, abs(float(one[k][0]) - float(batched[k][i])))
    print(f"[self-test] batched vs single (pad-invariance): max |Δ| = {worst:.3e}")
    if worst > 1e-4:
        raise AssertionError(f"Padding leaks into the prediction: {worst:.3e}")

    K = int(stats["n_comp"])
    channels = tuple(str(c) for c in np.atleast_1d(stats["channels"]))
    res = batched["res"]
    for j, name in enumerate(("pi", "mu", "sigma")):
        blk = np.abs(res[:, j * K:(j + 1) * K]).max()
        if name not in channels and blk > 0:
            raise AssertionError(f"channel '{name}' is masked off but got |res|={blk:.2e}")
    print(f"[self-test] channel mask honoured: corrections live only in "
          f"[{','.join(channels)}]")

    rs = float(np.max(np.abs(res)))
    cap = float(stats["res_scale"])
    print(f"[self-test] max |raw correction| = {rs:.4f} (cap res_scale = {cap:.2f}, "
          f"{100*rs/cap:.1f}% of headroom used)")
    if rs > cap + 1e-5:
        raise AssertionError("residual exceeded its tanh cap — impossible; check the loader")

    d = np.abs(batched["mean"] - batched["base_mean"])
    print(f"[self-test] sequence shifts mean Rg by {d.mean():.3f} on average "
          f"(max {d.max():.3f})")
    rn = np.linalg.norm(res, axis=1)
    print(f"[self-test] ||r|| per protein = {np.array2string(rn, precision=3)} "
          f"| exactly zero (left alone): {int((rn == 0).sum())}/{len(rn)}")
    print("[self-test] OK")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Shuffle (order-sensitivity) diagnostic
# ══════════════════════════════════════════════════════════════════════════════
def shuffle_test(model, stats, names, seqs, n_shuffle, seed, unit, out_path=None):
    """Composition-preserving permutations. The global descriptors are NOT all
    permutation-invariant (SCD, charge_asym, s_q depend on order), so the baseline
    moves a little too — the diagnostic is how much MORE the residual moves.

    If the residual's spread over shuffles is ~0, the BiGRU learned only composition
    and is duplicating the MLP's inputs. If it is large, it is genuinely reading order.
    """
    rng = np.random.default_rng(seed)
    real = predict(model, stats, seqs, names)

    B = len(seqs)
    sh_mean = np.zeros((B, n_shuffle), float)
    sh_base = np.zeros((B, n_shuffle), float)
    sh_res = np.zeros((B, n_shuffle), float)
    for t in range(n_shuffle):
        perm_seqs = ["".join(rng.permutation(list(clean_sequence(s)))) for s in seqs]
        q = predict(model, stats, perm_seqs, names)
        sh_mean[:, t] = q["mean"]
        sh_base[:, t] = q["base_mean"]
        sh_res[:, t] = q["res_rms"]

    print(f"\n{'name':<20s} {'L':>4s} {'true':>8s} {'shuf mean':>10s} {'shuf sd':>8s} "
          f"{'z':>6s} | {'base sd':>8s} {'res rms':>8s}")
    print("─" * 88)
    zs = []
    for i, nm in enumerate(names):
        sd = sh_mean[i].std()
        z = (real["mean"][i] - sh_mean[i].mean()) / (sd + 1e-9)
        zs.append(z)
        print(f"{nm:<20s} {int(real['length'][i]):4d} {real['mean'][i]:8.3f} "
              f"{sh_mean[i].mean():10.3f} {sd:8.3f} {z:+6.2f} | "
              f"{sh_base[i].std():8.3f} {real['res_rms'][i]:8.4f}")

    tot_res_sd = float(np.mean(sh_mean.std(axis=1)))
    tot_base_sd = float(np.mean(sh_base.std(axis=1)))
    print(f"\n[order] mean spread over shuffles: residual model {tot_res_sd:.4f} {unit} | "
          f"baseline {tot_base_sd:.4f} {unit} (the baseline moves only through the "
          f"order-dependent globals: SCD, charge_asym, s_q)")
    extra = tot_res_sd - tot_base_sd
    print(f"[order] sequence-branch order sensitivity = {extra:+.4f} {unit}")
    if tot_res_sd < 1e-3:
        print("[order] VERDICT: the BiGRU output is essentially permutation-invariant — "
              "it is re-encoding composition, not reading order. The gain in NLL is not "
              "coming from sequence patterns.")
    elif extra <= 0.25 * tot_base_sd:
        print("[order] VERDICT: order sensitivity is comparable to what the globals already "
              "provide. Weak evidence for genuine sequence-pattern learning.")
    else:
        print("[order] VERDICT: the residual responds to residue order well beyond the "
              "order-dependent globals — the BiGRU is reading patterns the descriptors miss.")
    print(f"[order] mean |z| of the true sequence within its own shuffle ensemble: "
          f"{np.mean(np.abs(zs)):.2f} (|z|>2 => the real ordering is an outlier, which is "
          f"what you want)")

    if out_path:
        plot_shuffle(names, real, dict(mean=sh_mean), unit, out_path)
    return real, sh_mean


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["single", "benchmark", "compare", "shuffle", "selftest", "export"])
    ap.add_argument("--weights", default="rg_resid_model.eqx")
    ap.add_argument("--stats", default="rg_resid_stats.npz")
    ap.add_argument("--unit", default="A", help="label for the Rg axis (A, nm, ...)")
    ap.add_argument("--unit-scale", type=float, default=1.0)
    # single
    ap.add_argument("--seq", default=None, help="one sequence (single mode)")
    ap.add_argument("--ref", type=float, default=None, help="reference Rg to overlay")
    # benchmark / compare / shuffle
    ap.add_argument("--yaml", default="config/sequences.yaml")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--ridgeline", default=None)
    ap.add_argument("--annotate", nargs="*", default=None)
    ap.add_argument("--n-shuffle", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    model, stats = load_model(a.weights, a.stats)
    sc = a.unit_scale

    if a.mode == "selftest":
        self_test(model, stats)
        return

    if a.mode == "single":
        if not a.seq:
            ap.error("single mode requires --seq")
        p = apply_units(predict(model, stats, [a.seq]), sc)
        m1, sd = float(p["mean"][0]), float(p["std"][0])

        print(f"\n[prediction]  L = {int(p['length'][0])}")
        print(f"{'':14s}{'residual':>10s} {'baseline':>10s} {'Δ':>9s}")
        for k, lab, f in (("mean", f"mean Rg [{a.unit}]", "8.3f"),
                          ("std", f"std Rg [{a.unit}]", "8.3f"),
                          ("skew", "skew", "+8.3f"),
                          ("exkurt", "ex-kurt", "+8.3f")):
            print(f"  {lab:<12s}{float(p[k][0]):{f}}  {float(p['base_'+k][0]):{f}}  "
                  f"{float(p['d_'+k][0]):+8.3f}")
        print(f"  {'CV':<12s}{sd/m1:8.3f}  {float(p['base_std'][0])/float(p['base_mean'][0]):8.3f}")
        pi = np.asarray(jax.nn.softmax(jnp.asarray(p["logit_pi"][0])))
        print(f"  live components (pi>0.01): {int((pi > 0.01).sum())}/{len(pi)}  "
              f"pi = {np.array2string(pi, precision=3, suppress_small=True)}")
        print(f"  residual rms (raw head space): {p['res_rms'][0]:.4f} "
              f"of a {float(stats['res_scale']):.2f} cap")
        plot_single(p, 0, a.unit, a.out or "rg_density.png", ref_rg=a.ref)
        return

    names, seqs, ref = load_yaml(a.yaml)
    print(f"[yaml] {len(names)} sequences compiled from {a.yaml}")

    if a.mode == "shuffle":
        shuffle_test(model, stats, names, seqs, a.n_shuffle, a.seed, a.unit,
                     out_path=a.out or "rg_shuffle.png")
        return

    p = apply_units(predict(model, stats, seqs, names), sc)
    ref = ref * 1.0

    print(f"\n{'name':<20s} {'L':>4s} {'pred':>9s} {'base':>9s} {'Δmean':>7s} {'std':>7s} "
          f"{'skew':>7s} {'exkurt':>7s} {'ref':>9s} {'err':>8s} {'base err':>9s}")
    print("─" * 108)
    for i, nm in enumerate(names):
        r = ref[i]
        err = f"{p['mean'][i] - r:+8.3f}" if np.isfinite(r) else "       —"
        berr = f"{p['base_mean'][i] - r:+9.3f}" if np.isfinite(r) else "        —"
        rs = f"{r:9.3f}" if np.isfinite(r) else "        —"
        print(f"{nm:<20s} {int(p['length'][i]):4d} {p['mean'][i]:9.3f} "
              f"{p['base_mean'][i]:9.3f} {p['d_mean'][i]:+7.3f} {p['std'][i]:7.3f} "
              f"{p['skew'][i]:+7.3f} {p['exkurt'][i]:+7.3f} {rs} {err} {berr}")

    print(f"\n[residual] mean |Δmean| = {np.mean(np.abs(p['d_mean'])):.3f} {a.unit} | "
          f"mean |Δskew| = {np.mean(np.abs(p['d_skew'])):.4f} | "
          f"mean |Δex-kurt| = {np.mean(np.abs(p['d_exkurt'])):.4f} | "
          f"mean res_rms = {p['res_rms'].mean():.4f}")

    check_units(p["mean"], ref, a.unit)

    if a.mode == "export":
        print("\n[export] Tracing single-example graph -> ONNX (Netron-ready)...")
        from jax2onnx import to_onnx

        n_globals = int(np.asarray(stats["g_mu"]).shape[0])
        train_L   = int(stats["max_len"])
        thresh    = float(stats.get("res_thresh", 0.0))

        # Single-example forward. NO vmap: jax2onnx's batched-split rule breaks on
        # vmap + lax.scan + GRUCell (the GRU splits its gate weights internally).
        # key/inference/thresh are baked in as constants; with inference=True the
        # key is unused and gets DCE'd out of the graph.
        def onnx_inference_wrapper(glob, logL, tok, mask):
            return model(glob, logL, tok, mask,
                         key=jax.random.PRNGKey(0), inference=True, thresh=thresh)

        # Unbatched specs — one protein. logL is a scalar here (shape ()).
        input_specs = [
            jax.ShapeDtypeStruct((n_globals,), jnp.float32),  # glob
            jax.ShapeDtypeStruct((),           jnp.float32),  # logL (scalar)
            jax.ShapeDtypeStruct((train_L,),   jnp.int32),    # tok
            jax.ShapeDtypeStruct((train_L,),   jnp.bool_),    # mask
        ]

        out_onnx = a.out if (a.out and a.out.endswith(".onnx")) else "rg_resid_model.onnx"

        try:
            to_onnx(
                onnx_inference_wrapper, input_specs,
                model_name="rg_residual",
                input_names=["glob", "logL", "tok", "mask"],
                output_names=["logit_pi", "mu", "sigma", "res"],
                return_mode="file", output_path=out_onnx,
            )
        except Exception as e:
            # Almost always an unsupported primitive in the real head (softplus,
            # a stray gather pattern, etc.). This logs every primitive the tracer
            # hits so you can see exactly which one has no plugin.
            print(f"[export] tracer failed ({type(e).__name__}: {e})")
            print("[export] re-running with a primitive log -> primitives.txt")
            to_onnx(onnx_inference_wrapper, input_specs,
                    record_primitive_calls_file="primitives.txt", return_mode="proto")
            raise

        print(f"[export] Architecture serialized to: {out_onnx}")
        print(f"[export] View it:  netron {out_onnx}   (or drag it into https://netron.app)")
        return

    if a.csv:
        write_csv(a.csv, names, p, ref, a.unit)
    if a.ridgeline:
        plot_ridgeline(names, p, a.unit, a.ridgeline)

    ok = np.isfinite(ref)
    if ok.sum() < 2:
        print("\n[skip] Fewer than 2 reference values in the YAML — no metrics to compute.")
        return

    if a.mode == "compare":
        sb, sm = plot_compare(names, p, ref, a.unit, a.out or "baseline_vs_residual.png",
                              annotate=a.annotate)
        print(f"\n[metrics] baseline : r={sb['r']:.3f}  R2={sb['r2']:.3f}  "
              f"MAE={sb['mae']:.3f}  RMSE={sb['rmse']:.3f}  bias={sb['bias']:+.3f} {a.unit}")
        print(f"[metrics] residual : r={sm['r']:.3f}  R2={sm['r2']:.3f}  "
              f"MAE={sm['mae']:.3f}  RMSE={sm['rmse']:.3f}  bias={sm['bias']:+.3f} {a.unit}")
        print(f"[metrics] Δ        : r={sm['r']-sb['r']:+.3f}  R2={sm['r2']-sb['r2']:+.3f}  "
              f"MAE={sm['mae']-sb['mae']:+.3f}  RMSE={sm['rmse']-sb['rmse']:+.3f} "
              f"(negative MAE/RMSE = the sequence helped)")
        return

    m = plot_benchmark([n for n, k in zip(names, ok) if k],
                       p["mean"][ok], ref[ok], a.unit,
                       a.out or "model_vs_reference.png",
                       annotate=a.annotate, pred_err=p["std"][ok])
    print(f"\n[metrics] n={m['n']}  r={m['pearson']:.3f}  rho={m['spearman']:.3f}  "
          f"R2={m['r2']:.3f}  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
          f"bias={m['bias']:+.3f} {a.unit}")


if __name__ == "__main__":
    main()