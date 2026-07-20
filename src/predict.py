"""
predict_idp.py — forward pass + plot for ONE IDP sequence.

Updated to align with the single-head distogram network architecture.
"""
from __future__ import annotations
from functools import partial
from typing import List

import argparse
import numpy as np
import jax, jax.numpy as jnp
from jax.scipy.linalg import toeplitz
import equinox as eqx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optax

# ══════════════════════════════════════════════════════════════════════════════
# 0.  Amino-acid table
# ══════════════════════════════════════════════════════════════════════════════
AA_PARAMS = {
    "-": dict(mass=0.0,   charge=0.0,  sigma=0.0,   HPS1=0.0,   HPS2=0.0),
    "A": dict(mass=71.08, charge=0.0,  sigma=0.504, HPS1=0.730, HPS2=0.003),
    "R": dict(mass=156.20,charge=1.0,  sigma=0.656, HPS1=0.000, HPS2=0.723),
    "N": dict(mass=114.10,charge=0.0,  sigma=0.568, HPS1=0.432, HPS2=0.160),
    "D": dict(mass=115.10,charge=-1.0, sigma=0.558, HPS1=0.378, HPS2=0.002),
    "C": dict(mass=103.10,charge=0.0,  sigma=0.548, HPS1=0.595, HPS2=0.400),
    "Q": dict(mass=128.10,charge=0.0,  sigma=0.602, HPS1=0.514, HPS2=0.468),
    "E": dict(mass=129.10,charge=-1.0, sigma=0.592, HPS1=0.459, HPS2=0.022),
    "G": dict(mass=57.05, charge=0.0,  sigma=0.450, HPS1=0.649, HPS2=0.784),
    "H": dict(mass=137.10,charge=0.5,  sigma=0.608, HPS1=0.514, HPS2=0.487),
    "I": dict(mass=113.20,charge=0.0,  sigma=0.618, HPS1=0.973, HPS2=0.687),
    "L": dict(mass=113.20,charge=0.0,  sigma=0.618, HPS1=0.973, HPS2=0.335),
    "K": dict(mass=128.20,charge=1.0,  sigma=0.636, HPS1=0.514, HPS2=0.095),
    "M": dict(mass=131.20,charge=0.0,  sigma=0.618, HPS1=0.838, HPS2=0.993),
    "F": dict(mass=147.20,charge=0.0,  sigma=0.636, HPS1=1.000, HPS2=0.871),
    "P": dict(mass=97.12, charge=0.0,  sigma=0.556, HPS1=1.000, HPS2=0.471),
    "S": dict(mass=87.08, charge=0.0,  sigma=0.518, HPS1=0.595, HPS2=0.487),
    "T": dict(mass=101.10,charge=0.0,  sigma=0.562, HPS1=0.676, HPS2=0.274),
    "W": dict(mass=186.20,charge=0.0,  sigma=0.678, HPS1=0.946, HPS2=0.753),
    "Y": dict(mass=163.20,charge=0.0,  sigma=0.646, HPS1=0.865, HPS2=0.984),
    "V": dict(mass=99.07, charge=0.0,  sigma=0.586, HPS1=0.892, HPS2=0.428),
}
AA_TO_ID = {aa: i for i, aa in enumerate(AA_PARAMS.keys())}
N_AA = len(AA_PARAMS)

MASS, CHARGE, SIGMA, HPS1, HPS2 = 0, 1, 2, 3, 4

param_matrix = jnp.array([[v["mass"], v["charge"], v["sigma"], v["HPS1"], v["HPS2"]]
                            for v in AA_PARAMS.values()])
is_charged_mask = jnp.array([v["charge"] != 0 for v in AA_PARAMS.values()])

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Physics per-window features
# ══════════════════════════════════════════════════════════════════════════════
def s_all(window_ids):
    counts = jnp.bincount(window_ids, length=N_AA)
    real = counts[1:]
    N = jnp.sum(real)
    p = real / jnp.where(N == 0, 1.0, N)
    safe = jnp.where(p > 0, p, 1.0)
    return jnp.where(N == 0, 0.0, -jnp.sum(p * jnp.log2(safe)))

def s_q(window_ids):
    counts = jnp.bincount(window_ids, length=N_AA)
    cc = counts * is_charged_mask
    N = jnp.sum(cc)
    p = cc / jnp.where(N == 0, 1.0, N)
    safe = jnp.where(p > 0, p, 1.0)
    return jnp.where(N == 0, 0.0, -jnp.sum(p * jnp.log2(safe)))

def scd(window):
    k = len(window)
    qij = jnp.outer(window, window)
    dist = toeplitz(jnp.arange(k)) ** 0.5
    return jnp.sum(jnp.tril(qij * dist, k=-1)) / k

def shd(window):
    k = len(window)
    lam = window[:, None] + window[None, :]
    dist = toeplitz(jnp.arange(k))
    safe = jnp.where(dist == 0, 1.0, dist)
    return jnp.sum(jnp.tril(lam * (1.0 / safe), k=-1)) / k

def net_charge(window):        return jnp.sum(window)
def fcr(window):               return jnp.sum(jnp.abs(window) > 0) / len(window)
def mean_hydro(window):        return jnp.mean(window)

def charge_asymmetry(window):
    n = len(window)
    pos = jnp.sum(window > 0) / n
    neg = jnp.sum(window < 0) / n
    tot = pos + neg
    return jnp.where(tot == 0, 0.0, (pos - neg) ** 2 / tot)

def sbcs(window_lambda):
    N = len(window_lambda)
    is_high = window_lambda > 0.5
    C = jnp.cumsum(is_high)
    C_diff = C[None, :] - C[:, None]
    idx = jnp.arange(N)
    dmat = idx[None, :] - idx[:, None]
    valid = (C_diff == 1) & jnp.outer(is_high, is_high) & (dmat > 0)
    safe = jnp.where(dmat == 0, 1.0, dmat)
    inv = jnp.where(valid, 1.0 / safe, 0.0)
    return jnp.mean(window_lambda) * jnp.sum(inv)

def conv(x, f, kernel_size):
    idx = jnp.arange(len(x) - kernel_size + 1)[:, None] + jnp.arange(kernel_size)
    return jax.vmap(f)(x[idx])

batch_conv = jax.vmap(conv, in_axes=(0, None, None))

@partial(jax.jit, static_argnames=["kernel_size"])
def process_batch_features(ids, features, mask, kernel_size):
    conv_mask = jax.lax.reduce_window(mask, 0.0, jax.lax.max,
                                      (1, kernel_size), (1, 1), "VALID")
    b_scd  = batch_conv(features[:, :, CHARGE], scd,             kernel_size)
    b_nc   = batch_conv(features[:, :, CHARGE], net_charge,      kernel_size)
    b_hps  = batch_conv(features[:, :, HPS1],  mean_hydro,       kernel_size)
    b_fcr  = batch_conv(features[:, :, CHARGE], fcr,             kernel_size)
    b_shd  = batch_conv(features[:, :, HPS1],  shd,              kernel_size)
    b_asym = batch_conv(features[:, :, CHARGE], charge_asymmetry, kernel_size)
    b_sbcs = batch_conv(features[:, :, HPS1],  sbcs,             kernel_size)
    b_s_q  = batch_conv(ids, s_q,   kernel_size)
    b_s_all= batch_conv(ids, s_all, kernel_size)
    stacked = jnp.stack([b_scd, b_nc, b_hps, b_fcr, b_shd,
                         b_asym, b_sbcs, b_s_q, b_s_all], axis=-1)
    return stacked * conv_mask[:, :, None], conv_mask
N_PHYS = 9

def _scd_np(q):
    k = len(q)
    if k < 2: return 0.0
    d = np.sqrt(np.abs(np.subtract.outer(np.arange(k), np.arange(k))))
    return float(np.tril(np.outer(q, q) * d, -1).sum() / k)

def _shd_np(lam):
    k = len(lam)
    if k < 2: return 0.0
    d = np.abs(np.subtract.outer(np.arange(k), np.arange(k))).astype(float)
    safe = np.where(d == 0, 1.0, d)
    return float(np.tril((lam[:, None] + lam[None, :]) / safe, -1).sum() / k)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Model Architecture Block
# ══════════════════════════════════════════════════════════════════════════════
def align_valid_to_full(phys_bnf, kernel_size, L):
    pad_total = L - phys_bnf.shape[1]
    left = (kernel_size - 1) // 2
    right = pad_total - left
    padded = jnp.pad(phys_bnf, ((0, 0), (left, right), (0, 0)))
    return padded.transpose(0, 2, 1)

def build_physics_stack(ids, features, mask, kernel_sizes, L):
    stacks = []
    for k in kernel_sizes:
        feats_k, _ = process_batch_features(ids, features, mask, k)
        stacks.append(align_valid_to_full(feats_k, k, L))
    return jnp.stack(stacks, axis=1)


class FiLM(eqx.Module):
    net: eqx.nn.MLP
    dim: int = eqx.field(static=True)
    def __init__(self, cond_dim, feat_dim, *, key, width=64):
        self.net = eqx.nn.MLP(cond_dim, 2 * feat_dim, width_size=width,
                              depth=2, activation=jax.nn.gelu, key=key)
        self.dim = feat_dim
    def __call__(self, h, c):
        gb = self.net(c)
        gamma, beta = gb[:self.dim], gb[self.dim:]
        return (1.0 + gamma) * h + beta


class ScaleEncoder(eqx.Module):
    norm: eqx.nn.LayerNorm
    conv: eqx.nn.Conv1d
    fuse: eqx.nn.Conv1d
    # attn_query: jax.Array  <── You can comment out or remove this line

    def __init__(self, in_ch, n_phys, hidden_ch, kernel_size, *, key):
        _, kc, kf, kq = jax.random.split(key, 4)
        assert kernel_size % 2 == 1, "use odd kernels so 'same' padding aligns"
        pad = (kernel_size - 1) // 2
        self.norm = eqx.nn.LayerNorm(in_ch)
        self.conv = eqx.nn.Conv1d(in_ch, hidden_ch, kernel_size, padding=pad, key=kc)
        self.fuse = eqx.nn.Conv1d(hidden_ch + n_phys, hidden_ch, 1, key=kf)
        # self.attn_query = jax.random.normal(kq, (hidden_ch,)) * 0.02  <── Remove

    def __call__(self, x, phys, mask):
        x = jax.vmap(self.norm, in_axes=1, out_axes=1)(x)
        h = jax.nn.gelu(self.conv(x))
        h = jnp.concatenate([h, phys], axis=0)
        h = jax.nn.gelu(self.fuse(h))  # Shape: (hidden_ch, L)
        
        # ══════════════════════════════════════════════════════════════════════
        # NEW: Masked Global Average Pooling over the active sequence length
        # ══════════════════════════════════════════════════════════════════════
        denom = jnp.sum(mask) + 1e-6
        return jnp.sum(h * mask[None, :], axis=-1) / denom

class RgDistogramNet(eqx.Module):
    embedding: eqx.nn.Embedding
    scales: List[ScaleEncoder]
    scale_query: jax.Array
    cond_mlp: eqx.nn.MLP
    film_global: FiLM
    trunk: eqx.nn.MLP
    film_disto: FiLM
    disto_mlp: eqx.nn.MLP
    kernel_sizes: tuple = eqx.field(static=True)

    def __init__(self, *, key, n_aa=N_AA, embed_dim=16, n_raw=5, n_phys=N_PHYS,
                 hidden_ch=128, kernel_sizes=(3, 5, 9, 15, 21, 33),
                 n_globals=2, cond_dim=32, n_bins=450):
        
        keys = jax.random.split(key, 10)
        in_ch = embed_dim + n_raw
        self.embedding = eqx.nn.Embedding(n_aa, embed_dim, key=keys[0])
        self.scales = [ScaleEncoder(in_ch, n_phys, hidden_ch, k, key=keys[1 + i])
                       for i, k in enumerate(kernel_sizes)]
        self.scale_query = jax.random.normal(keys[5], (hidden_ch,)) * 0.02
        self.kernel_sizes = tuple(kernel_sizes)
        self.cond_mlp = eqx.nn.MLP(n_globals + 1, cond_dim, width_size=64,
                                   depth=2, activation=jax.nn.gelu, key=keys[6])
        self.film_global = FiLM(cond_dim, hidden_ch, key=keys[7])
        self.trunk = eqx.nn.MLP(hidden_ch, hidden_ch, width_size=128,
                                depth=2, activation=jax.nn.gelu, key=keys[8])
        
        kt = jax.random.split(keys[9], 2)
        self.film_disto = FiLM(cond_dim, hidden_ch, key=kt[0])
        self.disto_mlp = eqx.nn.MLP(hidden_ch, n_bins, width_size=256,
                                    depth=2, activation=jax.nn.gelu, key=kt[1])

    def __call__(self, ids, raw_params, phys_stack, globals_, mask):
        emb = jax.vmap(self.embedding)(ids)
        feat = jnp.concatenate([emb, raw_params], axis=-1) * mask[:, None]
        x = feat.T
        tokens = jnp.stack([enc(x, phys_stack[s], mask)
                            for s, enc in enumerate(self.scales)], axis=0)
        scale_w = jax.nn.softmax(tokens @ self.scale_query)
        g = scale_w @ tokens
        logL = jnp.log(jnp.sum(mask) + 1e-6)
        c = self.cond_mlp(jnp.concatenate([globals_, logL[None]]))
        g = self.film_global(g, c)
        z = self.trunk(g)
        zd = self.film_disto(z, c)
        logits = self.disto_mlp(zd)
        return logits

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Moment Calculation
# ══════════════════════════════════════════════════════════════════════════════
def _pred_moments(p, grid):
    m1 = p @ grid
    d = grid[None, :] - m1[:, None]
    var = jnp.sum(p * d ** 2, axis=-1)
    sd = jnp.sqrt(jnp.clip(var, 1e-8))
    skew = jnp.sum(p * d ** 3, axis=-1) / (sd ** 3 + 1e-8)
    kurt = jnp.sum(p * d ** 4, axis=-1) / (sd ** 4 + 1e-8)
    return m1, sd, skew, kurt

# ══════════════════════════════════════════════════════════════════════════════
# 4.  Inference Pipelines
# ══════════════════════════════════════════════════════════════════════════════
KDE_MAX = 150.0
N_BINS  = 450
GRID    = jnp.linspace(0.0, KDE_MAX, N_BINS)
STANDARD_AA = set("ARNDCQEGHILKMFPSTWYV")
KERNELS = (3, 5, 9, 15, 21, 33)

def load_model(path):
    skeleton = RgDistogramNet(key=jax.random.PRNGKey(0),
                              kernel_sizes=KERNELS, n_bins=N_BINS, n_globals=2)
    return eqx.tree_deserialise_leaves(path, skeleton)

def read_sequence(args):
    if args.fasta:
        seq, started = [], False
        for line in open(args.fasta):
            if line.startswith(">"):
                if started:
                    break
                started = True
                continue
            seq.append(line.strip())
        raw = "".join(seq)
    else:
        raw = args.seq
    return "".join(raw.split()).upper()

def featurize(seq, stats):
    L = len(seq)
    if L < max(KERNELS):
        raise SystemExit(f"sequence length {L} < largest kernel {max(KERNELS)}")
    bad = sorted(set(seq) - STANDARD_AA)
    if bad:
        print(f"[warn] non-standard residues mapped to padding id 0: {bad}")

    pm   = np.asarray(param_matrix)
    ids  = np.array([[AA_TO_ID.get(a, 0) for a in seq]], np.int32)
    mask = np.ones((1, L), np.float32)
    raw  = pm[ids]
    q, lam = pm[ids[0], CHARGE], pm[ids[0], HPS1]
    glob = np.array([[_scd_np(q), _shd_np(lam)]], np.float32)
    phys = np.asarray(build_physics_stack(jnp.array(ids), jnp.array(raw),
                                          jnp.array(mask), KERNELS, L))

    raw  = ((raw  - stats["r_mu"]) / stats["r_sd"]) * mask[..., None]
    phys = ((phys - stats["p_mu"][None, :, :, None])
                  / stats["p_sd"][None, :, :, None]) * mask[:, None, None, :]
    glob = (glob - stats["g_mu"]) / stats["g_sd"]

    return (jnp.array(ids), jnp.array(raw, jnp.float32),
            jnp.array(phys, jnp.float32), jnp.array(glob, jnp.float32),
            jnp.array(mask))

def predict(model, feats):
    logits = jax.vmap(model)(*feats)
    p = jax.nn.softmax(logits, axis=-1)[0]
    m, sd, sk, ku = (float(x[0]) for x in _pred_moments(p[None], GRID))
    return np.asarray(p), dict(
        mean=m, std=sd, skew=sk,
        kurtosis_raw=ku,
        excess_kurtosis=ku - 3.0,
    )

def plot(p, info, name, out):
    grid = np.asarray(GRID)
    cdf = np.cumsum(p)
    lo = grid[max(0, np.searchsorted(cdf, 0.001) - 1)]
    hi = grid[min(len(grid) - 1, np.searchsorted(cdf, 0.999) + 1)]
    pad = 0.15 * (hi - lo + 1e-6)

    gauss = np.exp(-0.5 * ((grid - info["mean"]) / info["std"]) ** 2)
    gauss *= p.max() / gauss.max()

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.fill_between(grid, p, alpha=0.30, color="#4C72B0")
    ax.plot(grid, p, color="#4C72B0", lw=1.9, label="predicted P(Rg)")
    ax.plot(grid, gauss, "--", color="#999", lw=1.3, label="Gaussian (same μ, σ)")
    ax.axvline(info["mean"], color="#C44E52", lw=1.1, ls=":")
    ax.set_xlim(max(0.0, lo - pad), hi + pad)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Rg  (same length units as the training grid)")
    ax.set_ylabel("probability mass per bin")
    ax.set_title(f"Predicted Rg distribution — {name}")
    txt = (f"μ        = {info['mean']:.2f}\n"
           f"σ        = {info['std']:.2f}\n"
           f"skew     = {info['skew']:.3f}\n"
           f"kurt(raw)= {info['kurtosis_raw']:.3f}")
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[plot] saved -> {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq")
    ap.add_argument("--fasta")
    ap.add_argument("--weights", default="rg_distogram_model.eqx")
    ap.add_argument("--stats",   default="rg_standardisation_stats.npz")
    ap.add_argument("--name",    default="my IDP")
    ap.add_argument("--out",     default="rg_prediction.png")
    a = ap.parse_args()

    seq = read_sequence(a) if (a.fasta or a.seq) else 'GPGMRGKVKWFDSKKGYGFITKDEGGDVFVHWSAIEMEGFKTLKEGQVVEFEIQEGKKGGQAAHVKV'
    stats = dict(np.load(a.stats))
    model = load_model(a.weights)
    p, info = predict(model, featurize(seq, stats))

    print(f"[{a.name}]  L = {len(seq)}   sum(P) = {p.sum():.4f}")
    for k in ("mean", "std", "skew", "kurtosis_raw", "excess_kurtosis"):
        print(f"    {k:16s} {info[k]:+.4f}")
    plot(p, info, a.name, a.out)

if __name__ == "__main__":
    main()