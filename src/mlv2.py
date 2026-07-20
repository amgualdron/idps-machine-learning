from __future__ import annotations
import argparse
from functools import partial
from typing import List

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import toeplitz
from jax.scipy.special import logsumexp
import equinox as eqx
import optax

# ══════════════════════════════════════════════════════════════════════════════
#  Sequence -> P(Rg) via a LOG-NORMAL MIXTURE density, trained with continuous
#  per-frame NLL.  No bins, no KDE, no bandwidth, no grid.
#
#        p(r) = (1/r) * SUM_k  pi_k * N( log r ; mu_k , sigma_k )
#
#  KURTOSIS CONVENTION: extraction stores scipy.stats.kurtosis(fisher=True) =
#  EXCESS kurtosis (Gaussian -> 0). mixture_moments() returns EXCESS too.
#
#  ── STABILITY (v2) ──────────────────────────────────────────────────────────
#  v1 NaN'd at epoch 4. Two chained bugs, both fixed here:
#
#   (1) DEAD-COMPONENT DRIFT.  A component whose pi_k -> 0 receives essentially no
#       gradient from the NLL (logsumexp simply ignores it), so its mu_k / sigma_k
#       wander freely. The DENSITY doesn't care -- the NLL stayed healthy the whole
#       time. But the moment quadrature evaluates at r = exp(mu + sigma*GH_X) with
#       |GH_X|max ~= 14.9, so a runaway sigma ~ 3 gives exp(mu + 44) ~ 1e21 and
#       d**4 ~ 1e84 -> inf in float32 (max ~3.4e38). Hence the observed
#       exkurt: 1.6 -> 1.8 -> 56.8 -> inf.
#       FIX: BOUND the parameters. sigma via sigmoid into [SIGMA_MIN, SIGMA_MAX];
#       delta_mu via tanh into [-DELTA_MAX, DELTA_MAX]. Physically justified --
#       sigma is a width in LOG-Rg units, so sigma=0.6 already means Rg spans a
#       factor of ~3 within +-1 sigma. Real IDPs sit around 0.10-0.25.
#
#   (2) 0.0 * inf = nan.  v1 computed the moment terms unconditionally and then
#       multiplied by the lambdas, so lam_kurt=0.0 did NOT disable the term: the
#       instant it overflowed to inf the loss became NaN, and one step of NaN
#       gradients wiped every parameter.
#       FIX: a genuine static branch (lambdas are Python floats -> static under
#       filter_jit, so the branch compiles away), plus optax.zero_nans() as a
#       backstop. Note clip_by_global_norm alone does NOT save you: a NaN gradient
#       gives a NaN norm gives NaN updates.
# ══════════════════════════════════════════════════════════════════════════════

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

def net_charge(window):  return jnp.sum(window)
def fcr(window):         return jnp.sum(jnp.abs(window) > 0) / len(window)
def mean_hydro(window):  return jnp.mean(window)

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
    b_scd   = batch_conv(features[:, :, CHARGE], scd,              kernel_size)
    b_nc    = batch_conv(features[:, :, CHARGE], net_charge,       kernel_size)
    b_hps   = batch_conv(features[:, :, HPS1],  mean_hydro,        kernel_size)
    b_fcr   = batch_conv(features[:, :, CHARGE], fcr,              kernel_size)
    b_shd   = batch_conv(features[:, :, HPS1],  shd,               kernel_size)
    b_asym  = batch_conv(features[:, :, CHARGE], charge_asymmetry, kernel_size)
    b_sbcs  = batch_conv(features[:, :, HPS1],  sbcs,              kernel_size)
    b_s_q   = batch_conv(ids, s_q,   kernel_size)
    b_s_all = batch_conv(ids, s_all, kernel_size)
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
# 2.  Encoder  (unchanged: embedding -> multi-scale conv + physics -> masked
#     mean-pool -> scale attention -> FiLM -> trunk.  Only the HEAD is new.)
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

    def __init__(self, in_ch, n_phys, hidden_ch, kernel_size, *, key):
        kc, kf = jax.random.split(key, 2)
        assert kernel_size % 2 == 1, "use odd kernels so 'same' padding aligns"
        pad = (kernel_size - 1) // 2
        self.norm = eqx.nn.LayerNorm(in_ch)
        self.conv = eqx.nn.Conv1d(in_ch, hidden_ch, kernel_size, padding=pad, key=kc)
        self.fuse = eqx.nn.Conv1d(hidden_ch + n_phys, hidden_ch, 1, key=kf)

    def __call__(self, x, phys, mask):
        x = jax.vmap(self.norm, in_axes=1, out_axes=1)(x)
        h = jax.nn.gelu(self.conv(x))
        h = jnp.concatenate([h, phys], axis=0)
        h = jax.nn.gelu(self.fuse(h))                       # (hidden_ch, L)
        denom = jnp.sum(mask) + 1e-6
        return jnp.sum(h * mask[None, :], axis=-1) / denom   # masked mean-pool


# ── mixture head hyper-params ────────────────────────────────────────────────
# sigma is a width in LOG-Rg units: sigma=0.6 means Rg spans a factor ~3 within
# +-1 sigma. Real IDPs sit around 0.10-0.25, so 0.60 is a generous ceiling. The
# ceiling is what stops a dead component drifting off and exploding the 4th moment.
SIGMA_MIN    = 0.01
SIGMA_MAX    = 0.60
DELTA_MAX    = 1.00     # component means stay within e^{+-1} of the Flory anchor
SIGMA_INIT   = 0.15
DELTA_SPREAD = 0.25


class RgMixtureNet(eqx.Module):
    """Sequence -> parameters of a log-normal mixture over Rg."""
    embedding: eqx.nn.Embedding
    scales: List[ScaleEncoder]
    scale_query: jax.Array
    cond_mlp: eqx.nn.MLP
    film_global: FiLM
    trunk: eqx.nn.MLP
    film_head: FiLM
    head: eqx.nn.MLP                       # -> 3K  (logit_pi | raw_delta | raw_sigma)
    kernel_sizes: tuple = eqx.field(static=True)
    n_comp: int = eqx.field(static=True)
    flory_nu: float = eqx.field(static=True)
    flory_log_a: float = eqx.field(static=True)

    def __init__(self, *, key, flory_nu, flory_log_a,
                 n_aa=N_AA, embed_dim=16, n_raw=5, n_phys=N_PHYS,
                 hidden_ch=128, kernel_sizes=(3, 5, 9, 15, 21, 33),
                 n_globals=2, cond_dim=32, n_comp=6):
        n_scales = len(kernel_sizes)
        keys = jax.random.split(key, n_scales + 6)
        in_ch = embed_dim + n_raw

        self.embedding = eqx.nn.Embedding(n_aa, embed_dim, key=keys[0])
        self.scales = [ScaleEncoder(in_ch, n_phys, hidden_ch, k, key=keys[1 + i])
                       for i, k in enumerate(kernel_sizes)]
        i0 = 1 + n_scales
        self.scale_query = jax.random.normal(keys[i0], (hidden_ch,)) * 0.02
        self.kernel_sizes = tuple(kernel_sizes)
        self.n_comp = n_comp
        self.flory_nu = float(flory_nu)
        self.flory_log_a = float(flory_log_a)

        self.cond_mlp = eqx.nn.MLP(n_globals + 1, cond_dim, width_size=64,
                                   depth=2, activation=jax.nn.gelu, key=keys[i0 + 1])
        self.film_global = FiLM(cond_dim, hidden_ch, key=keys[i0 + 2])
        self.trunk = eqx.nn.MLP(hidden_ch, hidden_ch, width_size=128,
                                depth=2, activation=jax.nn.gelu, key=keys[i0 + 3])
        self.film_head = FiLM(cond_dim, hidden_ch, key=keys[i0 + 4])

        head = eqx.nn.MLP(hidden_ch, 3 * n_comp, width_size=256,
                          depth=2, activation=jax.nn.gelu, key=keys[i0 + 5])

        # ── Init matters a LOT for mixtures. Start with: components evenly spread
        #    around the Flory prediction, equal weights, sensible width. Final
        #    weights shrunk 100x so the BIAS dominates at step 0 -> epoch 1 already
        #    sits on the physics prior instead of flailing.
        #    Biases live in PRE-ACTIVATION space, so invert the bounded transforms:
        #       sigma = SIGMA_MIN + (SIGMA_MAX-SIGMA_MIN)*sigmoid(raw)  -> logit
        #       delta = DELTA_MAX * tanh(raw)                           -> arctanh
        s0 = (SIGMA_INIT - SIGMA_MIN) / (SIGMA_MAX - SIGMA_MIN)
        raw_sigma_0 = float(np.log(s0 / (1.0 - s0)))
        if n_comp == 1:
            delta_targets = jnp.zeros(1)                    # single component: centred
        else:
            delta_targets = jnp.linspace(-DELTA_SPREAD, DELTA_SPREAD, n_comp)
        raw_delta_0 = jnp.arctanh(delta_targets / DELTA_MAX)

        bias_init = jnp.concatenate([
            jnp.zeros(n_comp),                              # logit_pi -> equal weights
            raw_delta_0,                                    # raw delta_mu
            jnp.full((n_comp,), raw_sigma_0),               # raw sigma
        ])
        head = eqx.tree_at(lambda m: m.layers[-1].bias, head, bias_init)
        head = eqx.tree_at(lambda m: m.layers[-1].weight, head,
                           head.layers[-1].weight * 0.01)
        self.head = head

    def __call__(self, ids, raw_params, phys_stack, globals_, mask):
        emb = jax.vmap(self.embedding)(ids)
        feat = jnp.concatenate([emb, raw_params], axis=-1) * mask[:, None]
        x = feat.T
        tokens = jnp.stack([enc(x, phys_stack[s], mask)
                            for s, enc in enumerate(self.scales)], axis=0)
        scale_w = jax.nn.softmax(tokens @ self.scale_query)
        g = scale_w @ tokens

        n_res = jnp.sum(mask)
        logL = jnp.log(n_res + 1e-6)
        c = self.cond_mlp(jnp.concatenate([globals_, logL[None]]))

        g = self.film_global(g, c)
        z = self.trunk(g)
        zh = self.film_head(z, c)
        out = self.head(zh)                                  # (3K,)

        K = self.n_comp
        logit_pi = out[:K]

        # ── BOUNDED parameterisations: the fix for dead-component drift -> inf ──
        delta_mu = DELTA_MAX * jnp.tanh(out[K:2 * K])
        sigma    = SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * jax.nn.sigmoid(out[2 * K:])

        # Flory anchor: log Rg ~ log_a + nu*log N, so the net predicts RESIDUALS.
        anchor = self.flory_log_a + self.flory_nu * logL
        mu = anchor + delta_mu

        return logit_pi, mu, sigma


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Log-normal mixture: density, NLL, moments
# ══════════════════════════════════════════════════════════════════════════════
_LOG_2PI = float(np.log(2.0 * np.pi))

# Gauss-Hermite nodes/weights: E_{t~N(0,1)}[f(t)] = sum_j w_j f(x_j).
# Used for CENTRAL moments. Computing them from raw moments
# (E[R^4] - 4 m1 E[R^3] + ...) is catastrophic cancellation in float32 and would
# destroy exactly the skew/kurtosis you care about. Quadrature forms the
# differences (r - m1) directly, so there is nothing to cancel.
_GH_N = 64
_gh_x, _gh_w = np.polynomial.hermite.hermgauss(_GH_N)
GH_X = jnp.asarray(_gh_x * np.sqrt(2.0), dtype=jnp.float32)     # (J,), |x|max ~ 14.9
GH_W = jnp.asarray(_gh_w / np.sqrt(np.pi), dtype=jnp.float32)   # (J,), sums to 1


def mixture_log_prob(logit_pi, mu, sigma, r):
    """log p(r) for ONE protein at frames r (F,) -> (F,).

    p(r) = (1/r) * sum_k pi_k * N(log r; mu_k, sigma_k)
    """
    r = jnp.clip(r, 1e-8)
    y = jnp.log(r)                                            # (F,)
    log_pi = jax.nn.log_softmax(logit_pi)                     # (K,)
    z = (y[:, None] - mu[None, :]) / sigma[None, :]           # (F,K)
    log_comp = -0.5 * z ** 2 - jnp.log(sigma)[None, :] - 0.5 * _LOG_2PI
    log_py = logsumexp(log_pi[None, :] + log_comp, axis=-1)   # (F,)
    return log_py - y                                         # Jacobian |dy/dr| = 1/r


def mixture_moments(logit_pi, mu, sigma):
    """Moments of the log-normal mixture, ONE protein, by Gauss-Hermite quadrature.

    Returns (mean, std, skew, EXCESS kurtosis) -- conventions matching
    scipy.stats.skew / scipy.stats.kurtosis(fisher=True) on the raw frames.

    With sigma <= SIGMA_MAX = 0.6 the worst-case support point is
    exp(mu + 0.6*14.9) ~ 4e5, so d**4 ~ 3e22 -- comfortably inside float32.
    """
    pi = jax.nn.softmax(logit_pi)                              # (K,)
    r = jnp.exp(mu[:, None] + sigma[:, None] * GH_X[None, :])  # (K,J)
    w = pi[:, None] * GH_W[None, :]                            # (K,J), sums to 1

    m1  = jnp.sum(w * r)
    d   = r - m1
    var = jnp.sum(w * d ** 2)
    sd  = jnp.sqrt(jnp.clip(var, 1e-12))
    m3  = jnp.sum(w * d ** 3)
    m4  = jnp.sum(w * d ** 4)
    skew   = m3 / (sd ** 3 + 1e-12)
    exkurt = m4 / (sd ** 4 + 1e-12) - 3.0
    return m1, sd, skew, exkurt


batch_log_prob = jax.vmap(mixture_log_prob)     # (B,K)x3, (B,F) -> (B,F)
batch_moments  = jax.vmap(mixture_moments)      # (B,K)x3 -> 4 x (B,)


def mixture_density_on_grid(logit_pi, mu, sigma, grid):
    """For plotting only. Smooth by construction -- no bins, no bandwidth."""
    return jnp.exp(mixture_log_prob(logit_pi, mu, sigma, grid))


def mixture_weight_entropy(logit_pi):
    """Entropy of the mixture weights (nats). Max = log K (all components alive)."""
    log_pi = jax.nn.log_softmax(logit_pi)
    return -jnp.sum(jnp.exp(log_pi) * log_pi)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Loss
# ══════════════════════════════════════════════════════════════════════════════
def loss_fn(model, ids, raw, phys, glob, mask, frames,
            t_mean, t_std, t_skew, t_exkurt,
            lam_mean=0.0, lam_std=0.0, lam_skew=0.0, lam_kurt=0.0, lam_ent=0.0):
    """Continuous per-frame NLL (+ optional moment penalties, + optional entropy bonus).

    frames: (B, F). Every protein contributes exactly F frames, so the plain mean
    over (B,F) IS equal-per-protein weighting -- no trajectory-length bias.

    Defaults are PURE NLL. The lambdas are Python floats and therefore STATIC under
    filter_jit, so `if lam_*:` genuinely compiles the branch away. This matters: v1
    multiplied by lam=0.0 instead of branching, and 0.0*inf = nan, so a "disabled"
    penalty could still NaN the entire model.
    """
    logit_pi, mu, sigma = jax.vmap(model)(ids, raw, phys, glob, mask)

    nll = -jnp.mean(batch_log_prob(logit_pi, mu, sigma, frames))

    loss = nll
    zero = jnp.zeros(())
    aux = dict(nll=nll, l_mean=zero, l_std=zero, l_skew=zero, l_kurt=zero, ent=zero)

    # ── moment penalties: computed ONLY if actually requested ─────────────────
    if lam_mean or lam_std or lam_skew or lam_kurt:
        pm, psd, psk, pku = batch_moments(logit_pi, mu, sigma)
        l_mean = jnp.mean(jnp.abs(pm  - t_mean) / (t_mean + 1e-6))  # relative (has units)
        l_std  = jnp.mean(jnp.abs(psd - t_std)  / (t_std  + 1e-6))  # relative (has units)
        l_skew = jnp.mean(jnp.abs(psk - t_skew))                    # absolute (dimensionless)
        l_kurt = jnp.mean(jnp.abs(pku - t_exkurt))                  # absolute (dimensionless)
        loss = (loss + lam_mean * l_mean + lam_std * l_std
                     + lam_skew * l_skew + lam_kurt * l_kurt)
        aux.update(l_mean=l_mean, l_std=l_std, l_skew=l_skew, l_kurt=l_kurt)

    # ── entropy bonus: discourages collapse onto 1-2 live components ──────────
    if lam_ent:
        ent = jnp.mean(jax.vmap(mixture_weight_entropy)(logit_pi))
        loss = loss - lam_ent * ent          # MAXIMISE entropy -> subtract
        aux.update(ent=ent)

    return loss, aux


@eqx.filter_jit
def train_step(model, opt_state, optimizer, batch, lams):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, *batch, **lams)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux


@eqx.filter_jit
def eval_batch(model, ids, raw, phys, glob, mask, frames):
    logit_pi, mu, sigma = jax.vmap(model)(ids, raw, phys, glob, mask)
    nll = -jnp.mean(batch_log_prob(logit_pi, mu, sigma, frames))
    pm, psd, psk, pku = batch_moments(logit_pi, mu, sigma)
    ent = jnp.mean(jax.vmap(mixture_weight_entropy)(logit_pi))
    return nll, pm, psd, psk, pku, ent


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Data prep
# ══════════════════════════════════════════════════════════════════════════════
def clean_dataframe(df, frame_threshold=10_000, max_len=400, min_len=20):
    if "frame_row" not in df.columns:
        raise KeyError(
            "parquet has no 'frame_row' column -- this is the OLD dataset. "
            "Re-run build_idp_dataset.py to produce the parquet + idp_rg_frames.npy pair."
        )
    df = df[df["total_frames"] >= frame_threshold].copy()
    logN, logRg = np.log(df["seq_length"]), np.log(df["calculated_rg_mean"])
    nu, log_a = np.polyfit(logN, logRg, 1)
    resid = logRg - (log_a + nu * logN)
    df = df[np.abs(resid) <= 3 * resid.std()]
    df = df[(df["seq_length"] <= max_len) & (df["seq_length"] >= min_len)]
    print(f"[clean] {len(df)} rows kept | Flory nu={nu:.3f}")
    return df.reset_index(drop=True)


def tokenize(seq, L):
    tid = [AA_TO_ID.get(a, 0) for a in seq.upper()]
    m = [1.0] * len(tid)
    pad = L - len(tid)
    return np.array(tid + [0] * pad, np.int32), np.array(m + [0.0] * pad, np.float32)


def prepare_arrays(df, frames_path, kernel_sizes=(3, 5, 9, 15, 21, 33), phys_chunk=256):
    N = len(df)
    L = int(df["sequence"].str.len().max())
    print(f"[prep] N={N}  L(max)={L}  scales={kernel_sizes}")

    ids  = np.zeros((N, L), np.int32)
    mask = np.zeros((N, L), np.float32)
    glob = np.zeros((N, 2), np.float32)
    pm_np = np.asarray(param_matrix)
    for i, seq in enumerate(df["sequence"]):
        ids[i], mask[i] = tokenize(seq, L)
        ell = int(mask[i].sum())
        q   = pm_np[ids[i, :ell], CHARGE]
        lam = pm_np[ids[i, :ell], HPS1]
        glob[i] = (_scd_np(q), _shd_np(lam))

    raw = pm_np[ids]

    S = len(kernel_sizes)
    phys = np.zeros((N, S, N_PHYS, L), np.float32)
    for a in range(0, N, phys_chunk):
        b = min(a + phys_chunk, N)
        ps = build_physics_stack(jnp.array(ids[a:b]), jnp.array(raw[a:b]),
                                 jnp.array(mask[a:b]), kernel_sizes, L)
        phys[a:b] = np.asarray(ps)

    # ── Frames: memmap the (P, K) store; keep only the rows this df kept. ─────
    if isinstance(frames_path, np.ndarray):          # smoke test injects an array
        frames_mm = frames_path
    else:
        frames_mm = np.load(frames_path, mmap_mode="r")
    frame_rows = df["frame_row"].to_numpy().astype(np.int64)
    K_STORE = frames_mm.shape[1]
    print(f"[prep] frame store {frames_mm.shape} {frames_mm.dtype} | "
          f"{K_STORE} frames/protein available")

    # ── Targets: per-protein scalars from the FULL trajectory. ───────────────
    # calculated_rg_kurtosis is scipy's default => EXCESS kurtosis.
    t_mean   = df["calculated_rg_mean"].to_numpy().astype(np.float32)
    t_std    = df["calculated_rg_std"].to_numpy().astype(np.float32)
    t_skew   = df["calculated_rg_skew"].to_numpy().astype(np.float32)
    t_exkurt = df["calculated_rg_kurtosis"].to_numpy().astype(np.float32)

    # ── Flory anchor, fit on the kept set; centres the mixture means. ────────
    logN  = np.log(df["seq_length"].to_numpy().astype(np.float64))
    logRg = np.log(t_mean.astype(np.float64))
    nu, log_a = np.polyfit(logN, logRg, 1)
    print(f"[prep] Flory anchor: log Rg = {log_a:.4f} + {nu:.4f} * log N")

    return dict(ids=ids, raw=raw, phys=phys, glob=glob, mask=mask,
                frames=frames_mm, frame_rows=frame_rows, K_store=K_STORE,
                mean=t_mean, std=t_std, skew=t_skew, exkurt=t_exkurt,
                L=L, nu=float(nu), log_a=float(log_a),
                kernel_sizes=tuple(kernel_sizes))


def standardize(data, tr_idx):
    m = data["mask"][tr_idx].astype(bool)
    r = data["raw"][tr_idx]
    r_mu = r[m].mean(0);  r_sd = r[m].std(0) + 1e-6
    data["raw"] = ((data["raw"] - r_mu) / r_sd) * data["mask"][..., None]

    p = data["phys"][tr_idx]
    mp = m[:, None, None, :]
    p_mu = (p * mp).sum((0, 3)) / mp.sum((0, 3))
    p_var = (((p - p_mu[None, :, :, None]) ** 2) * mp).sum((0, 3)) / mp.sum((0, 3))
    p_sd = np.sqrt(p_var) + 1e-6
    data["phys"] = ((data["phys"] - p_mu[None, :, :, None]) / p_sd[None, :, :, None])
    data["phys"] *= data["mask"][:, None, None, :]

    g = data["glob"][tr_idx]
    g_mu = g.mean(0); g_sd = g.std(0) + 1e-6
    data["glob"] = (data["glob"] - g_mu) / g_sd

    data["_stats"] = dict(r_mu=r_mu, r_sd=r_sd, p_mu=p_mu, p_sd=p_sd,
                          g_mu=g_mu, g_sd=g_sd,
                          nu=np.float32(data["nu"]), log_a=np.float32(data["log_a"]),
                          kernel_sizes=np.array(data["kernel_sizes"], np.int32))
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Batching
# ══════════════════════════════════════════════════════════════════════════════
def draw_frames(d, idx, n_frames, rng):
    """Random subsample of n_frames per protein -> (B, n_frames).

    Unbiased: a uniform draw from the stored frames is a draw from the same
    empirical P(Rg). Resampled every step, so over training the model still sees
    the full richness of each trajectory.
    """
    rows = d["frame_rows"][idx]                       # (B,)
    block = np.asarray(d["frames"][rows])             # (B, K_store) -- memmap read
    cols = rng.integers(0, d["K_store"], size=(len(idx), n_frames))
    return np.take_along_axis(block, cols, axis=1).astype(np.float32)


def make_batch(d, idx, n_frames, rng):
    j = jnp.array
    return (j(d["ids"][idx]),  j(d["raw"][idx]),  j(d["phys"][idx]),
            j(d["glob"][idx]), j(d["mask"][idx]),
            j(draw_frames(d, idx, n_frames, rng)),
            j(d["mean"][idx]), j(d["std"][idx]),
            j(d["skew"][idx]), j(d["exkurt"][idx]))


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Training loop
# ══════════════════════════════════════════════════════════════════════════════
def run_training(d, *, epochs=40, batch_size=64, n_frames=1024, n_comp=6,
                 lr=5e-4, val_frac=0.15, seed=42,
                 lam_mean=0.0, lam_std=0.0, lam_skew=0.0, lam_kurt=0.0, lam_ent=0.0,
                 out_prefix="rg_mixture"):
    N = len(d["ids"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = int(N * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    d = standardize(d, tr_idx)

    model = RgMixtureNet(key=jax.random.PRNGKey(seed),
                         flory_nu=d["nu"], flory_log_a=d["log_a"],
                         kernel_sizes=d["kernel_sizes"],   # threaded from data ->
                         n_comp=n_comp, n_globals=2)       # no silent scale mismatch

    # zero_nans FIRST. clip_by_global_norm alone does NOT save you: a NaN gradient
    # gives a NaN norm gives NaN updates. This is the backstop that stops one bad
    # step from permanently poisoning every weight.
    opt = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr, weight_decay=1e-4),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    lams = dict(lam_mean=lam_mean, lam_std=lam_std, lam_skew=lam_skew,
                lam_kurt=lam_kurt, lam_ent=lam_ent)
    use_moments = any([lam_mean, lam_std, lam_skew, lam_kurt])

    val_rng = np.random.default_rng(0)          # fixed val frames -> comparable NLL
    val_frames = jnp.array(draw_frames(d, val_idx, 4096, val_rng))
    val_static = (jnp.array(d["ids"][val_idx]),  jnp.array(d["raw"][val_idx]),
                  jnp.array(d["phys"][val_idx]), jnp.array(d["glob"][val_idx]),
                  jnp.array(d["mask"][val_idx]))

    mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - b)))
    steps = max(1, len(tr_idx) // batch_size)
    best = np.inf
    max_ent = float(np.log(n_comp)) if n_comp > 1 else 0.0

    for ep in range(epochs):
        rng.shuffle(tr_idx)
        acc = dict(loss=0.0, nll=0.0, l_mean=0.0, l_std=0.0, l_skew=0.0, l_kurt=0.0)
        for s in range(steps):
            bidx = tr_idx[s * batch_size:(s + 1) * batch_size]
            batch = make_batch(d, bidx, n_frames, rng)
            model, opt_state, l, aux = train_step(model, opt_state, opt, batch, lams)
            acc["loss"] += float(l)
            for k in ("nll", "l_mean", "l_std", "l_skew", "l_kurt"):
                acc[k] += float(aux[k])

        vnll, pm, psd, psk, pku, vent = eval_batch(model, *val_static, val_frames)
        vnll = float(vnll)

        print(f"ep {ep+1:02d}/{epochs} | train loss {acc['loss']/steps:8.4f} "
              f"(nll {acc['nll']/steps:7.4f}) | val nll {vnll:7.4f} | "
              f"val MAE  mean {mae(pm,  d['mean'][val_idx]):.3f}  "
              f"std {mae(psd, d['std'][val_idx]):.3f}  "
              f"skew {mae(psk, d['skew'][val_idx]):.3f}  "
              f"exkurt {mae(pku, d['exkurt'][val_idx]):.3f}  "
              f"| pi-ent {float(vent):.2f}/{max_ent:.2f}")
        if use_moments:
            print(f"          components | mean {acc['l_mean']/steps:.4f}  "
                  f"std {acc['l_std']/steps:.4f}  skew {acc['l_skew']/steps:.4f}  "
                  f"kurt {acc['l_kurt']/steps:.4f}")

        if not np.isfinite(vnll):
            print("  !! non-finite val NLL -- stopping. Check the frames are strictly "
                  "positive and finite:  np.isfinite(fr).all() and (fr > 0).all()")
            break

        if vnll < best:
            best = vnll
            eqx.tree_serialise_leaves(f"{out_prefix}_model.eqx", model)

    np.savez(f"{out_prefix}_stats.npz", n_comp=np.int32(n_comp), **d["_stats"])
    print(f"saved: {out_prefix}_model.eqx (best val nll {best:.4f}) + {out_prefix}_stats.npz")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Smoke test  (synthetic FRAMES, not a synthetic KDE)
# ══════════════════════════════════════════════════════════════════════════════
def synthetic_data(P=40, K_store=5000, seed=0):
    import pandas as pd
    from scipy.stats import skew as _sk, kurtosis as _ku
    rng = np.random.default_rng(seed)
    aas = list("ARNDCQEGHILKMFPSTWYV")
    rows, frames = [], []
    for i in range(P):
        ell = int(rng.integers(25, 120))
        seq = "".join(rng.choice(aas, ell))
        mu_log = np.log(0.55 * ell ** 0.55)          # crude Flory
        sig_log = rng.uniform(0.10, 0.25)
        fr = np.exp(rng.normal(mu_log, sig_log, size=K_store)).astype(np.float32)
        rows.append(dict(folder_id=i, sequence=seq, seq_length=ell,
                         total_frames=K_store, frame_row=i,
                         calculated_rg_mean=float(fr.mean()),
                         calculated_rg_std=float(fr.std()),
                         calculated_rg_skew=float(_sk(fr)),
                         calculated_rg_kurtosis=float(_ku(fr))))   # excess
        frames.append(fr)
    return pd.DataFrame(rows), np.stack(frames)


def smoke():
    print("── SMOKE TEST ──")
    df, frames = synthetic_data()
    df = clean_dataframe(df, frame_threshold=1000, min_len=10)
    KERNELS = (3, 5, 9, 15, 21, 33)
    d = prepare_arrays(df, frames, kernel_sizes=KERNELS, phys_chunk=16)
    for k in ("ids", "raw", "phys", "glob", "mask"):
        print(f"   {k:5s} {np.asarray(d[k]).shape}")
    # exercise the moment + entropy branches too, so they can't silently rot
    run_training(d, epochs=3, batch_size=8, n_frames=512, n_comp=4, val_frac=0.25,
                 lam_skew=0.05, lam_kurt=0.05, lam_ent=1e-3,
                 out_prefix="smoke_rg_mixture")
    print("── SMOKE OK ──")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/idp_ml_dataset.parquet")
    ap.add_argument("--frames",  default="data/idp_rg_frames.npy")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_frames", type=int, default=1024,
                    help="frames drawn per protein per step (resampled each step)")
    ap.add_argument("--n_comp", type=int, default=6, help="mixture components K")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lam_mean", type=float, default=0.0)
    ap.add_argument("--lam_std",  type=float, default=0.0)
    ap.add_argument("--lam_skew", type=float, default=0.0)
    ap.add_argument("--lam_kurt", type=float, default=0.0)
    ap.add_argument("--lam_ent",  type=float, default=0.0,
                    help="entropy bonus on mixture weights; try 1e-3 if pi collapses")
    a = ap.parse_args()

    if a.smoke:
        smoke(); return

    import pandas as pd
    df = clean_dataframe(pd.read_parquet(a.parquet))
    KERNELS = (3, 5, 9, 15, 21, 33)
    d = prepare_arrays(df, a.frames, kernel_sizes=KERNELS)
    run_training(d, epochs=a.epochs, batch_size=a.batch_size,
                 n_frames=a.n_frames, n_comp=a.n_comp, lr=a.lr,
                 lam_mean=a.lam_mean, lam_std=a.lam_std,
                 lam_skew=a.lam_skew, lam_kurt=a.lam_kurt, lam_ent=a.lam_ent)


if __name__ == "__main__":
    main()