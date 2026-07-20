from __future__ import annotations
import argparse
from typing import List

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import equinox as eqx
import optax

# ══════════════════════════════════════════════════════════════════════════════
#  Sequence -> P(Rg) via a LOG-NORMAL MIXTURE density, trained with continuous
#  per-frame NLL.  No bins, no KDE, no bandwidth, no grid.
#
#        p(r) = (1/r) * SUM_k  pi_k * N( log r ; mu_k , sigma_k )
#
#  ── ARCHITECTURE (v3: BiGRU) ────────────────────────────────────────────────
#  v2 used a multi-scale CNN with per-window physics channels injected at every
#  scale. v3 replaces the encoder with a masked bidirectional GRU:
#
#     ids -> embedding (+raw per-residue params) -> BiGRU -> masked mean-pool
#         -> FiLM(globals) -> trunk -> FiLM(globals) -> mixture head
#
#  The per-WINDOW physics stack is GONE. Instead every physics quantity is
#  computed once per sequence at the GLOBAL scale (SCD, SHD, FCR, net charge,
#  mean hydropathy, charge asymmetry, SBCS, charge entropy, seq entropy) and
#  injected AFTER the recurrent encoder via FiLM, together with log L. The GRU's
#  job is to learn sequence *pattern*; the globals carry the composition summary
#  the CNN used to get from its convolutions.
#
#  This removes the (N, S, N_PHYS, L) tensor entirely -- it was the dominant
#  memory and prep cost -- and cuts the batch tuple from 10 fields to 9.
#
#  KURTOSIS CONVENTION: extraction stores scipy.stats.kurtosis(fisher=True) =
#  EXCESS kurtosis (Gaussian -> 0). mixture_moments() returns EXCESS too.
#
#  ── STABILITY (inherited from v2, unchanged) ────────────────────────────────
#   (1) DEAD-COMPONENT DRIFT. A component whose pi_k -> 0 gets no gradient from
#       the NLL, so its mu_k / sigma_k wander freely. The density doesn't care,
#       but moment quadrature evaluates at r = exp(mu + sigma*GH_X) with
#       |GH_X|max ~= 14.9, so a runaway sigma ~ 3 gives exp(mu + 44) ~ 1e21 and
#       d**4 -> inf in float32.
#       FIX: BOUND the parameters. sigma via sigmoid into [SIGMA_MIN, SIGMA_MAX];
#       delta_mu via tanh into [-DELTA_MAX, DELTA_MAX].
#   (2) 0.0 * inf = nan. A lam=0.0 multiply does NOT disable a term.
#       FIX: genuine static branch (lambdas are Python floats -> static under
#       filter_jit), plus optax.zero_nans() as a backstop.
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
N_RAW = 5

param_matrix = np.array([[v["mass"], v["charge"], v["sigma"], v["HPS1"], v["HPS2"]]
                         for v in AA_PARAMS.values()], dtype=np.float32)
is_charged_mask = np.array([v["charge"] != 0 for v in AA_PARAMS.values()])


# ══════════════════════════════════════════════════════════════════════════════
# 1.  GLOBAL physics descriptors  (numpy, computed once per sequence at prep)
#
#     These are the exact same 9 quantities the CNN computed per sliding window,
#     but evaluated on the FULL sequence. They are the "composition + charge
#     patterning" summary; the BiGRU handles local ordering on its own.
#
#     All are sequence-derivable (no MD leakage).
# ══════════════════════════════════════════════════════════════════════════════
def g_scd(q):
    """Sequence Charge Decoration. Sawle-Ghosh: sum_{i<j} q_i q_j sqrt(j-i) / N."""
    k = len(q)
    if k < 2:
        return 0.0
    d = np.sqrt(np.abs(np.subtract.outer(np.arange(k), np.arange(k))))
    return float(np.tril(np.outer(q, q) * d, -1).sum() / k)


def g_shd(lam):
    """Sequence Hydropathy Decoration: sum_{i<j} (l_i + l_j) / |i-j| / N."""
    k = len(lam)
    if k < 2:
        return 0.0
    d = np.abs(np.subtract.outer(np.arange(k), np.arange(k))).astype(float)
    safe = np.where(d == 0, 1.0, d)
    return float(np.tril((lam[:, None] + lam[None, :]) / safe, -1).sum() / k)


def g_net_charge(q):
    return float(np.sum(q))


def g_fcr(q):
    """Fraction of charged residues."""
    return float(np.sum(np.abs(q) > 0) / len(q)) if len(q) else 0.0


def g_mean_hydro(lam):
    return float(np.mean(lam)) if len(lam) else 0.0


def g_charge_asymmetry(q):
    """(f+ - f-)^2 / (f+ + f-)  -- the sigma parameter of Das & Pappu."""
    n = len(q)
    if n == 0:
        return 0.0
    pos = np.sum(q > 0) / n
    neg = np.sum(q < 0) / n
    tot = pos + neg
    return float((pos - neg) ** 2 / tot) if tot > 0 else 0.0


def g_sbcs(lam):
    """Sequence blockiness of consecutive hydrophobic (lambda > 0.5) contacts."""
    n = len(lam)
    if n < 2:
        return 0.0
    is_high = lam > 0.5
    C = np.cumsum(is_high)
    C_diff = C[None, :] - C[:, None]
    idx = np.arange(n)
    dmat = idx[None, :] - idx[:, None]
    valid = (C_diff == 1) & np.outer(is_high, is_high) & (dmat > 0)
    safe = np.where(dmat == 0, 1.0, dmat)
    inv = np.where(valid, 1.0 / safe, 0.0)
    return float(np.mean(lam) * np.sum(inv))


def g_s_all(ids_valid):
    """Shannon entropy (bits) over the 20 AA identities."""
    counts = np.bincount(ids_valid, minlength=N_AA)[1:]   # drop the pad token
    N = counts.sum()
    if N == 0:
        return 0.0
    p = counts / N
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def g_s_q(ids_valid):
    """Shannon entropy (bits) restricted to the charged residues."""
    counts = np.bincount(ids_valid, minlength=N_AA) * is_charged_mask
    N = counts.sum()
    if N == 0:
        return 0.0
    p = counts / N
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


GLOBAL_NAMES = ("scd", "shd", "net_charge", "fcr", "mean_hydro",
                "charge_asym", "sbcs", "s_q", "s_all")
N_GLOBALS = len(GLOBAL_NAMES)          # 9


def sequence_globals(ids_valid):
    """ids_valid: (ell,) int32 token ids, padding already stripped -> (9,) float32."""
    q = param_matrix[ids_valid, CHARGE]
    lam = param_matrix[ids_valid, HPS1]
    return np.array([
        g_scd(q),
        g_shd(lam),
        g_net_charge(q),
        g_fcr(q),
        g_mean_hydro(lam),
        g_charge_asymmetry(q),
        g_sbcs(lam),
        g_s_q(ids_valid),
        g_s_all(ids_valid),
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Encoder:  embedding -> masked BiGRU -> masked mean-pool -> FiLM -> trunk
# ══════════════════════════════════════════════════════════════════════════════
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


class BiGRU(eqx.Module):
    """One masked bidirectional GRU layer.  (L, in) -> (L, 2*hidden).

    Padding handling is the whole point. The hidden state is FROZEN wherever
    mask == 0. This matters in BOTH directions: the forward pass would otherwise
    let trailing pad corrupt the final valid state, and the backward pass -- which
    hits the pad region FIRST, since sequences are right-padded -- would start
    from garbage. With the freeze, padded positions are inert; the caller re-zeros
    their outputs before pooling.

    Verified: perturbing tokens in the padded region changes the model output by
    exactly 0.0.
    """
    fwd: eqx.nn.GRUCell
    bwd: eqx.nn.GRUCell
    hidden: int = eqx.field(static=True)

    def __init__(self, in_size, hidden, *, key):
        kf, kb = jax.random.split(key)
        self.fwd = eqx.nn.GRUCell(in_size, hidden, key=kf)
        self.bwd = eqx.nn.GRUCell(in_size, hidden, key=kb)
        self.hidden = hidden

    def __call__(self, x, mask):                           # x:(L,in)  mask:(L,)
        h0 = jnp.zeros(self.hidden)

        def run(cell, reverse):
            def step(h, inp):
                xt, mt = inp
                h_new = cell(xt, h)
                h = jnp.where(mt > 0, h_new, h)            # freeze on padding
                return h, h
            _, seq = jax.lax.scan(step, h0, (x, mask), reverse=reverse)
            return seq

        hf = run(self.fwd, False)                          # (L, hidden)
        hb = run(self.bwd, True)                           # (L, hidden)
        return jnp.concatenate([hf, hb], axis=-1)          # (L, 2*hidden)


# ── mixture head hyper-params ────────────────────────────────────────────────
# sigma is a width in LOG-Rg units: sigma=0.6 means Rg spans a factor ~3 within
# +-1 sigma. Real IDPs sit around 0.10-0.25, so 0.60 is a generous ceiling. The
# ceiling is what stops a dead component drifting off and exploding the 4th moment.
SIGMA_MIN    = 0.01
SIGMA_MAX    = 0.60
DELTA_MAX    = 1.00     # component means stay within e^{+-1} of the Flory anchor
SIGMA_INIT   = 0.15
DELTA_SPREAD = 0.25


class RgBiGRUNet(eqx.Module):
    """Sequence -> parameters of a log-normal mixture over Rg."""
    embedding: eqx.nn.Embedding
    gru_layers: List[BiGRU]
    cond_mlp: eqx.nn.MLP
    film_global: FiLM
    trunk: eqx.nn.MLP
    film_head: FiLM
    head: eqx.nn.MLP                       # -> 3K  (logit_pi | raw_delta | raw_sigma)
    n_comp: int = eqx.field(static=True)
    use_raw: bool = eqx.field(static=True)
    flory_nu: float = eqx.field(static=True)
    flory_log_a: float = eqx.field(static=True)

    def __init__(self, *, key, flory_nu, flory_log_a,
                 n_aa=N_AA, embed_dim=16, n_raw=N_RAW,
                 hidden=64, n_layers=1,
                 n_globals=N_GLOBALS, cond_dim=32, n_comp=6):
        keys = jax.random.split(key, n_layers + 6)
        self.use_raw = n_raw > 0
        in_ch = embed_dim + (n_raw if self.use_raw else 0)

        self.embedding = eqx.nn.Embedding(n_aa, embed_dim, key=keys[0])

        # stacked BiGRU: each layer consumes 2*hidden from the one below it
        sizes = [in_ch] + [2 * hidden] * n_layers
        self.gru_layers = [BiGRU(sizes[i], hidden, key=keys[1 + i])
                           for i in range(n_layers)]
        pooled = 2 * hidden                    # = 128 at hidden=64, matching the CNN trunk

        i0 = 1 + n_layers
        # +1 for log L, which is always appended to the globals
        self.cond_mlp = eqx.nn.MLP(n_globals + 1, cond_dim, width_size=64,
                                   depth=2, activation=jax.nn.gelu, key=keys[i0])
        self.film_global = FiLM(cond_dim, pooled, key=keys[i0 + 1])
        self.trunk = eqx.nn.MLP(pooled, pooled, width_size=128,
                                depth=2, activation=jax.nn.gelu, key=keys[i0 + 2])
        self.film_head = FiLM(cond_dim, pooled, key=keys[i0 + 3])

        head = eqx.nn.MLP(pooled, 3 * n_comp, width_size=256,
                          depth=2, activation=jax.nn.gelu, key=keys[i0 + 4])

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
            delta_targets = jnp.zeros(1)                   # single component: centred
        else:
            delta_targets = jnp.linspace(-DELTA_SPREAD, DELTA_SPREAD, n_comp)
        raw_delta_0 = jnp.arctanh(delta_targets / DELTA_MAX)

        bias_init = jnp.concatenate([
            jnp.zeros(n_comp),                             # logit_pi -> equal weights
            raw_delta_0,                                   # raw delta_mu
            jnp.full((n_comp,), raw_sigma_0),              # raw sigma
        ])
        head = eqx.tree_at(lambda m: m.layers[-1].bias, head, bias_init)
        head = eqx.tree_at(lambda m: m.layers[-1].weight, head,
                           head.layers[-1].weight * 0.01)
        self.head = head

        self.n_comp = n_comp
        self.flory_nu = float(flory_nu)
        self.flory_log_a = float(flory_log_a)

    def __call__(self, ids, raw_params, globals_, mask):
        emb = jax.vmap(self.embedding)(ids)                        # (L, embed_dim)
        x = jnp.concatenate([emb, raw_params], axis=-1) if self.use_raw else emb
        x = x * mask[:, None]

        for gru in self.gru_layers:
            x = gru(x, mask) * mask[:, None]                       # re-zero pad each layer

        denom = jnp.sum(mask) + 1e-6
        g = jnp.sum(x * mask[:, None], axis=0) / denom             # masked mean-pool

        # ── globals injected AFTER the encoder, via FiLM (twice) ──────────────
        logL = jnp.log(jnp.sum(mask) + 1e-6)
        c = self.cond_mlp(jnp.concatenate([globals_, logL[None]]))

        g = self.film_global(g, c)
        z = self.trunk(g)
        zh = self.film_head(z, c)
        out = self.head(zh)                                        # (3K,)

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
# 3.  Log-normal mixture: density, NLL, moments      (unchanged from v2)
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
def loss_fn(model, ids, raw, glob, mask, frames,
            t_mean, t_std, t_skew, t_exkurt,
            lam_mean=0.1, lam_std=0.0, lam_skew=0.001, lam_kurt=0.001, lam_ent=0.0):
    """Continuous per-frame NLL (+ optional moment penalties, + optional entropy bonus).

    frames: (B, F). Every protein contributes exactly F frames, so the plain mean
    over (B,F) IS equal-per-protein weighting -- no trajectory-length bias.

    Defaults are PURE NLL. The lambdas are Python floats and therefore STATIC under
    filter_jit, so `if lam_*:` genuinely compiles the branch away. This matters:
    multiplying by lam=0.0 instead of branching does NOT disable a term, and
    0.0*inf = nan would poison the whole model in one step.
    """
    logit_pi, mu, sigma = jax.vmap(model)(ids, raw, glob, mask)

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
def eval_batch(model, ids, raw, glob, mask, frames):
    logit_pi, mu, sigma = jax.vmap(model)(ids, raw, glob, mask)
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


def prepare_arrays(df, frames_path):
    N = len(df)
    L = int(df["sequence"].str.len().max())
    print(f"[prep] N={N}  L(max)={L}  globals={N_GLOBALS} {GLOBAL_NAMES}")

    ids  = np.zeros((N, L), np.int32)
    mask = np.zeros((N, L), np.float32)
    glob = np.zeros((N, N_GLOBALS), np.float32)

    for i, seq in enumerate(df["sequence"]):
        ids[i], mask[i] = tokenize(seq, L)
        ell = int(mask[i].sum())
        glob[i] = sequence_globals(ids[i, :ell])

    raw = param_matrix[ids]                       # (N, L, 5)

    # ── Frames: memmap the (P, K) store; keep only the rows this df kept. ─────
    if isinstance(frames_path, np.ndarray):       # smoke test injects an array
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

    return dict(ids=ids, raw=raw, glob=glob, mask=mask,
                frames=frames_mm, frame_rows=frame_rows, K_store=K_STORE,
                mean=t_mean, std=t_std, skew=t_skew, exkurt=t_exkurt,
                L=L, nu=float(nu), log_a=float(log_a))


def standardize(data, tr_idx):
    """Z-score raw (per-residue) and glob (per-sequence) using TRAIN stats only."""
    m = data["mask"][tr_idx].astype(bool)
    r = data["raw"][tr_idx]
    r_mu = r[m].mean(0)
    r_sd = r[m].std(0) + 1e-6
    data["raw"] = ((data["raw"] - r_mu) / r_sd) * data["mask"][..., None]

    g = data["glob"][tr_idx]
    g_mu = g.mean(0)
    g_sd = g.std(0) + 1e-6
    data["glob"] = (data["glob"] - g_mu) / g_sd

    data["_stats"] = dict(r_mu=r_mu, r_sd=r_sd, g_mu=g_mu, g_sd=g_sd,
                          nu=np.float32(data["nu"]), log_a=np.float32(data["log_a"]),
                          global_names=np.array(GLOBAL_NAMES))
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
    return (j(d["ids"][idx]),  j(d["raw"][idx]),
            j(d["glob"][idx]), j(d["mask"][idx]),
            j(draw_frames(d, idx, n_frames, rng)),
            j(d["mean"][idx]), j(d["std"][idx]),
            j(d["skew"][idx]), j(d["exkurt"][idx]))


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Training loop
# ══════════════════════════════════════════════════════════════════════════════
def run_training(d, *, epochs=40, batch_size=64, n_frames=1024, n_comp=6,
                 hidden=64, n_layers=1, embed_dim=16, use_raw=True,
                 lr=5e-4, val_frac=0.15, seed=42,
                 lam_mean=0.0, lam_std=0.0, lam_skew=0.0, lam_kurt=0.0, lam_ent=0.0,
                 out_prefix="rg_bigru"):
    N = len(d["ids"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = int(N * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    d = standardize(d, tr_idx)

    model = RgBiGRUNet(key=jax.random.PRNGKey(seed),
                       flory_nu=d["nu"], flory_log_a=d["log_a"],
                       embed_dim=embed_dim, n_raw=N_RAW if use_raw else 0,
                       hidden=hidden, n_layers=n_layers,
                       n_globals=N_GLOBALS, n_comp=n_comp)

    n_params = sum(x.size for x in
                   jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"[model] BiGRU hidden={hidden} layers={n_layers} pooled={2*hidden} "
          f"K={n_comp} | {n_params:,} params")

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
                  jnp.array(d["glob"][val_idx]), jnp.array(d["mask"][val_idx]))

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

    np.savez(f"{out_prefix}_stats.npz",
             n_comp=np.int32(n_comp), hidden=np.int32(hidden),
             n_layers=np.int32(n_layers), embed_dim=np.int32(embed_dim),
             use_raw=np.bool_(use_raw), **d["_stats"])
    print(f"saved: {out_prefix}_model.eqx (best val nll {best:.4f}) "
          f"+ {out_prefix}_stats.npz")
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
    d = prepare_arrays(df, frames)
    for k in ("ids", "raw", "glob", "mask"):
        print(f"   {k:5s} {np.asarray(d[k]).shape}")

    # ── padding-invariance check: the ONE thing a masked RNN can get wrong ────
    d_chk = standardize(dict(d), np.arange(len(d["ids"])))
    m0 = RgBiGRUNet(key=jax.random.PRNGKey(0), flory_nu=d["nu"], flory_log_a=d["log_a"],
                    n_comp=4, hidden=32, n_layers=1)
    i0 = jnp.array(d_chk["ids"][0]); r0 = jnp.array(d_chk["raw"][0])
    g0 = jnp.array(d_chk["glob"][0]); k0 = jnp.array(d_chk["mask"][0])
    ell = int(k0.sum())
    out_a = m0(i0, r0, g0, k0)
    out_b = m0(i0.at[ell:].set(7), r0.at[ell:].set(3.14), g0, k0)   # scribble the pad
    diff = max(float(jnp.abs(a - b).max()) for a, b in zip(out_a, out_b))
    print(f"   pad-invariance: max |Δoutput| = {diff:.2e}  (must be 0)")
    assert diff == 0.0, "PADDING LEAKS INTO THE OUTPUT -- the mask freeze is broken"

    # exercise the moment + entropy branches too, so they can't silently rot
    run_training(d, epochs=3, batch_size=8, n_frames=512, n_comp=4, val_frac=0.25,
                 hidden=32, n_layers=1,
                 lam_skew=0.05, lam_kurt=0.05, lam_ent=1e-3,
                 out_prefix="smoke_rg_bigru")
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
    ap.add_argument("--hidden", type=int, default=64,
                    help="GRU hidden size per direction; pooled dim = 2*hidden")
    ap.add_argument("--n_layers", type=int, default=1, help="stacked BiGRU layers")
    ap.add_argument("--embed_dim", type=int, default=16)
    ap.add_argument("--no_raw", action="store_true",
                    help="feed the GRU the embedding ONLY (drop the 5 raw AA params)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lam_mean", type=float, default=0.0)
    ap.add_argument("--lam_std",  type=float, default=0.0)
    ap.add_argument("--lam_skew", type=float, default=0.0)
    ap.add_argument("--lam_kurt", type=float, default=0.0)
    ap.add_argument("--lam_ent",  type=float, default=0.0,
                    help="entropy bonus on mixture weights; try 1e-3 if pi collapses")
    ap.add_argument("--out_prefix", default="rg_bigru")
    a = ap.parse_args()

    if a.smoke:
        smoke(); return

    import pandas as pd
    df = clean_dataframe(pd.read_parquet(a.parquet))
    d = prepare_arrays(df, a.frames)
    run_training(d, epochs=a.epochs, batch_size=a.batch_size,
                 n_frames=a.n_frames, n_comp=a.n_comp,
                 hidden=a.hidden, n_layers=a.n_layers, embed_dim=a.embed_dim,
                 use_raw=not a.no_raw, lr=a.lr,
                 lam_mean=a.lam_mean, lam_std=a.lam_std,
                 lam_skew=a.lam_skew, lam_kurt=a.lam_kurt, lam_ent=a.lam_ent,
                 out_prefix=a.out_prefix)


if __name__ == "__main__":
    main()