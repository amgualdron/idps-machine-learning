from __future__ import annotations
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# Everything below reuses the *exact* head parameterisation, loss and data pipeline
# from mlp_mixture.py so that the two models are directly comparable.
from mlp_mixture import (
    RgMLPNet, GLOBAL_NAMES, N_GLOBALS,
    SIGMA_MIN, SIGMA_MAX, DELTA_MAX,
    batch_log_prob, batch_moments, mixture_weight_entropy,
    clean_dataframe, prepare_arrays, draw_frames,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Residual sequence correction:  raw = MLP(globals, logL) + g(sequence)
#
#  Design contract
#  ---------------
#  1. The correction is added in the *raw* (pre-activation) head space, i.e. to
#     the 3K vector that mlp_mixture feeds into softmax / tanh / sigmoid. All the
#     box constraints (SIGMA_MIN..SIGMA_MAX, |delta_mu| < DELTA_MAX) therefore
#     survive untouched, and the Flory anchor mu = log_a + nu*logL is unchanged.
#  2. The residual branch's final layer is *zero-initialised* and squashed by
#     res_scale * tanh(.). At step 0 the residual is identically 0, so the
#     network starts exactly at the MLP baseline: the BiGRU can only improve on
#     it, and it cannot blow up the head (|correction| <= res_scale per channel).
#  3. The MLP baseline is frozen during stage B (optax.set_to_zero on its label
#     group). Set --base_lr_mult > 0 for joint fine-tuning after the BiGRU warms up.
#  4. TWO penalties on the residual, and the choice of which one you use decides
#     whether the correction is SELECTIVE or SMEARED:
#
#       --lam_res   L2:          lam * mean_i mean_c r_ic^2
#           Quadratic in the norm => a thousand tiny corrections are CHEAPER than
#           one large one. This objective literally asks the model to spread the
#           correction thinly over every protein. It is why turning lam_res down
#           to let the outliers move also drags the already-correct IDPs.
#
#       --lam_group GROUP LASSO: lam * mean_i ||r_i||_2      <-- L2 within a
#           protein, L1 across proteins. The marginal cost of touching a protein
#           at all is a constant `lam` (the "toll"), independent of how large the
#           existing correction is. A protein gets a residual only if the NLL it
#           buys exceeds lam * ||r_i||. Most proteins are left alone; a few get
#           large corrections. That is the sparsity pattern you asked for.
#
#     ||r||_2 is non-differentiable at 0 and the zero-init contract starts every
#     residual at EXACTLY 0, so the norm is eps-smoothed: grad = r/sqrt(eps) = 0
#     at the origin. The branch starts at rest and is never kicked by the penalty.
#
#  5. Subgradient group lasso shrinks toward zero but does not produce exact
#     zeros. --res_thresh applies the group soft-threshold operator
#         r <- r * max(0, 1 - tau/||r||)
#     to the residual OUTPUT, which does: below the toll, the branch is switched
#     off for that protein and reports identically 0. It is disabled for the
#     first --prune_warmup epochs, because at init ||r||=0 and thresholding would
#     freeze the branch dead before it ever learns anything.
# ══════════════════════════════════════════════════════════════════════════════

PAD_ID = 0  # tokens are AA_TO_ID[a] + 1, so 0 is a free padding slot


# ── head decode (identical algebra to RgMLPNet.__call__, lines 82-90) ─────────
def decode_head(raw, logL, n_comp, flory_nu, flory_log_a):
    K = n_comp
    logit_pi = raw[:K]
    delta_mu = DELTA_MAX * jnp.tanh(raw[K:2 * K])
    sigma = SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * jax.nn.sigmoid(raw[2 * K:])
    mu = flory_log_a + flory_nu * logL + delta_mu
    return logit_pi, mu, sigma


# ══════════════════════════════════════════════════════════════════════════════
# Bidirectional GRU (masked, variable length)
# ══════════════════════════════════════════════════════════════════════════════
class BiGRULayer(eqx.Module):
    fwd: eqx.nn.GRUCell
    bwd: eqx.nn.GRUCell
    hidden: int = eqx.field(static=True)

    def __init__(self, in_size, hidden, *, key):
        kf, kb = jax.random.split(key)
        self.fwd = eqx.nn.GRUCell(in_size, hidden, key=kf)
        self.bwd = eqx.nn.GRUCell(in_size, hidden, key=kb)
        self.hidden = hidden

    def __call__(self, x, mask):
        """x: (L, D) | mask: (L,) bool -> (L, 2H). Padded steps carry h forward."""
        h0 = jnp.zeros(self.hidden)

        def scanner(cell):
            def step(h, inp):
                xt, mt = inp
                h_new = cell(xt, h)
                h_out = jnp.where(mt, h_new, h)   # frozen state on pad
                return h_out, h_out
            return step

        _, h_f = jax.lax.scan(scanner(self.fwd), h0, (x, mask))
        _, h_b_rev = jax.lax.scan(scanner(self.bwd), h0, (x[::-1], mask[::-1]))
        return jnp.concatenate([h_f, h_b_rev[::-1]], axis=-1)


class SeqResidual(eqx.Module):
    """Embedded sequence -> BiGRU -> pooled -> zero-init head -> raw 3K correction."""
    embed: eqx.nn.Embedding
    grus: List[BiGRULayer]
    norms: List[eqx.nn.LayerNorm]
    attn: eqx.nn.Linear
    head: eqx.nn.MLP
    drop: eqx.nn.Dropout
    n_comp: int = eqx.field(static=True)
    res_scale: float = eqx.field(static=True)
    use_pi: bool = eqx.field(static=True)
    use_mu: bool = eqx.field(static=True)
    use_sigma: bool = eqx.field(static=True)
    cond_globals: bool = eqx.field(static=True)

    def __init__(self, *, key, vocab, n_comp, embed_dim=32, hidden=64, layers=1,
                 head_width=128, dropout=0.1, res_scale=1.0,
                 channels=("pi", "mu", "sigma"), cond_globals=True):
        ke, kg, ka, kh = jax.random.split(key, 4)
        self.embed = eqx.nn.Embedding(vocab, embed_dim, key=ke)

        gkeys = jax.random.split(kg, layers)
        self.grus, self.norms = [], []
        in_size = embed_dim
        for i in range(layers):
            self.grus.append(BiGRULayer(in_size, hidden, key=gkeys[i]))
            self.norms.append(eqx.nn.LayerNorm(2 * hidden))
            in_size = 2 * hidden

        self.attn = eqx.nn.Linear(2 * hidden, 1, key=ka)

        pool_dim = 4 * hidden                       # [mean_pool ; attn_pool]
        cond_dim = (N_GLOBALS + 1) if cond_globals else 0
        self.head = eqx.nn.MLP(
            in_size=pool_dim + cond_dim,
            out_size=3 * n_comp,
            width_size=head_width,
            depth=2,
            activation=jax.nn.gelu,
            key=kh,
        )
        # Zero-init the last layer => residual is exactly 0 at initialisation.
        self.head = eqx.tree_at(lambda m: m.layers[-1].weight, self.head,
                                jnp.zeros_like(self.head.layers[-1].weight))
        self.head = eqx.tree_at(lambda m: m.layers[-1].bias, self.head,
                                jnp.zeros_like(self.head.layers[-1].bias))

        self.drop = eqx.nn.Dropout(dropout)
        self.n_comp = n_comp
        self.res_scale = float(res_scale)
        self.use_pi = "pi" in channels
        self.use_mu = "mu" in channels
        self.use_sigma = "sigma" in channels
        self.cond_globals = cond_globals

    def channel_mask(self):
        K = self.n_comp
        return jnp.concatenate([
            jnp.full(K, float(self.use_pi)),
            jnp.full(K, float(self.use_mu)),
            jnp.full(K, float(self.use_sigma)),
        ])

    def __call__(self, glob, logL, tok, mask, *, key=None, inference=False, thresh=0.0):
        h = jax.vmap(self.embed)(tok)                       # (L, E)
        for gru, ln in zip(self.grus, self.norms):
            h = gru(h, mask)                               # (L, 2H)
            h = jax.vmap(ln)(h)

        m = mask.astype(h.dtype)[:, None]
        mean_pool = jnp.sum(h * m, axis=0) / jnp.clip(jnp.sum(m), 1.0)

        scores = jax.vmap(self.attn)(h)[:, 0]              # (L,)
        scores = jnp.where(mask, scores, -1e9)
        attn_pool = jnp.sum(jax.nn.softmax(scores)[:, None] * h, axis=0)

        z = jnp.concatenate([mean_pool, attn_pool])
        if self.cond_globals:
            z = jnp.concatenate([z, glob, logL[None]])
        z = self.drop(z, key=key, inference=inference)

        raw = self.res_scale * jnp.tanh(self.head(z))      # bounded correction
        raw = raw * self.channel_mask()

        # Group soft-threshold: switch the branch OFF entirely for this protein
        # unless its correction is worth more than the toll `thresh`. Exact zeros.
        n = jnp.sqrt(jnp.sum(raw ** 2) + 1e-12)
        keep = jnp.where(thresh > 0.0, jnp.clip(1.0 - thresh / n, 0.0, None), 1.0)
        return raw * keep


class ResidualRgNet(eqx.Module):
    """Frozen global-feature MLP + additive BiGRU correction in raw head space."""
    base: RgMLPNet
    seq: SeqResidual

    def __call__(self, glob, logL, tok, mask, *, key=None, inference=False, thresh=0.0):
        raw_base = self.base.mlp(jnp.concatenate([glob, logL[None]]))
        res = self.seq(glob, logL, tok, mask, key=key, inference=inference, thresh=thresh)
        logit_pi, mu, sigma = decode_head(raw_base + res, logL, self.base.n_comp,
                                          self.base.flory_nu, self.base.flory_log_a)
        return logit_pi, mu, sigma, res


def forward(model, glob, logL, tok, mask, key, inference, thresh=0.0):
    keys = jax.random.split(key, glob.shape[0])
    f = lambda g, l, t, m, k: model(g, l, t, m, key=k, inference=inference, thresh=thresh)
    return jax.vmap(f)(glob, logL, tok, mask, keys)


# ══════════════════════════════════════════════════════════════════════════════
# Loss (mlp_mixture's loss + a residual-magnitude penalty)
# ══════════════════════════════════════════════════════════════════════════════
GROUP_EPS = 1e-8   # smooths ||r||_2 at the origin: grad(0) = 0, not NaN


def group_norm(res):
    """Per-protein L2 norm of the raw head-space correction, eps-smoothed."""
    return jnp.sqrt(jnp.sum(res ** 2, axis=-1) + GROUP_EPS)


def loss_fn(model, glob, logL, tok, mask, frames, t_mean, t_std, t_skew, t_exkurt, key,
            lam_mean=0.0, lam_std=0.0, lam_skew=0.0, lam_kurt=0.0, lam_ent=0.0,
            lam_res=0.0, lam_group=0.0, thresh=0.0):
    logit_pi, mu, sigma, res = forward(model, glob, logL, tok, mask, key,
                                       inference=False, thresh=thresh)
    nll = -jnp.mean(batch_log_prob(logit_pi, mu, sigma, frames))

    loss = nll
    zero = jnp.zeros(())
    res_rms = jnp.sqrt(jnp.mean(res ** 2) + 1e-12)
    aux = dict(nll=nll, l_mean=zero, l_std=zero, l_skew=zero, l_kurt=zero,
               ent=zero, res_rms=res_rms)

    if lam_mean or lam_std or lam_skew or lam_kurt:
        pm, psd, psk, pku = batch_moments(logit_pi, mu, sigma)
        l_mean = jnp.mean(jnp.abs(pm - t_mean) / (t_mean + 1e-6))
        l_std = jnp.mean(jnp.abs(psd - t_std) / (t_std + 1e-6))
        l_skew = jnp.mean(jnp.abs(psk - t_skew))
        l_kurt = jnp.mean(jnp.abs(pku - t_exkurt))
        loss = (loss + lam_mean * l_mean + lam_std * l_std
                + lam_skew * l_skew + lam_kurt * l_kurt)
        aux.update(l_mean=l_mean, l_std=l_std, l_skew=l_skew, l_kurt=l_kurt)

    if lam_ent:
        ent = jnp.mean(jax.vmap(mixture_weight_entropy)(logit_pi))
        loss = loss - lam_ent * ent
        aux.update(ent=ent)

    if lam_res:                                    # ridge: smears corrections thin
        loss = loss + lam_res * jnp.mean(res ** 2)

    if lam_group:                                  # group lasso: sparse across proteins
        loss = loss + lam_group * jnp.mean(group_norm(res))

    return loss, aux


@eqx.filter_jit
def train_step(model, opt_state, optimizer, batch, key, lams, thresh):
    glob, logL, tok, mask, frames, t_mean, t_std, t_skew, t_exkurt = batch
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, glob, logL, tok, mask, frames, t_mean, t_std, t_skew, t_exkurt, key,
        thresh=thresh, **lams
    )
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux


@eqx.filter_jit
def eval_batch(model, glob, logL, tok, mask, frames, thresh=0.0):
    key = jax.random.PRNGKey(0)  # unused: inference=True disables dropout
    logit_pi, mu, sigma, res = forward(model, glob, logL, tok, mask, key,
                                       inference=True, thresh=thresh)
    nll = -jnp.mean(batch_log_prob(logit_pi, mu, sigma, frames))
    pm, psd, psk, pku = batch_moments(logit_pi, mu, sigma)
    ent = jnp.mean(jax.vmap(mixture_weight_entropy)(logit_pi))
    res_rms = jnp.sqrt(jnp.mean(res ** 2) + 1e-12)
    return nll, pm, psd, psk, pku, ent, res_rms


@eqx.filter_jit
def eval_per_seq(model, glob, logL, tok, mask, frames, thresh=0.0):
    """Per-protein NLL for BOTH heads + per-protein correction norm.

    This is the measurement that decides whether the residual is selective:
    dnll_i = base_nll_i - resid_nll_i is the nats the sequence bought on protein i.
    Positive = helped. Negative = the correction DAMAGED a protein the MLP had right.
    """
    key = jax.random.PRNGKey(0)
    lp, mu, sg, res = forward(model, glob, logL, tok, mask, key, inference=True, thresh=thresh)
    nll_i = -jnp.mean(batch_log_prob(lp, mu, sg, frames), axis=1)

    blp, bmu, bsg = jax.vmap(model.base)(glob, logL)          # sequence branch OFF
    bnll_i = -jnp.mean(batch_log_prob(blp, bmu, bsg, frames), axis=1)

    return nll_i, bnll_i, jnp.sqrt(jnp.sum(res ** 2, axis=-1))


# ══════════════════════════════════════════════════════════════════════════════
# Optimiser with frozen / fine-tuned parameter groups
# ══════════════════════════════════════════════════════════════════════════════
def make_optimizer(model, *, train: str, lr, base_lr_mult=0.0, wd=1e-4,
                   total_steps=1, warmup_frac=0.05, cosine=True):
    """train='base' -> only the MLP moves; train='seq' -> only the BiGRU moves
    (unless base_lr_mult > 0, which fine-tunes the MLP at lr*base_lr_mult)."""
    params = eqx.filter(model, eqx.is_inexact_array)
    labels = jax.tree_util.tree_map(lambda _: "seq", params)
    labels = eqx.tree_at(lambda m: m.base, labels,
                         replace=jax.tree_util.tree_map(lambda _: "base", params.base))

    if cosine:
        sched = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.1, peak_value=lr,
            warmup_steps=max(1, int(warmup_frac * total_steps)),
            decay_steps=max(2, total_steps), end_value=lr * 0.05)
    else:
        sched = lr

    def adam(scale):
        s = (lambda c: (lambda cnt: scale * c(cnt)))(sched) if callable(sched) else scale * sched
        return optax.chain(optax.zero_nans(),
                           optax.clip_by_global_norm(1.0),
                           optax.adamw(s, weight_decay=wd))

    if train == "base":
        txs = {"base": adam(1.0), "seq": optax.set_to_zero()}
    else:
        txs = {"seq": adam(1.0),
               "base": adam(base_lr_mult) if base_lr_mult > 0 else optax.set_to_zero()}

    # NB: `labels` is an eqx.Module, hence callable -> optax would mistake it for a
    # label *function*. Wrap it so it is passed through as a pytree.
    opt = optax.multi_transform(txs, lambda _p: labels)
    return opt, opt.init(params)


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════
def prepare_arrays_seq(df, frames_path, max_len=None) -> Dict[str, Any]:
    from bigru_mixture import AA_TO_ID

    d = prepare_arrays(df, frames_path)

    seqs = [str(s).upper() for s in df["sequence"]]
    lens = np.array([len(s) for s in seqs], np.int32)
    L = int(max_len) if max_len else int(lens.max())
    tok = np.zeros((len(seqs), L), np.int32)
    for i, s in enumerate(seqs):
        ids = [AA_TO_ID.get(a, 0) + 1 for a in s[:L]]   # +1 keeps 0 free for PAD
        tok[i, :len(ids)] = ids

    d.update(tok=tok, mask=(tok != PAD_ID), lens=np.minimum(lens, L),
             max_len=L, vocab=int(max(AA_TO_ID.values())) + 2)
    print(f"[prep] Tokenised sequences: pad-to-L={L} | vocab={d['vocab']} | "
          f"median length {int(np.median(lens))}")
    return d


def standardize(d, tr_idx, stats: Optional[Dict[str, Any]] = None):
    """Reuse the frozen MLP's feature statistics when a checkpoint is loaded,
    otherwise fit them on this split's training fold."""
    if stats is not None:
        g_mu, g_sd = np.asarray(stats["g_mu"]), np.asarray(stats["g_sd"])
        print("[prep] Reusing g_mu/g_sd from the pretrained MLP stats file")
    else:
        g = d["glob"][tr_idx]
        g_mu, g_sd = g.mean(0), g.std(0) + 1e-6
    d["glob"] = (d["glob"] - g_mu) / g_sd
    d["_stats"] = dict(g_mu=g_mu, g_sd=g_sd,
                       nu=np.float32(d["nu"]), log_a=np.float32(d["log_a"]),
                       global_names=np.array(GLOBAL_NAMES))
    return d


def make_batch(d, idx, n_frames, rng):
    j = jnp.array
    return (j(d["glob"][idx]), j(d["logL"][idx]), j(d["tok"][idx]), j(d["mask"][idx]),
            j(draw_frames(d, idx, n_frames, rng)),
            j(d["mean"][idx]), j(d["std"][idx]), j(d["skew"][idx]), j(d["exkurt"][idx]))


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════
def _rank(x):
    """Ranks (0..n-1), ties broken arbitrarily — enough for a Spearman estimate."""
    return np.argsort(np.argsort(np.asarray(x))).astype(float)


def selectivity_report(model, d, val, thresh, res_scale, tag=""):
    """Does the correction TARGET the proteins the MLP got wrong, or smear over all?

    Splits the validation set into quartiles by BASELINE difficulty and reports,
    per quartile, how many nats the sequence branch bought and how hard it pushed.
    The pathology you are hunting looks like: Q4 (hard) improves, Q1 (already
    correct) gets a nonzero ||r|| and a NEGATIVE dnll — collateral damage.
    """
    val_idx, vg, vl, vt, vm, vf = val
    nll_i, bnll_i, rnorm = eval_per_seq(model, vg, vl, vt, vm, vf, thresh)
    nll_i, bnll_i, rnorm = map(np.asarray, (nll_i, bnll_i, rnorm))
    dnll = bnll_i - nll_i                       # > 0 == the sequence helped

    silent = rnorm < 0.01 * res_scale
    helped, harmed = dnll > 0, dnll < 0
    help_nats = float(dnll[helped].sum() / len(dnll)) if helped.any() else 0.0
    harm_nats = float(-dnll[harmed].sum() / len(dnll)) if harmed.any() else 0.0

    print(f"\n┌─ selectivity{tag} ─ n={len(dnll)} val proteins")
    print(f"│  quartiles by BASELINE difficulty (Q1 = MLP already correct, Q4 = MLP worst)")
    print(f"│  {'':4s} {'n':>4s} {'base nll':>9s} {'Δnll':>8s} {'‖r‖':>7s} {'silent':>7s} {'harmed':>7s}")
    q = np.argsort(bnll_i)
    for k, chunk in enumerate(np.array_split(q, 4)):
        print(f"│  Q{k+1:<3d} {len(chunk):4d} {bnll_i[chunk].mean():9.4f} "
              f"{dnll[chunk].mean():+8.4f} {rnorm[chunk].mean():7.3f} "
              f"{100*silent[chunk].mean():6.0f}% {100*harmed[chunk].mean():6.0f}%")
    print(f"│")
    print(f"│  help  {help_nats:+.4f} nats   (summed gain, spread over all n)")
    print(f"│  harm  {-harm_nats:+.4f} nats   (summed damage to proteins made WORSE)")
    print(f"│  net   {help_nats - harm_nats:+.4f} nats   "
          f"= {float(dnll.mean()):+.4f} (mean Δnll)")
    print(f"│  proteins left untouched (‖r‖ < 1% of res_scale): "
          f"{100*silent.mean():.0f}%   |  made worse: {100*harmed.mean():.0f}%")
    rho = float(np.corrcoef(_rank(rnorm), _rank(bnll_i))[0, 1])
    print(f"│  Spearman(‖r‖, baseline nll) = {rho:+.3f}   "
          f"(> 0 means the correction goes where the MLP is weak — that is targeting)")
    print(f"└{'─'*62}")
    return dict(dnll=dnll, bnll=bnll_i, rnorm=rnorm, help=help_nats, harm=harm_nats,
                silent=float(silent.mean()), harmed=float(harmed.mean()), rho=rho)


def _run_stage(model, d, tr_idx, val, *, stage, epochs, batch_size, n_frames, lr,
               base_lr_mult, wd, lams, seed, cosine, ref_nll=None,
               res_thresh=0.0, prune_warmup=0, res_scale=1.0):
    val_idx, val_glob, val_logL, val_tok, val_mask, val_frames = val
    rng = np.random.default_rng(seed + (0 if stage == "base" else 1))
    key = jax.random.PRNGKey(seed + (0 if stage == "base" else 1))

    steps = max(1, len(tr_idx) // batch_size)
    opt, opt_state = make_optimizer(model, train=stage, lr=lr, base_lr_mult=base_lr_mult,
                                    wd=wd, total_steps=epochs * steps, cosine=cosine)

    mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - b)))
    use_moments = any(lams[k] for k in ("lam_mean", "lam_std", "lam_skew", "lam_kurt"))
    best, best_model = np.inf, model

    tag = "A/base" if stage == "base" else "B/residual"
    print(f"\n── stage {tag}: {epochs} epochs x {steps} steps "
          f"(lr {lr:g}, base_lr_mult {base_lr_mult:g}) ──")

    for ep in range(epochs):
        rng.shuffle(tr_idx)
        # Thresholding is OFF during warmup: at init ||r|| = 0 and the soft-threshold
        # would zero the branch before it ever produced a gradient.
        tau = jnp.float32(res_thresh if (stage == "seq" and ep >= prune_warmup) else 0.0)
        acc = dict(loss=0.0, nll=0.0, l_mean=0.0, l_std=0.0, l_skew=0.0, l_kurt=0.0, res_rms=0.0)
        for s in range(steps):
            bidx = tr_idx[s * batch_size:(s + 1) * batch_size]
            batch = make_batch(d, bidx, n_frames, rng)
            key, sk = jax.random.split(key)
            model, opt_state, l, aux = train_step(model, opt_state, opt, batch, sk, lams, tau)
            acc["loss"] += float(l)
            for k in ("nll", "l_mean", "l_std", "l_skew", "l_kurt", "res_rms"):
                acc[k] += float(aux[k])

        vnll, pm, psd, psk, pku, vent, vres = eval_batch(
            model, val_glob, val_logL, val_tok, val_mask, val_frames, tau)
        vnll = float(vnll)

        # cheap per-epoch selectivity: what fraction is left alone, what got hurt
        if stage == "seq":
            ni, bi, rn = eval_per_seq(model, val_glob, val_logL, val_tok,
                                      val_mask, val_frames, tau)
            dn = np.asarray(bi) - np.asarray(ni)
            rn = np.asarray(rn)
            sil = float(np.mean(rn < 0.01 * res_scale))
            hrm = float(np.mean(dn < 0))
            sel = (f" | silent {100*sil:3.0f}% harmed {100*hrm:3.0f}% "
                   f"‖r‖ {rn.mean():.3f}")
        else:
            sel = ""

        delta = f" (Δ vs base {vnll - ref_nll:+.4f})" if ref_nll is not None else ""
        print(f"ep {ep+1:02d}/{epochs} | train {acc['loss']/steps:8.4f} "
              f"(nll {acc['nll']/steps:7.4f}) | val nll {vnll:7.4f}{delta} | "
              f"val MAE: mean {mae(pm, d['mean'][val_idx]):.3f}  "
              f"std {mae(psd, d['std'][val_idx]):.3f}  "
              f"skew {mae(psk, d['skew'][val_idx]):.3f}  "
              f"kurt {mae(pku, d['exkurt'][val_idx]):.3f}  "
              f"| pi-ent {float(vent):.2f}{sel}")

        if use_moments:
            print(f"          moment terms -> mean {acc['l_mean']/steps:.4f}  "
                  f"std {acc['l_std']/steps:.4f}  skew {acc['l_skew']/steps:.4f}  "
                  f"kurt {acc['l_kurt']/steps:.4f}")

        if not np.isfinite(vnll):
            print("Non-finite validation loss. Stopping this stage.")
            break
        if vnll < best:
            best, best_model = vnll, model

    print(f"── stage {tag} best val nll {best:.4f} ──")
    return best_model, best


def load_pretrained_base(ckpt, stats_path, key):
    st = np.load(stats_path, allow_pickle=True)
    base = RgMLPNet(key=key, flory_nu=float(st["nu"]), flory_log_a=float(st["log_a"]),
                    width=int(st["width"]), depth=int(st["depth"]), n_comp=int(st["n_comp"]))
    base = eqx.tree_deserialise_leaves(ckpt, base)
    print(f"[load] Pretrained MLP from {ckpt} "
          f"(n_comp={int(st['n_comp'])}, width={int(st['width'])}, depth={int(st['depth'])})")
    print("[load] NOTE: keep --seed / --val_frac identical to the mlp_mixture run "
          "that produced this checkpoint, or its training proteins will leak into val.")
    return base, st


def run_training(d, *, epochs=40, base_epochs=30, stage="both", batch_size=64, n_frames=1024,
                 n_comp=6, width=128, depth=3, lr=1e-3, base_lr=5e-4, base_lr_mult=0.0,
                 embed_dim=32, hidden=64, gru_layers=1, head_width=128, dropout=0.1,
                 res_scale=1.0, channels=("pi", "mu", "sigma"), cond_globals=True,
                 wd=1e-4, cosine=True, val_frac=0.15, seed=42,
                 lam_mean=0.0, lam_std=0.0, lam_skew=0.0, lam_kurt=0.0, lam_ent=0.0,
                 lam_res=0.0, lam_group=0.0, res_thresh=0.0, prune_warmup=5,
                 mlp_ckpt=None, mlp_stats=None, out_prefix="rg_resid"):
    N = len(d["glob"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = int(N * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    pre_stats = None
    base_key, seq_key = jax.random.split(jax.random.PRNGKey(seed))
    if mlp_ckpt:
        base, pre_stats = load_pretrained_base(mlp_ckpt, mlp_stats, base_key)
        n_comp = base.n_comp
    else:
        base = RgMLPNet(key=base_key, flory_nu=d["nu"], flory_log_a=d["log_a"],
                        width=width, depth=depth, n_comp=n_comp)

    d = standardize(d, tr_idx, stats=pre_stats)

    seq = SeqResidual(key=seq_key, vocab=d["vocab"], n_comp=n_comp, embed_dim=embed_dim,
                      hidden=hidden, layers=gru_layers, head_width=head_width,
                      dropout=dropout, res_scale=res_scale, channels=channels,
                      cond_globals=cond_globals)
    model = ResidualRgNet(base=base, seq=seq)

    p_base = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model.base, eqx.is_array)))
    p_seq = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model.seq, eqx.is_array)))
    print(f"[model] base MLP {p_base:,} params (frozen in stage B) | "
          f"BiGRU residual {p_seq:,} params | K={n_comp} | res_scale={res_scale} | "
          f"channels={','.join(channels)}")

    val_rng = np.random.default_rng(0)
    val = (val_idx,
           jnp.array(d["glob"][val_idx]), jnp.array(d["logL"][val_idx]),
           jnp.array(d["tok"][val_idx]), jnp.array(d["mask"][val_idx]),
           jnp.array(draw_frames(d, val_idx, 4096, val_rng)))

    lams = dict(lam_mean=lam_mean, lam_std=lam_std, lam_skew=lam_skew,
                lam_kurt=lam_kurt, lam_ent=lam_ent, lam_res=lam_res, lam_group=lam_group)
    common = dict(batch_size=batch_size, n_frames=n_frames, wd=wd, lams=lams,
                  seed=seed, cosine=cosine, res_scale=res_scale)
    if lam_group and lam_res:
        print(f"[warn] lam_res={lam_res} (ridge, smears) and lam_group={lam_group} "
              f"(lasso, sparsifies) are fighting each other. Set --lam_res 0.")
    if lam_group:
        print(f"[penalty] group lasso lam={lam_group}: a protein gets a correction only "
              f"if it buys more than {lam_group} x ||r|| nats")

    # ── Stage A: train the global-feature MLP (residual is frozen at exactly 0,
    #    so this reproduces mlp_mixture training inside the combined model) ────
    if stage in ("both", "base"):
        model, base_nll = _run_stage(model, d, tr_idx, val, stage="base", epochs=base_epochs,
                                     lr=base_lr, base_lr_mult=0.0, **common)
        eqx.tree_serialise_leaves(f"{out_prefix}_base.eqx", model.base)
    else:
        nll0, *_ = eval_batch(model, *val[1:])
        base_nll = float(nll0)
    print(f"\n[ref] frozen-baseline val NLL = {base_nll:.4f}  "
          f"(stage B must beat this for the sequence to be doing work)")

    # ── Stage B: freeze the MLP, learn the BiGRU residual ────────────────────
    if stage in ("both", "residual"):
        model, best = _run_stage(model, d, tr_idx, val, stage="seq", epochs=epochs, lr=lr,
                                 base_lr_mult=base_lr_mult, ref_nll=base_nll,
                                 res_thresh=res_thresh, prune_warmup=prune_warmup, **common)
        eqx.tree_serialise_leaves(f"{out_prefix}_model.eqx", model)
        gain = base_nll - best
        print(f"\n[result] baseline {base_nll:.4f} -> residual {best:.4f} "
              f"| sequence gain {gain:+.4f} nats "
              f"({'sequence helps' if gain > 0 else 'no gain: composition is sufficient'})")
        selectivity_report(model, d, val, res_thresh, res_scale)

    # Read the base geometry off the module itself: with --mlp_ckpt the checkpoint's
    # width/depth win over the CLI args, and the stats file must describe the real model.
    np.savez(f"{out_prefix}_stats.npz",
             n_comp=np.int32(model.base.n_comp),
             width=np.int32(model.base.mlp.width_size),
             depth=np.int32(model.base.mlp.depth),
             embed_dim=np.int32(embed_dim), hidden=np.int32(hidden),
             gru_layers=np.int32(gru_layers), head_width=np.int32(head_width),
             dropout=np.float32(dropout), res_scale=np.float32(res_scale),
             max_len=np.int32(d["max_len"]), vocab=np.int32(d["vocab"]),
             cond_globals=np.bool_(cond_globals), channels=np.array(channels),
             **d["_stats"])
    return model


# ══════════════════════════════════════════════════════════════════════════════
def smoke():
    print("── RESIDUAL BiGRU SMOKE RUN ──")
    from bigru_mixture import synthetic_data
    df, frames = synthetic_data(P=24, K_store=1000)
    df = clean_dataframe(df, frame_threshold=500, min_len=10)
    d = prepare_arrays_seq(df, frames)

    # Contract check: at init the residual model must equal the plain MLP exactly.
    base = RgMLPNet(key=jax.random.PRNGKey(0), flory_nu=d["nu"], flory_log_a=d["log_a"],
                    width=32, depth=2, n_comp=3)
    seq = SeqResidual(key=jax.random.PRNGKey(1), vocab=d["vocab"], n_comp=3,
                      embed_dim=8, hidden=8, head_width=16)
    m = ResidualRgNet(base=base, seq=seq)
    g = jnp.array((d["glob"][0] - d["glob"].mean(0)) / (d["glob"].std(0) + 1e-6))
    lp0, mu0, sg0 = base(g, jnp.float32(d["logL"][0]))
    lp1, mu1, sg1, res = m(g, jnp.float32(d["logL"][0]),
                           jnp.array(d["tok"][0]), jnp.array(d["mask"][0]), inference=True)
    err = max(float(jnp.abs(lp0 - lp1).max()), float(jnp.abs(mu0 - mu1).max()),
              float(jnp.abs(sg0 - sg1).max()), float(jnp.abs(res).max()))
    assert err < 1e-6, f"zero-init contract broken: {err}"
    print(f"[check] residual == 0 at init, head matches MLP exactly (max err {err:.2e})")

    run_training(d, base_epochs=2, epochs=3, batch_size=4, n_frames=256, n_comp=3,
                 width=32, depth=2, embed_dim=8, hidden=16, head_width=32,
                 val_frac=0.2, lam_skew=0.01, lam_kurt=0.01, lam_group=0.02,
                 res_thresh=0.02, prune_warmup=1, out_prefix="smoke_resid")
    print("── SMOKE STATUS OK ──")


def main():
    ap = argparse.ArgumentParser(description="BiGRU residual correction on top of the "
                                             "global-feature MLP log-normal mixture.")
    ap.add_argument("--parquet", default="data/idp_ml_dataset.parquet")
    ap.add_argument("--frames", default="data/idp_rg_frames.npy")
    ap.add_argument("--smoke", action="store_true")

    ap.add_argument("--stage", choices=["both", "base", "residual"], default="both",
                    help="'both' trains the MLP then freezes it; 'residual' requires --mlp_ckpt")
    ap.add_argument("--mlp_ckpt", default=None, help="pretrained rg_mlp_model.eqx")
    ap.add_argument("--mlp_stats", default=None, help="matching rg_mlp_stats.npz")

    ap.add_argument("--base_epochs", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_frames", type=int, default=1024)
    ap.add_argument("--max_len", type=int, default=None, help="pad/truncate length (default: dataset max)")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    # baseline MLP
    ap.add_argument("--n_comp", type=int, default=6)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--base_lr", type=float, default=5e-4)

    # residual branch
    ap.add_argument("--embed_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64, help="BiGRU hidden size per direction")
    ap.add_argument("--gru_layers", type=int, default=1)
    ap.add_argument("--head_width", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--res_scale", type=float, default=1.0,
                    help="max |correction| per raw head channel")
    ap.add_argument("--channels", default="pi,mu,sigma",
                    help="which head channels the sequence may correct")
    ap.add_argument("--no_cond_globals", action="store_true",
                    help="hide the global descriptors from the residual branch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base_lr_mult", type=float, default=0.0,
                    help=">0 unfreezes the MLP in stage B at lr*mult (e.g. 0.1)")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--no_cosine", action="store_true")

    ap.add_argument("--lam_mean", type=float, default=0.0)
    ap.add_argument("--lam_std", type=float, default=0.0)
    ap.add_argument("--lam_skew", type=float, default=0.0)
    ap.add_argument("--lam_kurt", type=float, default=0.0)
    ap.add_argument("--lam_ent", type=float, default=0.0)
    ap.add_argument("--lam_res", type=float, default=0.0,
                    help="RIDGE penalty mean(r^2). Quadratic => prefers many small "
                         "corrections over a few large ones. This is what smears.")
    ap.add_argument("--lam_group", type=float, default=0.02,
                    help="GROUP LASSO mean_i ||r_i||_2: flat toll per protein touched, so "
                         "most proteins keep a zero correction and outliers get large ones")
    ap.add_argument("--res_thresh", type=float, default=0.0,
                    help="group soft-threshold on the output => EXACT zeros. Try ~0.5*lam_group "
                         "after checking the ||r|| distribution. 0 = off (subgradient only)")
    ap.add_argument("--prune_warmup", type=int, default=5,
                    help="epochs before --res_thresh switches on (0 would freeze the "
                         "zero-initialised branch dead)")
    ap.add_argument("--out_prefix", default="rg_resid")
    a = ap.parse_args()

    if a.smoke:
        smoke(); return

    if a.stage == "residual" and not (a.mlp_ckpt and a.mlp_stats):
        ap.error("--stage residual needs --mlp_ckpt and --mlp_stats")

    import pandas as pd
    df = clean_dataframe(pd.read_parquet(a.parquet))
    d = prepare_arrays_seq(df, a.frames, max_len=a.max_len)

    run_training(d, epochs=a.epochs, base_epochs=a.base_epochs, stage=a.stage,
                 batch_size=a.batch_size, n_frames=a.n_frames, n_comp=a.n_comp,
                 width=a.width, depth=a.depth, lr=a.lr, base_lr=a.base_lr,
                 base_lr_mult=a.base_lr_mult, embed_dim=a.embed_dim, hidden=a.hidden,
                 gru_layers=a.gru_layers, head_width=a.head_width, dropout=a.dropout,
                 res_scale=a.res_scale, channels=tuple(a.channels.split(",")),
                 cond_globals=not a.no_cond_globals, wd=a.wd, cosine=not a.no_cosine,
                 val_frac=a.val_frac, seed=a.seed,
                 lam_mean=a.lam_mean, lam_std=a.lam_std, lam_skew=a.lam_skew,
                 lam_kurt=a.lam_kurt, lam_ent=a.lam_ent, lam_res=a.lam_res,
                 lam_group=a.lam_group, res_thresh=a.res_thresh, prune_warmup=a.prune_warmup,
                 mlp_ckpt=a.mlp_ckpt, mlp_stats=a.mlp_stats, out_prefix=a.out_prefix)


if __name__ == "__main__":
    main()