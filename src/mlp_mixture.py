from __future__ import annotations
import argparse
from typing import Dict, Any

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import equinox as eqx
import optax

# ══════════════════════════════════════════════════════════════════════════════
#  Global Features -> Dense MLP -> P(Rg) via Log-Normal Mixture Density
#
#  This architecture explicitly targets the "BiGRU collapse" hypothesis by
#  bypassing sequence pattern learning entirely. It uses pre-computed 
#  global physics descriptors and sequence lengths as direct inputs to a 
#  fully connected MLP.
# ══════════════════════════════════════════════════════════════════════════════

GLOBAL_NAMES = ("scd", "shd", "net_charge", "fcr", "mean_hydro",
                "charge_asym", "sbcs", "s_q", "s_all")
N_GLOBALS = len(GLOBAL_NAMES)          # 9

# Mixture head constraint hyper-parameters
SIGMA_MIN    = 0.01
SIGMA_MAX    = 0.60
DELTA_MAX    = 1.00     
SIGMA_INIT   = 0.15
DELTA_SPREAD = 0.25

class RgMLPNet(eqx.Module):
    """Dense MLP prioritizing composition summaries over sequence order."""
    mlp: eqx.nn.MLP
    n_comp: int = eqx.field(static=True)
    flory_nu: float = eqx.field(static=True)
    flory_log_a: float = eqx.field(static=True)

    def __init__(self, *, key, flory_nu, flory_log_a, 
                 n_globals=N_GLOBALS, width=128, depth=3, n_comp=6):
        keys = jax.random.split(key, 2)
        
        # Input layer expects 9 global descriptors + 1 log(L) feature = 10
        self.mlp = eqx.nn.MLP(
            in_size=n_globals + 1,
            out_size=3 * n_comp,
            width_size=width,
            depth=depth,
            activation=jax.nn.gelu,
            key=keys[0]
        )
        
        # Initialize mixture head parameter constraints
        s0 = (SIGMA_INIT - SIGMA_MIN) / (SIGMA_MAX - SIGMA_MIN)
        raw_sigma_0 = float(np.log(s0 / (1.0 - s0)))
        
        if n_comp == 1:
            delta_targets = jnp.zeros(1)
        else:
            delta_targets = jnp.linspace(-DELTA_SPREAD, DELTA_SPREAD, n_comp)
        raw_delta_0 = jnp.arctanh(delta_targets / DELTA_MAX)

        bias_init = jnp.concatenate([
            jnp.zeros(n_comp),                       # Equal logit_pi initialization
            raw_delta_0,                             # Distinct component spreads
            jnp.full((n_comp,), raw_sigma_0),        # Baseline sigma scaling
        ])
        
        # Apply specialized bias configurations to the final dense transformation layer
        self.mlp = eqx.tree_at(lambda m: m.layers[-1].bias, self.mlp, bias_init)
        self.mlp = eqx.tree_at(lambda m: m.layers[-1].weight, self.mlp, self.mlp.layers[-1].weight * 0.01)

        self.n_comp = n_comp
        self.flory_nu = float(flory_nu)
        self.flory_log_a = float(flory_log_a)

    def __call__(self, glob, logL):
        # Concatenate standardized features with sequence scale parameter
        x = jnp.concatenate([glob, logL[None]])
        out = self.mlp(x)

        K = self.n_comp
        logit_pi = out[:K]
        delta_mu = DELTA_MAX * jnp.tanh(out[K:2 * K])
        sigma    = SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * jax.nn.sigmoid(out[2 * K:])

        # Anchor using the empirical Flory scaling law prediction
        anchor = self.flory_log_a + self.flory_nu * logL
        mu = anchor + delta_mu

        return logit_pi, mu, sigma

# ══════════════════════════════════════════════════════════════════════════════
# Log-Normal Mixture Operations (Vectorized over Batches via vmap)
# ══════════════════════════════════════════════════════════════════════════════
_LOG_2PI = float(np.log(2.0 * np.pi))
_GH_N = 64
_gh_x, _gh_w = np.polynomial.hermite.hermgauss(_GH_N)
GH_X = jnp.asarray(_gh_x * np.sqrt(2.0), dtype=jnp.float32)
GH_W = jnp.asarray(_gh_w / np.sqrt(np.pi), dtype=jnp.float32)

def mixture_log_prob(logit_pi, mu, sigma, r):
    r = jnp.clip(r, 1e-8)
    y = jnp.log(r)
    log_pi = jax.nn.log_softmax(logit_pi)
    z = (y[:, None] - mu[None, :]) / sigma[None, :]
    log_comp = -0.5 * z ** 2 - jnp.log(sigma)[None, :] - 0.5 * _LOG_2PI
    log_py = logsumexp(log_pi[None, :] + log_comp, axis=-1)
    return log_py - y

def mixture_moments(logit_pi, mu, sigma):
    pi = jax.nn.softmax(logit_pi)
    r = jnp.exp(mu[:, None] + sigma[:, None] * GH_X[None, :])
    w = pi[:, None] * GH_W[None, :]

    m1  = jnp.sum(w * r)
    d   = r - m1
    var = jnp.sum(w * d ** 2)
    sd  = jnp.sqrt(jnp.clip(var, 1e-12))
    m3  = jnp.sum(w * d ** 3)
    m4  = jnp.sum(w * d ** 4)
    skew   = m3 / (sd ** 3 + 1e-12)
    exkurt = m4 / (sd ** 4 + 1e-12) - 3.0
    return m1, sd, skew, exkurt

def mixture_weight_entropy(logit_pi):
    log_pi = jax.nn.log_softmax(logit_pi)
    return -jnp.sum(jnp.exp(log_pi) * log_pi)

batch_log_prob = jax.vmap(mixture_log_prob)
batch_moments  = jax.vmap(mixture_moments)

# ══════════════════════════════════════════════════════════════════════════════
# Core Loss Optimization Protocol
# ══════════════════════════════════════════════════════════════════════════════
def loss_fn(model, glob, logL, frames, t_mean, t_std, t_skew, t_exkurt,
            lam_mean=0.1, lam_std=0.0, lam_skew=0.001, lam_kurt=0.001, lam_ent=0.0):
    logit_pi, mu, sigma = jax.vmap(model)(glob, logL)
    nll = -jnp.mean(batch_log_prob(logit_pi, mu, sigma, frames))

    loss = nll
    zero = jnp.zeros(())
    aux = dict(nll=nll, l_mean=zero, l_std=zero, l_skew=zero, l_kurt=zero, ent=zero)

    if lam_mean or lam_std or lam_skew or lam_kurt:
        pm, psd, psk, pku = batch_moments(logit_pi, mu, sigma)
        l_mean = jnp.mean(jnp.abs(pm  - t_mean) / (t_mean + 1e-6))
        l_std  = jnp.mean(jnp.abs(psd - t_std)  / (t_std  + 1e-6))
        l_skew = jnp.mean(jnp.abs(psk - t_skew))
        l_kurt = jnp.mean(jnp.abs(pku - t_exkurt))
        loss = (loss + lam_mean * l_mean + lam_std * l_std
                     + lam_skew * l_skew + lam_kurt * l_kurt)
        aux.update(l_mean=l_mean, l_std=l_std, l_skew=l_skew, l_kurt=l_kurt)

    if lam_ent:
        ent = jnp.mean(jax.vmap(mixture_weight_entropy)(logit_pi))
        loss = loss - lam_ent * ent
        aux.update(ent=ent)

    return loss, aux

@eqx.filter_jit
def train_step(model, opt_state, optimizer, batch, lams):
    glob, logL, frames, t_mean, t_std, t_skew, t_exkurt = batch
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, glob, logL, frames, t_mean, t_std, t_skew, t_exkurt, **lams
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux

@eqx.filter_jit
def eval_batch(model, glob, logL, frames):
    logit_pi, mu, sigma = jax.vmap(model)(glob, logL)
    nll = -jnp.mean(batch_log_prob(logit_pi, mu, sigma, frames))
    pm, psd, psk, pku = batch_moments(logit_pi, mu, sigma)
    ent = jnp.mean(jax.vmap(mixture_weight_entropy)(logit_pi))
    return nll, pm, psd, psk, pku, ent

# ══════════════════════════════════════════════════════════════════════════════
# Pipelines & Processing Datasets
# ══════════════════════════════════════════════════════════════════════════════
def clean_dataframe(df, frame_threshold=10_000, max_len=400, min_len=20):
    if "frame_row" not in df.columns:
        raise KeyError("Parquet schema variant mismatch. Ensure correct file configuration targets.")
    df = df[df["total_frames"] >= frame_threshold].copy()
    logN, logRg = np.log(df["seq_length"]), np.log(df["calculated_rg_mean"])
    nu, log_a = np.polyfit(logN, logRg, 1)
    resid = logRg - (log_a + nu * logN)
    df = df[np.abs(resid) <= 3 * resid.std()]
    df = df[(df["seq_length"] <= max_len) & (df["seq_length"] >= min_len)]
    print(f"[clean] Kept {len(df)} records | Empirical Flory scaling exponent nu={nu:.3f}")
    return df.reset_index(drop=True)

def prepare_arrays(df, frames_path) -> Dict[str, Any]:
    # Extract structural summaries using global preprocessing definitions from original script
    from bigru_mixture import sequence_globals, AA_TO_ID
    
    N = len(df)
    print(f"[prep] N={N} | Extracting global feature descriptors: {GLOBAL_NAMES}")

    glob = np.zeros((N, N_GLOBALS), np.float32)
    logL = np.zeros(N, np.float32)

    for i, seq in enumerate(df["sequence"]):
        tokens = np.array([AA_TO_ID.get(a, 0) for a in seq.upper()], dtype=np.int32)
        glob[i] = sequence_globals(tokens)
        logL[i] = np.log(len(tokens) + 1e-6)

    if isinstance(frames_path, np.ndarray):
        frames_mm = frames_path
    else:
        frames_mm = np.load(frames_path, mmap_mode="r")
        
    frame_rows = df["frame_row"].to_numpy().astype(np.int64)
    K_STORE = frames_mm.shape[1]

    t_mean   = df["calculated_rg_mean"].to_numpy().astype(np.float32)
    t_std    = df["calculated_rg_std"].to_numpy().astype(np.float32)
    t_skew   = df["calculated_rg_skew"].to_numpy().astype(np.float32)
    t_exkurt = df["calculated_rg_kurtosis"].to_numpy().astype(np.float32)

    logN  = np.log(df["seq_length"].to_numpy().astype(np.float64))
    logRg = np.log(t_mean.astype(np.float64))
    nu, log_a = np.polyfit(logN, logRg, 1)
    print(f"[prep] Flory scaling model alignment: log Rg = {log_a:.4f} + {nu:.4f} * log N")

    return dict(glob=glob, logL=logL, frames=frames_mm, frame_rows=frame_rows, K_store=K_STORE,
                mean=t_mean, std=t_std, skew=t_skew, exkurt=t_exkurt, nu=float(nu), log_a=float(log_a))

def standardize(data, tr_idx):
    g = data["glob"][tr_idx]
    g_mu = g.mean(0)
    g_sd = g.std(0) + 1e-6
    data["glob"] = (data["glob"] - g_mu) / g_sd

    data["_stats"] = dict(g_mu=g_mu, g_sd=g_sd, nu=np.float32(data["nu"]), 
                          log_a=np.float32(data["log_a"]), global_names=np.array(GLOBAL_NAMES))
    return data

def draw_frames(d, idx, n_frames, rng):
    rows = d["frame_rows"][idx]
    block = np.asarray(d["frames"][rows])
    cols = rng.integers(0, d["K_store"], size=(len(idx), n_frames))
    return np.take_along_axis(block, cols, axis=1).astype(np.float32)

def make_batch(d, idx, n_frames, rng):
    j = jnp.array
    return (j(d["glob"][idx]), j(d["logL"][idx]),
            j(draw_frames(d, idx, n_frames, rng)),
            j(d["mean"][idx]), j(d["std"][idx]),
            j(d["skew"][idx]), j(d["exkurt"][idx]))

# ══════════════════════════════════════════════════════════════════════════════
# Model Optimization & Evaluation Routing
# ══════════════════════════════════════════════════════════════════════════════
def run_training(d, *, epochs=40, batch_size=64, n_frames=1024, n_comp=6,
                 width=128, depth=3, lr=5e-4, val_frac=0.15, seed=42,
                 lam_mean=0.0, lam_std=0.0, lam_skew=0.0, lam_kurt=0.0, lam_ent=0.0,
                 out_prefix="rg_mlp"):
    N = len(d["glob"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = int(N * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    d = standardize(d, tr_idx)

    model = RgMLPNet(key=jax.random.PRNGKey(seed), flory_nu=d["nu"], flory_log_a=d["log_a"],
                     width=width, depth=depth, n_comp=n_comp)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"[model] MLP Architecture Width={width}, Depth={depth} | total parameter dimensions: {n_params:,}")

    opt = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr, weight_decay=1e-4),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    lams = dict(lam_mean=lam_mean, lam_std=lam_std, lam_skew=lam_skew, lam_kurt=lam_kurt, lam_ent=lam_ent)
    use_moments = any([lam_mean, lam_std, lam_skew, lam_kurt])

    val_rng = np.random.default_rng(0)
    val_frames = jnp.array(draw_frames(d, val_idx, 4096, val_rng))
    val_glob = jnp.array(d["glob"][val_idx])
    val_logL = jnp.array(d["logL"][val_idx])

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

        vnll, pm, psd, psk, pku, vent = eval_batch(model, val_glob, val_logL, val_frames)
        vnll = float(vnll)

        print(f"ep {ep+1:02d}/{epochs} | train loss {acc['loss']/steps:8.4f} "
              f"(nll {acc['nll']/steps:7.4f}) | val nll {vnll:7.4f} | "
              f"val MAE: mean {mae(pm, d['mean'][val_idx]):.3f}  "
              f"std {mae(psd, d['std'][val_idx]):.3f}  "
              f"skew {mae(psk, d['skew'][val_idx]):.3f}  "
              f"kurt {mae(pku, d['exkurt'][val_idx]):.3f}  "
              f"| pi-ent {float(vent):.2f}/{max_ent:.2f}")

        if use_moments:
            print(f"          Components breakdown -> mean {acc['l_mean']/steps:.4f}  "
                  f"std {acc['l_std']/steps:.4f}  skew {acc['l_skew']/steps:.4f}  "
                  f"kurt {acc['l_kurt']/steps:.4f}")

        if not np.isfinite(vnll):
            print("Non-finite optimization values detected. Terminating routing session.")
            break

        if vnll < best:
            best = vnll
            eqx.tree_serialise_leaves(f"{out_prefix}_model.eqx", model)

    np.savez(f"{out_prefix}_stats.npz", n_comp=np.int32(n_comp), width=np.int32(width),
             depth=np.int32(depth), **d["_stats"])
    return model

def smoke():
    print("── MLP MIXTURE SMOKE RUN ──")
    from bigru_mixture import synthetic_data
    df, frames = synthetic_data(P=20, K_store=1000)
    df = clean_dataframe(df, frame_threshold=500, min_len=10)
    d = prepare_arrays(df, frames)
    
    run_training(d, epochs=2, batch_size=4, n_frames=256, n_comp=3, val_frac=0.2,
                 width=32, depth=2, lam_skew=0.01, lam_kurt=0.01, out_prefix="smoke_mlp")
    print("── SMOKE STATUS OK ──")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/idp_ml_dataset.parquet")
    ap.add_argument("--frames",  default="data/idp_rg_frames.npy")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n_frames", type=int, default=1024)
    ap.add_argument("--n_comp", type=int, default=6)
    ap.add_argument("--width", type=int, default=128, help="MLP hidden feature size")
    ap.add_argument("--depth", type=int, default=3, help="Number of dense hidden layers")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lam_mean", type=float, default=0.0)
    ap.add_argument("--lam_std",  type=float, default=0.0)
    ap.add_argument("--lam_skew", type=float, default=0.0)
    ap.add_argument("--lam_kurt", type=float, default=0.0)
    ap.add_argument("--lam_ent",  type=float, default=0.0)
    ap.add_argument("--out_prefix", default="rg_mlp")
    a = ap.parse_args()

    if a.smoke:
        smoke(); return

    import pandas as pd
    df = clean_dataframe(pd.read_parquet(a.parquet))
    d = prepare_arrays(df, a.frames)
    run_training(d, epochs=a.epochs, batch_size=a.batch_size, n_frames=a.n_frames, 
                 n_comp=a.n_comp, width=a.width, depth=a.depth, lr=a.lr,
                 lam_mean=a.lam_mean, lam_std=a.lam_std, lam_skew=a.lam_skew, 
                 lam_kurt=a.lam_kurt, lam_ent=a.lam_ent, out_prefix=a.out_prefix)

if __name__ == "__main__":
    main()