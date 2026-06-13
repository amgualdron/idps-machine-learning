"""
Rg-distogram network for IDPs (JAX + Equinox).
 
Pipeline
--------
ids (L,) ──embed──┐
raw params (L,5) ─┴─concat─► (in_ch, L) ─┐
                                          ├─ per scale s:  Conv1d(k_s, 'same') ─► (C,L)
physics per scale (S,9,L) ────────────────┘        concat physics ─► 1x1 conv ─► (C,L)
                                                   masked attention-pool over L ─► token v_s (C,)

tokens (S,C) ─ attention ACROSS scales ─► g (C,)   [softmax weights = per-scale importance]
g ─ FiLM_global(c) ─► trunk ─► z (C,)
z ─► topology head ─► (Ree/Rg)^2  (scalar, softplus)  + hidden th
z ─ FiLM_disto([c, th]) ─► distogram MLP ─► logits over Rg bins

Loss = CrossEntropy(softmax(logits), p_MD)  [ == forward KL up to a const ]
     + lambda * Huber( log t_pred , log t_MD )
"""

from typing import List
import jax
import jax.numpy as jnp
import equinox as eqx


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helper: align a VALID-conv physics output back to length L.
# Your `conv` produces (L - k + 1) windows. With an odd kernel we pad
# symmetrically so it lines up with a learnable Conv1d using padding=(k-1)//2.
# ─────────────────────────────────────────────────────────────────────────────
def align_valid_to_full(phys_bnf: jnp.ndarray, kernel_size: int, L: int) -> jnp.ndarray:
    """phys_bnf: (B, n_windows, n_phys)  ->  (B, n_phys, L)  (channels-first)."""
    pad_total = L - phys_bnf.shape[1]            # == k - 1
    left = (kernel_size - 1) // 2
    right = pad_total - left
    padded = jnp.pad(phys_bnf, ((0, 0), (left, right), (0, 0)))   # (B, L, n_phys)
    return padded.transpose(0, 2, 1)             # (B, n_phys, L)


def build_physics_stack(process_batch_features_fn, ids, features, mask,
                        kernel_sizes, L):
    """
    Wraps YOUR process_batch_features over several scales and returns
    (B, S, n_phys, L), aligned and channels-first, ready for the model.
    """
    stacks = []
    for k in kernel_sizes:
        feats_k, _ = process_batch_features_fn(ids, features, mask, k)  # (B, L-k+1, 9)
        stacks.append(align_valid_to_full(feats_k, k, L))              # (B, 9, L)
    return jnp.stack(stacks, axis=1)                                   # (B, S, 9, L)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────
class FiLM(eqx.Module):
    """Feature-wise linear modulation: h -> (1+gamma)*h + beta, conditioned on c."""
    net: eqx.nn.MLP
    dim: int = eqx.field(static=True)

    def __init__(self, cond_dim, feat_dim, *, key, width=64):
        self.net = eqx.nn.MLP(cond_dim, 2 * feat_dim, width_size=width,
                              depth=2, activation=jax.nn.gelu, key=key)
        self.dim = feat_dim

    def __call__(self, h, c):
        gb = self.net(c)
        gamma, beta = gb[: self.dim], gb[self.dim:]
        return (1.0 + gamma) * h + beta          # 1+gamma => identity at init


class ScaleEncoder(eqx.Module):
    """One scale: learnable conv over residues, fuse precomputed physics, pool over L."""
    norm: eqx.nn.LayerNorm
    conv: eqx.nn.Conv1d
    fuse: eqx.nn.Conv1d                          # 1x1 conv = per-position mixing
    attn_query: jax.Array                        # learned pooling query (C,)

    def __init__(self, in_ch, n_phys, hidden_ch, kernel_size, *, key):
        kn, kc, kf, kq = jax.random.split(key, 4)
        assert kernel_size % 2 == 1, "use odd kernel sizes so 'same' padding aligns"
        pad = (kernel_size - 1) // 2
        self.norm = eqx.nn.LayerNorm(in_ch)
        self.conv = eqx.nn.Conv1d(in_ch, hidden_ch, kernel_size, padding=pad, key=kc)
        self.fuse = eqx.nn.Conv1d(hidden_ch + n_phys, hidden_ch, 1, key=kf)
        self.attn_query = jax.random.normal(kq, (hidden_ch,)) * 0.02

    def __call__(self, x, phys, mask):
        # x: (in_ch, L)  phys: (n_phys, L)  mask: (L,)
        x = jax.vmap(self.norm, in_axes=1, out_axes=1)(x)     # safety-net normalization
        h = jax.nn.gelu(self.conv(x))                         # (hidden, L)
        h = jnp.concatenate([h, phys], axis=0)                # (hidden+n_phys, L)
        h = jax.nn.gelu(self.fuse(h))                         # (hidden, L)
        # masked attention pooling over the length axis
        scores = self.attn_query @ h                          # (L,)
        scores = jnp.where(mask > 0, scores, -jnp.inf)
        w = jax.nn.softmax(scores)                            # (L,)
        return h @ w                                          # (hidden,)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────
class RgDistogramNet(eqx.Module):
    embedding: eqx.nn.Embedding
    scales: List[ScaleEncoder]
    scale_query: jax.Array                       # (hidden,) attention across scales
    cond_mlp: eqx.nn.MLP
    film_global: FiLM
    trunk: eqx.nn.MLP
    topo_hidden: eqx.nn.Linear
    topo_out: eqx.nn.Linear
    film_disto: FiLM
    disto_mlp: eqx.nn.MLP
    kernel_sizes: tuple = eqx.field(static=True)

    def __init__(self, *, key,
                 n_aa=21, embed_dim=16, n_raw=5, n_phys=9,
                 hidden_ch=64, kernel_sizes=(3, 5, 9, 15),
                 n_globals=2, cond_dim=32, topo_hidden=32, n_bins=64):
        keys = jax.random.split(key, 10)
        in_ch = embed_dim + n_raw

        self.embedding = eqx.nn.Embedding(n_aa, embed_dim, key=keys[0])
        self.scales = [
            ScaleEncoder(in_ch, n_phys, hidden_ch, k, key=keys[1 + i])
            for i, k in enumerate(kernel_sizes)
        ]
        self.scale_query = jax.random.normal(keys[5], (hidden_ch,)) * 0.02
        self.kernel_sizes = tuple(kernel_sizes)

        # globals -> conditioning embedding (we append log L internally -> +1)
        self.cond_mlp = eqx.nn.MLP(n_globals + 1, cond_dim, width_size=64,
                                   depth=2, activation=jax.nn.gelu, key=keys[6])
        self.film_global = FiLM(cond_dim, hidden_ch, key=keys[7])
        self.trunk = eqx.nn.MLP(hidden_ch, hidden_ch, width_size=128,
                                depth=2, activation=jax.nn.gelu, key=keys[8])

        # heads
        kt = jax.random.split(keys[9], 4)
        self.topo_hidden = eqx.nn.Linear(hidden_ch, topo_hidden, key=kt[0])
        self.topo_out = eqx.nn.Linear(topo_hidden, 1, key=kt[1])
        self.film_disto = FiLM(cond_dim + topo_hidden, hidden_ch, key=kt[2])
        self.disto_mlp = eqx.nn.MLP(hidden_ch, n_bins, width_size=128,
                                    depth=2, activation=jax.nn.gelu, key=kt[3])

    def __call__(self, ids, raw_params, phys_stack, globals_, mask):
        # ids:(L,)  raw_params:(L,5)  phys_stack:(S,9,L)  globals_:(G,)  mask:(L,)
        emb = jax.vmap(self.embedding)(ids)                   # (L, embed_dim)
        feat = jnp.concatenate([emb, raw_params], axis=-1)    # (L, in_ch)
        feat = feat * mask[:, None]
        x = feat.T                                            # (in_ch, L)

        tokens = jnp.stack(
            [enc(x, phys_stack[s], mask) for s, enc in enumerate(self.scales)],
            axis=0,                                           # (S, hidden)
        )

        # attention ACROSS scales -> g, with interpretable per-scale weights
        scale_w = jax.nn.softmax(tokens @ self.scale_query)   # (S,)
        g = scale_w @ tokens                                  # (hidden,)

        # global conditioning (log L computed from the mask)
        logL = jnp.log(jnp.sum(mask) + 1e-6)
        c = self.cond_mlp(jnp.concatenate([globals_, logL[None]]))   # (cond_dim,)

        g = self.film_global(g, c)
        z = self.trunk(g)                                     # shared trunk

        # topology head
        th = jax.nn.gelu(self.topo_hidden(z))
        ratio = jax.nn.softplus(self.topo_out(th))[0]         # (Ree/Rg)^2 > 0

        # distogram head, modulated by globals AND topology hidden state
        zd = self.film_disto(z, jnp.concatenate([c, th]))
        logits = self.disto_mlp(zd)                           # (n_bins,)

        return logits, ratio, scale_w


# ─────────────────────────────────────────────────────────────────────────────
# Loss + training step
# ─────────────────────────────────────────────────────────────────────────────
def _huber(x, delta=1.0):
    a = jnp.abs(x)
    return jnp.where(a <= delta, 0.5 * a ** 2, delta * (a - 0.5 * delta))


def loss_fn(model, ids, raw, phys, glob, mask,
            target_dist, target_ratio,
            grid=None, target_mean=None, target_std=None,
            lam_topo=0.3, lam_mom=0.1):
    """
    grid:        (n_bins,) Rg bin centres (your np.linspace(0,150,450)).
    target_mean: (B,) MD Rg mean   (your rg_mean)   -> matches the *location*.
    target_std:  (B,) MD Rg std    (your rg_std)    -> matches the *width*.
    Pass grid=None to disable moment matching. Whether it's enabled is fixed
    per training run (a static branch under jit), so don't toggle mid-run.
    """
    logits, ratio, _ = jax.vmap(model)(ids, raw, phys, glob, mask)
    logp = jax.nn.log_softmax(logits, axis=-1)
    # forward KL(p_MD || p_pred) == cross-entropy up to an additive constant
    ce = -jnp.sum(target_dist * logp, axis=-1)                # (B,)
    loss = jnp.mean(ce)

    # topology head
    loss += lam_topo * jnp.mean(
        _huber(jnp.log(ratio + 1e-6) - jnp.log(target_ratio + 1e-6)))

    # moment matching on the predicted distribution (uses your free MD moments)
    if grid is not None and target_mean is not None:
        p = jax.nn.softmax(logits, axis=-1)                   # (B, n_bins)
        pred_mean = p @ grid                                  # (B,)
        loss += lam_mom * jnp.mean(
            ((pred_mean - target_mean) / (target_mean + 1e-6)) ** 2)
        if target_std is not None:
            pred_var = p @ (grid ** 2) - pred_mean ** 2
            pred_std = jnp.sqrt(jnp.clip(pred_var, 1e-6))
            loss += lam_mom * jnp.mean(
                ((pred_std - target_std) / (target_std + 1e-6)) ** 2)
    return loss


@eqx.filter_jit
def train_step(model, opt_state, optimizer, batch):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, *batch)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (random data) — verifies shapes flow end to end.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import optax

    key = jax.random.PRNGKey(0)
    B, L, S, n_bins, n_phys = 4, 32, 4, 450, 9       # 450 = your grid resolution
    kernel_sizes = (3, 5, 9, 15)

    model = RgDistogramNet(key=key, kernel_sizes=kernel_sizes,
                           n_bins=n_bins, n_globals=2)

    k = jax.random.PRNGKey(1)
    ids = jax.random.randint(k, (B, L), 0, 21)
    raw = jax.random.normal(k, (B, L, 5))
    phys = jax.random.normal(k, (B, S, n_phys, L))
    glob = jax.random.normal(k, (B, 2))                       # e.g. [global SCD, FCR]
    mask = (jnp.arange(L)[None, :] < jnp.array([20, 25, 32, 15])[:, None]).astype(jnp.float32)
    target = jax.nn.softmax(jax.random.normal(k, (B, n_bins)), axis=-1)   # sums to 1
    target_ratio = jnp.full((B,), 6.0)

    grid = jnp.linspace(0.0, 150.0, n_bins)                   # your GRID_POINTS
    target_mean = jnp.array([18.0, 22.0, 30.0, 14.0])         # your rg_mean
    target_std = jnp.array([4.0, 5.0, 7.0, 3.0])              # your rg_std

    logits, ratio, scale_w = jax.vmap(model)(ids, raw, phys, glob, mask)
    print("logits   :", logits.shape)        # (B, n_bins)
    print("ratio    :", ratio.shape)         # (B,)
    print("scale_w  :", scale_w.shape)       # (B, S)  <- per-scale importances

    optimizer = optax.adamw(3e-4, weight_decay=1e-4)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    batch = (ids, raw, phys, glob, mask, target, target_ratio,
             grid, target_mean, target_std)
    model, opt_state, loss = train_step(model, opt_state, optimizer, batch)
    print("loss     :", float(loss))