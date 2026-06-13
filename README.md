```text 
project/
│
├── configs/                          # CONTROL PANEL
│   ├── sequences.yaml                # Full sequence database: ID -> AA string + metadata
│   ├── physics.yaml                  # Named parameter sets (default, cold)
│   └── config.yaml                   # control panel, choose sequences to run, steps, simulation parameters     
│                                
├── R15_R17_full_run_0414/
│   ├── data/                         # processed data (Rg, anything else needed for ml) 
│   ├── job_manifest.csv              # stores sequence,task id, parameters, trajectory file location
│   ├── logs/                         # per job info, avoids race conditions, metadata. 
│   │   ├── R15_task_1.json
│   │   ├── R15_task_2.json
│   │   ├── R17_task_3.json
│   │   └── R17_task_4.json
│   └── trajectories/                # raw output files
│       ├── R15_traj_1.gsd
│       ├── R15_traj_2.gsd
│       ├── R17_traj_3.gsd
│       └── R17_traj_4.gsd
│
├── src/                              # Core 
│   ├── simulation.py                 # HOOMD script, reads job_manifest(only one line), writes logs and trajectories, called by run_local or slurm
│   ├── analysis.py                   # Rg, contacts, asphericity, distributions, etc.
│   ├── run_local.py                  # reads jobs_manifest, calls simulation per task
│   └── gen_jobs.py                   # reads config.yaml, creates run directory and sub directories, creates job_manifest.csv
│
├── slurm/
│   └── submit_array.sh               # Indexes into manifest.csv via $SLURM_ARRAY_TASK_ID
│
├── environment.yml
└── README.md
project/
│
├── configs/                          # CONTROL PANEL
│   ├── sequences.yaml                # Full sequence database: ID -> AA string + metadata
│   ├── physics.yaml                  # Named parameter sets (default, cold)
│   └── config.yaml                   # control panel, choose sequences to run, steps, simulation parameters     
│                                
├── R15_R17_full_run_0414/
│   ├── data/                         # processed data (Rg, anything else needed for ml) 
│   ├── job_manifest.csv              # stores sequence,task id, parameters, trajectory file location
│   ├── logs/                         # per job info, avoids race conditions, metadata. 
│   │   ├── R15_task_1.json
│   │   ├── R15_task_2.json
│   │   ├── R17_task_3.json
│   │   └── R17_task_4.json
│   └── trajectories/                # raw output files
│       ├── R15_traj_1.gsd
│       ├── R15_traj_2.gsd
│       ├── R17_traj_3.gsd
│       └── R17_traj_4.gsd
│
├── src/                              # Core 
│   ├── simulation.py                 # HOOMD script, reads job_manifest(only one line), writes logs and trajectories, called by run_local or slurm
│   ├── analysis.py                   # Rg, contacts, asphericity, distributions, etc.
│   ├── run_local.py                  # reads jobs_manifest, calls simulation per task
│   └── gen_jobs.py                   # reads config.yaml, creates run directory and sub directories, creates job_manifest.csv
│
├── slurm/
│   └── submit_array.sh               # Indexes into manifest.csv via $SLURM_ARRAY_TASK_ID
│
├── environment.yml
└── README.md
``` `
# Sequence-to-Distribution Prediction of IDP Radius-of-Gyration Distograms

A physics-informed, multi-scale neural network (JAX + Equinox) that predicts the **full
distribution** of the radius of gyration `Rg` for an intrinsically disordered protein (IDP)
directly from its amino-acid sequence — not a fitted parametric form, but a dense histogram
("distogram") over a fixed `Rg` grid.

---

## 1. Motivation

IDPs do not fold into a single structure; they sample a broad conformational ensemble, and that
ensemble's *shape* — its skew, its heavy tails, its multi-modality — encodes the dynamical
heterogeneity that matters for function and disease. Molecular dynamics captures this faithfully
but is expensive: predicting it from sequence alone is the goal.

This project builds on a prior model from our group that predicted `Rg` and a handful of higher
moments with a fully-connected network and a biGRU, then **reconstructed** the distribution by
fitting a **Pearson Type III** (shifted-gamma) curve from the predicted mean, variance, and skew.
That approach is accurate for the mean (R² ≈ 0.95) and variance (R² ≈ 0.89) but degrades sharply
for higher moments (skew R² ≈ 0.53, kurtosis R² ≈ −0.06, fifth moment R² ≈ 0.23).

### The core thesis of this project

The prior model's higher-moment failure is **structural, not a matter of tuning**. A gamma family
has a locked relationship between its moments — excess kurtosis is forced to equal `1.5 × skew²`.
So even when kurtosis is predicted as a separate output, that number never reaches the
reconstructed curve: the parametric family literally cannot represent an arbitrary (skew, kurtosis)
pair. This is a ceiling baked into the functional form.

**Predicting the histogram directly removes that ceiling.** A 450-bin distribution trained against
KL divergence can represent any shape — heavy tails, leptokurtosis, even bimodality — with no
coupling between low- and high-order moments. That is the contribution:

> Direct binned distributional prediction removes the parametric bottleneck that caps
> higher-moment fidelity in fitted-distribution approaches.

The model is **not** expected to win on the mean — a plain physics-feature FCNN already nails that.
The win must be demonstrated on **shape**: KL divergence and the 3rd/4th/5th moments, head-to-head
on comparable proteins.

---

## 2. Data

- **~4,000 IDP sequences**, lengths spanning roughly **60–400** residues.
- Per IDP: **~1,000,000** `Rg` frames from Brownian/molecular dynamics.
- Each frame set is smoothed with a **Gaussian KDE** and evaluated on a **fixed shared grid**
  `np.linspace(0, 150, 450)`. The fixed grid is essential for KL comparability across sequences
  and with prior work.
- Precomputed per IDP (used as labels and/or conditioning): `Rg` mean, std, skew, kurtosis,
  `<R_end²>`, persistence length.

### Two preprocessing requirements

1. **Targets must be probability vectors.** `gaussian_kde(...).evaluate(grid)` returns a *density*
   (integrates to ~1, sums to ~3 at this grid spacing). Normalize once so each target sums to 1:
   ```python
   target_dist = p / p.sum()
   ```
   Mostly-empty bins are harmless: `0 · log q = 0` contributes nothing to cross-entropy.

2. **Topology label comes for free** from the moments:
   ```python
   rg_sq_mean   = rg_mean**2 + rg_std**2        # <Rg²> = mean² + variance
   target_ratio = mean_r_end_sq / rg_sq_mean    # <R_end²> / <Rg²>  =  (R_end/Rg)²
   ```

### Heavy tails: physical vs. artifact

Some distributions have long tails. **Do not blanket-trim them.** Forward KL is mass-covering and
*should* chase physical tails — they are exactly the extended-conformation heterogeneity we want to
capture. Trim only clear sampling artifacts: an isolated clump of frames far from the bulk,
implausible given the chain's contour length. With ~1M frames per IDP the KDE is otherwise very
trustworthy. Inspect the worst offenders at the frame level before the KDE step.

---

## 3. Input Representation

Per residue, the model sees three sources of information:

| Source | Shape | Description |
|---|---|---|
| Learned AA embedding | `(L, 16)` | trainable, 21 tokens (20 AAs + padding) |
| Physicochemical params | `(L, 5)` | mass, charge, σ, HPS1, HPS2 (z-scored) |
| Per-scale physics features | `(S, 9, L)` | precomputed, one set per scale (see below) |

The 9 physics features are window-aggregated, non-linear sequence descriptors computed at a matched
window size per scale: **SCD** (sequence charge decoration), net charge, mean hydropathy, **FCR**
(fraction charged residues), **ShD** (hydropathy decoration), charge asymmetry, **SBCS**,
charge entropy `S_q`, and total entropy `S_all`.

Global scalars used for conditioning: `log L` (computed from the mask) and global SCD / FCR.

Variable length is handled by **padding to a per-batch bucket length and carrying a 0/1 mask** from
the very first convolution all the way through pooling.

---

## 4. Architecture

```
ids (L,) ──embed──┐
raw params (L,5) ─┴─concat─► (in_ch, L)
                              │
        ┌─────────────────────┼─────────────────────┐         one branch per scale s
        ▼                     ▼                       ▼
   Conv1d(k=3)           Conv1d(k=5)   ...      Conv1d(k=15)    'same' padding (odd kernels)
        │                     │                       │
   concat physics_s      concat physics_s        concat physics_s     ◄── (9, L) matched window
        │                     │                       │
     1×1 conv              1×1 conv                1×1 conv             learnable physics fusion
        │                     │                       │
  masked attn-pool      masked attn-pool        masked attn-pool       → scale token v_s (C,)
        └─────────────────────┴───────────────────────┘
                              ▼
                  tokens (S, C) ── attention ACROSS scales ──► g (C,)
                              │        (softmax weights = per-scale importance)
                  FiLM_global( c = MLP([log L, global SCD…]) )
                              ▼
                          trunk ──► z (C,)
                       ┌──────┴───────────────┐
                       ▼                       ▼
              TOPOLOGY HEAD            DISTOGRAM HEAD
              MLP → softplus           FiLM_disto([c, topo_hidden])
              → (R_end/Rg)²            → MLP → logits over 450 Rg bins
```

### Stage-by-stage rationale

- **Multi-scale CNN channels.** Each kernel size is a "scale detector." Short kernels capture local
  charge/hydropathy patterning; long kernels capture block-level organization. This is a strong,
  sample-efficient inductive bias for a 4k-sequence dataset.

- **Physics fusion via 1×1 conv.** Rather than treating the precomputed physics as an *additive*
  correction, we concatenate it channel-wise with the learned conv output (at matched receptive
  field) and let a 1×1 conv learn the mixing. The network decides how much to trust each physics
  channel at each scale.

- **Pool-then-attend (within scale, then across scales).** We pool over length *within* each scale
  (masked attention pooling), then run attention *across* the few scale tokens. Because there are
  only a handful of scales, the across-scale attention weights are directly interpretable as
  per-scale importances — the "which scale matters" signal you can read off and report.

- **Dual FiLM conditioning.** FiLM #1 injects global conditioning (`log L`, global SCD) into the
  aggregated representation. FiLM #2 modulates the distogram branch using *both* the global
  conditioning and the **topology head's hidden state** — so the topology head genuinely shapes the
  distogram. The topology head is fed by its *predicted* representation (not ground truth) to avoid
  train/test mismatch.

- **Two heads.** The topology head predicts `(R_end/Rg)²` (a polymer-physics shortcut, kept positive
  via `softplus`); the distogram head produces logits over the 450 `Rg` bins.

---

## 5. Loss Functions

The total loss has three terms:

```
L = CrossEntropy(softmax(logits), p_MD)              # main objective
  + λ_topo · Huber( log(R_end/Rg)²_pred , log(...)_MD )
  + λ_mom  · moment_matching(predicted histogram, rg_mean, rg_std)
```

- **Cross-entropy ≡ forward KL.** `KL(p_MD ‖ p_pred) = CE(p_MD, p_pred) − H(p_MD)`. The entropy
  term is constant w.r.t. the model, so minimizing CE minimizes the forward KL. CE is the
  numerically nicer form. Forward (mass-covering) KL is the right choice here: it penalizes
  predicting ~0 mass where MD has mass, forcing the model to populate physical tails.

- **Topology head — Huber on the log ratio.** Working in `log` keeps the scale sane and places the
  ideal-chain reference (`(R_end/Rg)² ≈ 6`) naturally; Huber is robust to outliers.

- **Moment-matching auxiliary.** The fixed 0–150 grid is wide relative to any single IDP's
  distribution, so most of the learning signal is just *locating* the bump — which is dominated by
  Flory scaling `Rg ~ L^ν`. Since `rg_mean` and `rg_std` are precomputed, we pin the predicted
  histogram's mean and width to them with a **relative** (dimensionless) penalty:
  ```
  ((mean(p̂) − rg_mean)/rg_mean)²  +  ((std(p̂) − rg_std)/rg_std)²
  ```
  This makes location/width nearly free to learn while leaving cross-entropy to resolve fine shape.
  Start at `λ_mom ≈ 0.1`; it is a regularizer, anneal it down once shapes are good.

**Optional:** a small total-variation penalty on the predicted histogram discourages jaggedness
relative to the smooth KDE targets.

### What the loss values mean

- A model that "knows nothing" outputs a near-uniform softmax → CE ≈ `ln(450) ≈ 6.11`. That is the
  **random baseline**; any working model must drop below it.
- CE does **not** go to zero for a perfect model. Its floor is the **entropy** `H(p_MD)` of the
  true distribution (a few nats). Track `KL = CE − H(p_MD)` for a metric with a clean zero floor.

---

## 6. Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| AA representation | learned embedding **+** physicochemical params | trainable features plus hard physics priors; safe at 4k sequences |
| Output form | direct 450-bin histogram | removes the gamma-family moment ceiling (the thesis) |
| Grid | fixed `linspace(0,150,450)` | KL comparability across sequences and with prior work |
| Scale alignment | **odd** kernels + `padding=(k-1)/2` | all scales stay length `L` so they fuse/attend cleanly |
| Physics fusion | concat + 1×1 conv | learnable mixing beats hand-set additive correction |
| Attention | pool-then-attend across scales | interpretable per-scale importance weights |
| Modulation | topology hidden → FiLM on distogram | learned conditioning bottleneck; robust, simple |
| Length handling | bucket + 0-mask through whole pipeline | static shapes for JAX, minimal wasted compute |
| Normalization | z-score inputs (+ LayerNorm safety net) | mass (~100) would otherwise swamp charge (~1) |
| Regularization | dropout + L2/weight decay, validation KL | small dataset, expressive model → overfitting risk |

---

## 7. Training

- **Hardware:** a single consumer GPU (e.g. RTX 3060 Ti) is more than enough. The model is ~0.3 M
  parameters; an epoch over 4k sequences is seconds, and the first step's slowness is one-time XLA
  compilation, not training cost.
- **Real bottleneck:** the `O(L²)` physics descriptors (SCD/ShD/SBCS) on the ~400-residue chains.
  **Precompute the physics stack once and cache it to disk** — it is static model input, not
  something learned. Recomputing it every epoch is the only thing that will feel slow.
- **Bucketing caveat:** different buckets have different shapes → JAX recompiles per unique shape.
  Keep buckets coarse (a small fixed set of length bins) so you pay a few compilations up front,
  then reuse cached kernels.

### Sanity checks (do these first)

1. **Shape/plumbing test.** Run the module's `__main__` smoke test — confirms shapes line up,
   gradients flow, one optimizer step runs without NaNs. (Proves nothing about *learning*.)
2. **Overfit a tiny batch of real data.** Take 4–8 real sequences, train a couple thousand steps
   with regularization off. A correct model drives CE down to the entropy floor and the moment
   terms toward zero. **If it cannot overfit 8 examples, there is a bug** — usually masking
   (padding leaking into pooling), missing input normalization, or a bad learning rate.
3. **Eyeball overlays.** Plot a few predicted histograms on top of their MD targets. Numbers say
   it's converging; the overlay says it's converging to the *right shape*.

---

## 8. Evaluation (how to make the paper's case)

The headline result is a **moments-from-histogram comparison** against the prior model's table.

1. Compute V, S, K, M5 from the predicted histogram:
   `mean = Σ pᵢ gᵢ`, central moments via `Σ pᵢ (gᵢ − mean)ⁿ`.
2. Tabulate against simulated values using the **same columns and proteins** as the prior work's
   Table I. The win is showing materially smaller K and M5 error.
3. Report **KL divergence** per protein, and specifically on the prior model's hardest cases
   (CspTm, ProTaN, and similar). Note that very short proteins (e.g. L=24) fall below this dataset's
   length range — compare on overlapping lengths.

Generate this comparison **early, on a partially trained model**, to confirm you are actually
beating the baseline on higher moments before investing in the full writeup.

---

## 9. Files

- `rg_distogram_model.py` — the full Equinox model, loss, and a runnable smoke test.
  - `align_valid_to_full` / `build_physics_stack` — align per-scale physics (valid conv) back to
    length `L` and assemble the `(B, S, 9, L)` tensor.
  - `FiLM`, `ScaleEncoder` — building blocks.
  - `RgDistogramNet` — the model (`n_bins=450`, configurable kernel sizes).
  - `loss_fn` — cross-entropy + topology Huber + moment matching.
  - `train_step` — JIT-compiled optimizer step (optax AdamW).

---

## 10. Limitations and Future Work

- **Higher moments are intrinsically noisy** even from MD; the model can only be as good as the
  KDE targets. Distinguish physical tails from artifacts at the frame level.
- **4k sequences is small** for an expressive model — resist scaling capacity; lean on the physics
  inductive bias and regularization.
- **Length range (60–400)** is wide; if performance is uneven across lengths, consider stratified
  validation by length bucket.
- **Inherited error** the neural network is fully trained on data generated using MD and a coarse grained model
for this reason it closely acts like a surrogate rather than fully representing reality. Corrections based on experimental
data could be implemented on another loss head in the future. 
- Possible extensions: third/fourth-moment matching terms (hold off — higher moments from a
  histogram are noisy and can destabilize), persistence length and `config_avg_Rg` as extra
  conditioning features or auxiliary targets, and scheduled sampling between predicted and
  ground-truth topology if early training is unstable.