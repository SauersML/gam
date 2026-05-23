# Proposal: `LatentCoord` — per-row latent coordinates as a first-class gamfit parameter

Status: draft. Author: scottsauersc@gmail.com.  
Companion prototype: `/Users/user/Manifold-SAE/experiments/auto_75.py`.

## 1. Motivation

gamfit today fits

```
y = Φ(x) · β + ε,   penalty(β, ψ),   x observed
```

where `β` are smoothing-spline coefficients, `ψ` are kernel / anisotropy
hyper-parameters (currently flowing through `SpatialLogKappaCoords`), and `x`
is an observed covariate. The REML/LAML loop selects smoothing; an
IFT-warm-started nested optimizer moves `ψ` because `∂Φ/∂ψ` is computable
from the radial basis.

The proposal: promote some covariates from observed to **latent**:

```
y = Φ(t) · β + ε,   penalty(β, ψ),   t ∈ ℝ^(N×d) estimated per-row.
```

Mechanically this is the same problem as `ψ` already solves: we need
`∂Φ/∂t` per row, and that's structurally the same radial-basis-gradient
that gamfit's spatial machinery already computes — except differentiating
with respect to the *first* argument of the kernel rather than the
log-anisotropy. Everything downstream (IFT-warm-started Newton on the
inner system, REML for the outer ridges, persistent warm-start) is reused
verbatim.

Use case: **GP-LVM and principal-manifold inference** (Lawrence 2005,
Titsias 2010), entirely in gamfit primitives. This subsumes a non-trivial
set of workloads — manifold learning, latent-trait inference, identifiable
nonlinear ICA via iVAE-style auxiliary priors — that currently require
GPy / sklearn `IsoMap` or hand-rolled alternation (see this repo's
`experiments/auto_71.py` for a geodesic-LM gauge-unfixed example, and
`auto_74.py` for the alternation-then-ICA workaround).

We already ship `gaussian_reml_fit_positions{,_backward}` for the 1-D
case (B-spline positions with `grad_t` exposed). `LatentCoord` is the
*N-D Duchon* generalization of that same idea, lifted to a configuration
type rather than a bespoke function.

## 2. Math

### 2.1 Loss

For Gaussian-identity, profiling out σ²:

```
ℒ(t, β, ψ, λ) = ½‖y − Φ(t; ψ) β‖² + ½λ βᵀ S(ψ) β
                + R_id(t, u)                              # identifiability
                − ½ log|Σ(λ, ψ, t)|                       # REML log-det
```

`Φ(t; ψ)_{i,k} = K_ψ(t_i, c_k)` is the design row built from the latent
coordinate `t_i ∈ ℝ^d` and the basis centers `c_k`. For Duchon thin-plate
splines, `K_ψ` is radial: `K(r)` with `r = ‖t_i − c_k‖`, modulated by
ψ-anisotropy exactly as today.

### 2.2 Gradients

Let `g_i = ∂Φ_i / ∂t_i ∈ ℝ^(K × d)`. For a Duchon kernel
`Φ_{i,k} = φ(r_{ik})` we have `∂Φ_{i,k}/∂t_i = φ'(r_{ik}) · (t_i − c_k)/r_{ik}`,
the same radial-gradient routine that
`SpatialLogKappaCoords::apply_tospec` invokes for `∂Φ/∂ψ` (anisotropy
just rescales components of `(t_i − c_k)` before contraction).

The gradient w.r.t. `t_i` of the residual is row-local:

```
∂(½‖y − Φβ‖²) / ∂t_i = −(y_i − Φ_i β) · (g_i β)        # shape (d,)
```

so the per-row Hessian block (Gauss–Newton) is `(g_i β)(g_i β)ᵀ`. The
joint Hessian `H_tt` is block-diagonal in rows (huge — `N·d × N·d` — but
sparse-by-blocks), `H_tβ = − Φ_i ⊗ (g_i β) + …` is dense in β but only
row-local in t. This is the *same* structure gamfit's matrix-free CG
already exploits along the ψ direction (`hyper_dirs`).

### 2.3 Gauge symmetry — load-bearing caveat

The bare data-fit loss `½‖y − Φ(t) β‖²` is invariant under any
diffeomorphism `t ↦ φ(t)`, because we can absorb `φ` into a re-fit of
`β` against the relabeled basis. Concretely:

- Translation / rotation of `t` ↔ relabeling of centers.
- Smooth coordinate change ↔ basis transformation; β re-fits.

The penalty `βᵀ S β` *does* depend on coordinates (it penalizes
spline-curvature in `t`-space), but is invariant under isometries
(rotation + translation), which still leaves at least a `d(d+1)/2`-dim
gauge orbit per fit. **IFT is singular along this orbit** — the inner
Hessian has zero eigenvalues, so the nested optimizer's predictor step
blows up. `auto_71.py` in the companion repo demonstrates this: geodesic
LM hits R² = 0.640 but Procrustes disparity = 0.99 across seeds —
solutions wander freely along the gauge.

Three model-level fixes, all exposed as `LatentCoord` configurations:

**(a) `aux_prior`** — conditional prior `p(t | u)` with `u` observed
(iVAE; Hyvärinen 2019, Khemakhem et al. 2020). Add a regularizer
`R_id(t, u) = ½ τ · ‖t − g_φ(u)‖²` where `g_φ` is a small learned map
(ridge / shallow GAM). Different φ have different `g_φ(u)` log-prob,
so the orbit collapses to a single representative. This is the
**principled fix**; identifiability of the resulting `t` follows from
the iVAE result (assuming `u` varies enough across rows).

**(b) `orthogonality`** — penalize `‖TᵀT/N − I_d‖²_F` after centering
`T ← T − T̄`. Cheap to compute, kills the rotation + scale gauge, but
doesn't help with the diffeomorphism gauge for `d > 1`. Useful as a
warm-start regularizer or a cheap second-best.

**(c) `dim_selection`** — REML on `t` directly. Adding an extra latent
dimension costs an `O(N)` increase in free parameters; the marginal
likelihood penalizes this *unless* the data supports a higher-dim
manifold. This gives an automatic intrinsic-dim estimator that
matches the spirit of `twoNN` but inside the same loss the model is
trained against. Combines with (a): aux-prior identifies the active
dims, REML zeroes out the rest.

## 3. API sketch

### 3.1 Python

```python
import gamfit
from gamfit import LatentCoord

# Replaces an observed-x covariate with a learned t of dimension d.
t = LatentCoord(
    n=N,
    d=4,
    init="pca",                 # 'pca' | 'random' | array (N, d)
    bounds=(-3.0, 3.0),         # per-axis clamp (broadcast or per-d)
    aux_prior=dict(             # (a) iVAE-style
        u=rgb,                  # (N, p) observed auxiliary
        family="ridge",         # 'ridge' | 'duchon'
        strength=1.0,           # τ in the math
    ),
    # OR
    orthogonality=0.1,          # (b)
    # OR
    dim_selection="reml",       # (c)   ⇒ marginal-likelihood prunes d
)

model = gamfit.GAM(
    y ~ s(t, basis="duchon", m=2, nullspace_order="degree2")
       + s(observed_x, basis="duchon", m=2),
    data=df,
    latents={"t": t},
)
model.fit()                      # joint outer over (t, ψ, λ); REML inner
print(model.latents["t"].values) # (N, d) posterior mean
```

### 3.2 Rust

A new sibling of `SpatialLogKappaCoords`:

```rust
// src/terms/smooth.rs  (alongside SpatialLogKappaCoords)
pub(crate) struct LatentCoordValues {
    /// Flattened (N, d) latent matrix, row-major.
    values: Array2<f64>,
    /// Which design-term consumes this latent (one term per LatentCoord).
    term_index: usize,
    /// Identifiability mode.
    id_mode: LatentIdMode,
}

pub(crate) enum LatentIdMode {
    AuxPrior { u: Array2<f64>, g: AuxPredictor, tau: f64 },
    Orthogonality { weight: f64 },
    ReMlDimSelect,
}

impl LatentCoordValues {
    /// Mirrors SpatialLogKappaCoords::apply_tospec: re-materializes the
    /// design column block for the consuming term at the current t.
    pub(crate) fn apply_tospec(
        &self,
        spec: &TermCollectionSpec,
    ) -> Result<TermCollectionSpec, ...>;

    /// Mirrors the radial-gradient routine already used by ψ.
    /// Returns (N, K, d) jet for the consuming term.
    pub(crate) fn design_gradient_wrt_t(
        &self,
        spec: &TermCollectionSpec,
    ) -> Result<Array3<f64>, ...>;
}
```

The outer optimizer receives `t` as a new block of `hyper_dirs` (existing
machinery — `try_build_spatial_log_kappa_hyper_dirs` already builds these
for ψ; the new builder is `try_build_latent_coord_hyper_dirs`).

## 4. Mechanical reuse map

| Existing piece | Reused as | Notes |
| --- | --- | --- |
| `SpatialLogKappaCoords::apply_tospec` | template for `LatentCoordValues::apply_tospec` | both re-materialize Φ |
| `spatial_log_kappa_hyper_dirs_frominfo_list` | template for `latent_coord_hyper_dirs_frominfo_list` | both emit `DirectionalHyperParam` |
| `try_build_spatial_log_kappa_hyper_dirs` | template for `try_build_latent_coord_hyper_dirs` | dispatch at outer-fit assembly |
| `solver::estimate` IFT loop | unchanged | new hyper_dirs flow through |
| `solver::persistent_warm_start::last_ift_prediction_residual` | unchanged | same diagnostic |
| `solver::outer_strategy::EfsEval` | unchanged | same eval-and-gradient shape |
| `duchon_radial_jets` in `terms/basis.rs` | now also queried via `design_gradient_wrt_t` | same routine; first-arg derivative |
| `gaussian_reml_fit_positions_backward_impl` + `contract_position_gradient` | shipped 1D special case → generalize to ND | already returns `grad_t` |

**New pieces required:**
1. `LatentCoordValues` struct + the three id-mode variants.
2. A new term-spec attribute `latent: Option<LatentCoordRef>` on the
   smooth-term builder so a term consumes `t` instead of an observed col.
3. Outer-fit dispatch to build `hyper_dirs` for any registered
   `LatentCoord`s, identical pattern to ψ.
4. `dim_selection="reml"` path: when set, expand `d` by one and let
   REML decide; needs an outer loop over `d` analogous to the existing
   smoothing-grid search.

Crucially: **no new optimizer**, **no new IFT code**, **no new
matrix-free CG**. The ψ machinery covers it.

## 5. Test plan

Reproduce `auto_74`'s stacked CV R² = 0.608 on cogito-L40 color
centroids using *only* a `LatentCoord`-equipped GAM config — no
hand-rolled alternation, no FastICA, no per-fold ICA refit.

```
y = Z_top16  ∈ ℝ^(N × 16)
config:
  s(hue, periodic, m=2)               # observed
  + s(sat, val, m=2, ns=degree2)      # observed
  + s(t, m=2, ns=degree2,             # LATENT
       latent=LatentCoord(d=4, aux_prior=u=rgb))
```

Expected:
- CV R² ≥ 0.60 (matches auto_74 stacked).
- Procrustes disparity across seeds ≤ 0.2 (auto_71 was 0.99 unfixed).
- REML-selected `d` ∈ {3, 4} when `dim_selection="reml"` is enabled.

Empirical baseline observed in the prototype (`auto_75.py`, K_PC=16):
- observed-only (hue + sv) CV R² = 0.41 (the K_PC=16 ceiling, vs 0.32 at K_PC=64).
- LatentCoord d=4 with τ=0.5 aux-prior on RGB: CV R² ≈ 0.50 (early).
- The prototype's pure-Python out-of-fold `t_test` refinement is the
  bottleneck for full parity with auto_74; the Rust LatentCoord with
  IFT-warm-started `grad_t` would close this gap with better step
  control (and at production speed).

The companion prototype `experiments/auto_75.py` simulates this in
pure Python (using gamfit's existing primitives + a hand-rolled outer
on `t`) to validate the loss surface and identifiability gain *before*
the Rust work lands.

## 6. Open questions

1. **`d` initialization for `dim_selection`**: brute-force `d` sweep,
   or jet-style continuous relaxation (one column at a time, scaled by
   a regularizer that REML drives to zero)?
2. **Auxiliary-prior choice for `g_φ`**: ridge sufficient, or do we
   need a full nested GAM for the conditional mean?
3. **Periodic latent axes**: trivial to support (`periodic_per_axis`
   already exists) but exposes the question of *which* axes the user
   wants periodic. Default false; let user opt in.
4. **Backprop through latent**: should `LatentCoord` be backprop-visible
   to PyTorch (`gamfit.torch`) so the latent participates in upstream
   gradients? Probably yes — mirrors the existing `positions_backward`.
5. **Scaling**: for `N = 10⁶`, `(N, d)` is small but the per-row
   row-local Hessian needs the matrix-free CG path. Confirm this falls
   out of the existing infrastructure without new code.
